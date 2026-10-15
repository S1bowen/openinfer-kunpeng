#!/usr/bin/env python3
"""Audit the runtime dependency closure of the prebuilt KuTACC stack.

Answers the objective's dependency questions with a measurement rather than a
list someone typed. Run from the crate root:

    python3 tools/closure_audit.py

It stages the manifest's `runtime` rows into a scratch directory, then walks
`NEEDED` transitively with `readelf`, and reports:

* how many libraries were staged and how many distinct `NEEDED` names the closure
  contains;
* whether any library the objective says is **no longer needed** appears anywhere
  in that closure (LibTorch, libc10, KBLAS, libkpmalloc, MPI);
* whether the **conditionally** needed ones appear (SDMA, libatomic);
* which `NEEDED` names the staging does **not** provide, i.e. what has to come from
  the system rather than the bundle.

This is the check behind the claim "run Rust + a prebuilt native library without a
full HPCKit": if a name shows up in the last list that is not a base system
library, the bundle is not self-sufficient.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

MANIFEST = pathlib.Path("native/manifest.tsv")
RUNTIME = "runtime"

# The objective states these are no longer needed because PyTorch is gone. If any
# appears in the closure, that statement is wrong.
NO_LONGER_NEEDED = ("torch", "c10", "kblas", "kpmalloc", "mpi")

# The objective marks these as needed only when their path is used.
CONDITIONAL = ("sdma", "atomic")


def needed_of(path: pathlib.Path) -> set[str]:
    out = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        if "NEEDED" in line:
            names.add(line.split("[")[1].rstrip("]"))
    return names


def main() -> int:
    if not MANIFEST.is_file():
        print(f"no manifest at {MANIFEST}", file=sys.stderr)
        return 1

    rows = []
    for line in MANIFEST.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 5 and fields[4] == RUNTIME:
            rows.append((fields[0], pathlib.Path(fields[2])))

    staged: dict[str, pathlib.Path] = {}
    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="closure-") as tmp:
        stage = pathlib.Path(tmp)
        for name, source in rows:
            if not source.is_file():
                missing.append(name)
                continue
            destination = stage / source.name
            if not destination.exists():
                shutil.copy(source, destination)
            staged[name] = destination

        # Walk NEEDED transitively, following names the staging provides.
        seen: set[pathlib.Path] = set()
        queue = list(staged.values())
        edges: dict[str, set[str]] = {}
        while queue:
            library = queue.pop()
            if library in seen:
                continue
            seen.add(library)
            names = needed_of(library)
            edges[library.name] = names
            for name in names:
                candidate = stage / name
                if candidate.exists():
                    queue.append(candidate)

        all_needed = sorted({n for names in edges.values() for n in names})
        unstaged = [n for n in all_needed if not (stage / n).exists()]

    distinct_files = len({p.name for p in staged.values()})
    row_names = [name for name, _ in rows]
    duplicates = sorted({n for n in row_names if row_names.count(n) > 1})

    print(f"manifest runtime rows: {len(rows)}   distinct files: {distinct_files}")
    if duplicates:
        print(f"  rows sharing a file name (check they are the same binary): {duplicates}")
    if missing:
        print(f"  rows whose file is absent here: {missing}")
    print(f"staged libraries: {len(seen)}   distinct NEEDED names: {len(all_needed)}")

    forbidden = [n for n in all_needed if any(key in n.lower() for key in NO_LONGER_NEEDED)]
    print(f"no-longer-needed libraries in the closure: {forbidden or 'none'}")

    conditional = [n for n in all_needed if any(key in n.lower() for key in CONDITIONAL)]
    print(f"conditional libraries in the closure:      {conditional or 'none'}")

    print(f"NEEDED not provided by the staging ({len(unstaged)}) — these must come from the system:")
    for name in unstaged:
        print(f"    {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
