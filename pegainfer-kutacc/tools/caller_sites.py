#!/usr/bin/env python3
"""List where production calls each KuTACC function, next to what this crate wraps.

Three findings in a row came from reading the *caller* rather than the header:

* the indexer's `output_coord` was hardcoded here while production selects it per
  call, and it turned out the two modes feed two different operators;
* `flash_attention`'s scratch extents were derived from the kernel and then
  confirmed by production's own allocation, and its `is_kv_packed` turned out dead
  rather than unexercised;
* sparse MLA decode is called with `seqlen_q > 1` in production while the gates
  only used `seqlen_q = 1`.

This script makes that check mechanical. Run it with the SGLang tree available:

    python3 tools/caller_sites.py [/path/to/SGLang_920F]

It prints, per KuTACC function: how many times production calls it, and where. A
wrapped function with **no** call site is worth a second look — either it is not on
the LLM path, or the call site is somewhere this search does not look. A function
production calls that this crate does **not** wrap is a candidate gap.
"""

import pathlib
import re
import sys

DEFAULT_TREE = pathlib.Path("/home/share/fengguangnan/sibow/mlsysu/SGLang_920F")
KUNPENG_SUBDIR = "sgl-kernel/csrc/cpu/cpu_kunpeng"
SHIMS = pathlib.Path(__file__).resolve().parent.parent / "csrc/shim"

# Calls that are plumbing rather than operators.
IGNORED = {
    "parallel_for",
    "parallel_barrier",
    "get_thread_id",
    "get_thread_num",
    "attention_schedule",
    "FlashMLAMetaHandle",
}


def wrapped_functions() -> set[str]:
    names = set()
    for path in SHIMS.glob("*.cpp"):
        names.update(re.findall(r"kutacc::([a-zA-Z_0-9]+)\s*\(", path.read_text()))
    # The shims also name types and helpers; keep anything that looks like a call.
    return {n for n in names if n not in IGNORED}


def main() -> int:
    tree = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TREE
    root = tree / KUNPENG_SUBDIR
    if not root.is_dir():
        print(f"no Kunpeng tree at {root}", file=sys.stderr)
        return 1

    sites: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if path.suffix not in (".cpp", ".h", ".cc", ".hpp"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for match in re.finditer(r"kutacc::([a-zA-Z_0-9]+)\s*\(", text):
            name = match.group(1)
            if name in IGNORED:
                continue
            line = text[: match.start()].count("\n") + 1
            sites.setdefault(name, []).append(f"{path.relative_to(tree)}:{line}")

    wrapped = wrapped_functions()
    print(f"production call sites found: {len(sites)} distinct functions")
    print(f"this crate wraps:            {len(wrapped)} kutacc calls\n")

    print("wrapped and called by production:")
    for name in sorted(wrapped & sites.keys()):
        where = ", ".join(sorted(sites[name])[:3])
        more = "" if len(sites[name]) <= 3 else f" (+{len(sites[name]) - 3} more)"
        print(f"    {name:38s} {len(sites[name]):2d}x  {where}{more}")

    only_wrapped = sorted(wrapped - sites.keys())
    if only_wrapped:
        print("\nwrapped but not called from the Kunpeng tree:")
        for name in only_wrapped:
            print(f"    {name}")

    only_called = sorted(set(sites) - wrapped)
    if only_called:
        print("\ncalled by production but not wrapped here (candidate gaps):")
        for name in only_called:
            print(f"    {name:38s} {len(sites[name]):2d}x  {sorted(sites[name])[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
