#!/usr/bin/env python3
"""Stage a CPU-only SDK and runtime closure on the target compute node.

Uses the explicitly selected KuPL build, not a filename-based deployment guess.
System ABI libraries (loader/libc/libm/libpthread/libdl/librt) remain on the host.
No downloads, package installation, PyTorch or CUDA dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE_SYSTEM = re.compile(r"^(ld-linux|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|linux-vdso)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hpckit', type=Path)
    p.add_argument('--kupl', type=Path, help='override KuPL root; default: HPCKit/kupl/bisheng/release')
    p.add_argument('--sdma', type=Path, help='optional directory containing libsdma_dk.so')
    p.add_argument('--sdma-include', type=Path, default=Path('/usr/include'), help='SDMA header directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true', help='verify an existing bundle without consulting its source paths')
    a = p.parse_args()
    if a.verify:
        out = a.output.resolve(strict=True)
        manifest = json.loads((out/'manifest.json').read_text())
        env = dict(os.environ, LD_LIBRARY_PATH=str(out/'lib'))
        for record in manifest['artifacts']:
            artifact = out/record['file']
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError('hash mismatch: '+str(artifact))
            if record['file'].startswith('lib/'):
                result = subprocess.run(['ldd', str(artifact)], text=True, capture_output=True, env=env, check=True)
                if 'not found' in result.stdout:
                    raise RuntimeError(result.stdout)
        print(f"Verified {len(manifest['artifacts'])} artifact hashes and all ELF dependency resolutions.")
        return
    if not a.hpckit:
        p.error('--hpckit is required for staging')
    hpckit = a.hpckit.resolve(strict=True)
    kupl = (a.kupl or hpckit/'kupl/bisheng/release').resolve(strict=True)
    out = a.output.absolute()
    if out.exists() and any(out.iterdir()):
        p.error('output must be absent or empty (never overwrite a live runtime)')
    compiler_lib = hpckit/'compiler/bisheng/lib/aarch64-unknown-linux-gnu'
    search = [kupl/'lib', compiler_lib, Path('/usr/lib64'), Path('/lib64')]
    if a.sdma:
        search.insert(0, a.sdma.resolve(strict=True))
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, search)))
    lib = out/'lib'
    include = out/'include'
    lib.mkdir(parents=True)
    include.mkdir()
    records = []
    done = {}
    system = set()

    def copy(source, dest, role):
        real = source.resolve(strict=True)
        digest = hashlib.sha256(real.read_bytes()).hexdigest()
        if dest.name in done:
            if done[dest.name] != digest:
                raise RuntimeError('different libraries share name: '+dest.name)
            return False
        shutil.copy2(real, dest)
        done[dest.name] = digest
        records.append(dict(file=str(dest.relative_to(out)), source=str(real), sha256=digest, role=role))
        return True

    for header in sorted((kupl/'include').glob('*.h')):
        copy(header, include/header.name, 'build-header')
    if not (include/'kupl.h').exists():
        raise RuntimeError('selected KuPL SDK lacks kupl.h')
    queue = [(kupl/'lib/libkupl.so', 'kupl')]
    # Include the native link dependencies, not just KuPL's transitive closure.
    for name in ['libnuma.so', 'libibverbs.so', 'libomp.so']:
        choices = [d/name for d in search if (d/name).is_file()]
        if not choices:
            raise RuntimeError('development library missing: '+name)
        queue.append((choices[0], 'native-link'))
    # Providers are dynamically discovered by libibverbs, so ldd alone misses them.
    provider_dir = Path('/usr/lib64/libibverbs')
    if provider_dir.is_dir():
        queue.extend((f, 'verbs-provider') for f in sorted(provider_dir.glob('*.so')))
    if a.sdma:
        copy(a.sdma_include/'mdk_sdma.h', include/'mdk_sdma.h', 'build-header')
        queue.append((a.sdma/'libsdma_dk.so', 'sdma'))
    while queue:
        source, role = queue.pop(0)
        if not copy(source, lib/source.name, role):
            continue
        dynamic = subprocess.check_output(['readelf', '-d', str(source)], text=True)
        soname = re.search(r'\(SONAME\).*\[(.*?)\]', dynamic)
        if soname and soname[1] != source.name:
            copy(source, lib/soname[1], role)
        result = subprocess.run(['ldd', str(source)], text=True, capture_output=True, env=env, check=True)
        if 'not found' in result.stdout:
            raise RuntimeError(str(source)+' has unresolved dependencies:\n'+result.stdout)
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields:
                continue
            name = fields[0]
            if BASE_SYSTEM.match(name) or name.startswith('/') and 'ld-linux' in name:
                system.add(name)
                continue
            if len(fields) >= 3 and fields[1] == '=>' and fields[2].startswith('/'):
                if any(s in name for s in ['torch', 'python', 'c10', 'cuda']):
                    raise RuntimeError('unexpected framework dependency: '+name)
                queue.append((Path(fields[2]), 'runtime'))
    report = dict(kupl_root=str(kupl), hpckit_root=str(hpckit), artifacts=records, system=sorted(system))
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Staged {len(records)} artifacts to {out}; manifest.json records identities and system ABI requirements.')
    print('PEGAINFER_KUPL_ROOT and PEGAINFER_NATIVE_LIB_DIR can now point to this root and its lib directory.')


if __name__ == '__main__':
    main()
