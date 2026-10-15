#!/usr/bin/env bash
# Native dependency manifest for the Rust inference framework.
#
# Two jobs:
#   check        (default) verify every pinned artifact still matches its hash
#   regen        re-hash from the current environment and rewrite manifest.tsv
#   stage DIR    copy the runtime set into DIR (run-time bundle, no HPCKit)
#
# Why this exists: KuTACC's filename `25.1.RC1` does NOT identify the code.
# Two files with that exact name in this workspace have different sha256
# (`9eeaf3ee...` packaged vs `fd32ce88...` DSA), so a path is not an identity.
# Everything the framework links or loads is pinned by hash here instead.
#
# Roots, overridable:
#   R  SGLang_920F checkout          (default below)
#   H  HPCKit root                   (default below)
#   L  packaged deployment `_internal` (default $R/scripts/cpu_kunpeng/pyinstall/dist/sglang_server_tp0/_internal)
#   D  DSA KuTACC install            (default $R/run_artifacts/dsa-graph-20261008/kutacc/install)
#   KUTACC_DIR  this crate's root    (default the script's parent)
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
R=${R:-/home/share/fengguangnan/sibow/mlsysu/SGLang_920F}
H=${H:-/home/share/fengguangnan/lvtx/HPCKit/HPCKit/26.1.RC1}
L=${L:-$R/scripts/cpu_kunpeng/pyinstall/dist/sglang_server_tp0/_internal}
D=${D:-$R/run_artifacts/dsa-graph-20261008/kutacc/install}
KUTACC_DIR=${KUTACC_DIR:-$(cd -- "$here/.." && pwd -P)}
manifest="$here/manifest.tsv"

# name | role | path | requirement
# requirement: runtime = must ship with the binary, build = compile-time only,
#              alt = known sibling artifact, tracked but not shipped
entries() {
    cat <<EOF
libkutacc.so.25.1.RC1	compute ops + comm (packaged deployment copy)	$L/libkutacc.so.25.1.RC1	runtime
libkutacc.so.25.1.RC1-dsa	compute ops + comm (DSA build; same filename, different code)	$D/lib/libkutacc.so.25.1.RC1	alt
libkupl.so.1	worker scheduling, parallel, sync	$L/libkupl.so.1	runtime
libnuma.so.1	NUMA / HBM placement	$L/libnuma.so.1	runtime
libomp.so	OpenMP runtime (KuPL dependency)	$L/libomp.so	runtime
libibverbs.so.1	RDMA verbs	$L/libibverbs.so.1	runtime
libhns.so.1	RDMA provider (Huawei hns)	$L/libhns.so.1	runtime
libmlx5.so.1	RDMA provider (Mellanox)	$L/libmlx5.so.1	runtime
libsdma_dk.so	SDMA data-movement SDK	$L/libsdma_dk.so	optional
libatomic.so.1	atomic operations	$L/libatomic.so.1	runtime
libstdc++.so.6	C++ runtime	$L/libstdc++.so.6	runtime
libgcc_s.so.1	compiler runtime	$L/libgcc_s.so.1	runtime
kupl.h	KuPL header (compile-time only)	$R/deployment/kupl-install/include/kupl.h	build
kutacc.h-vendored	KuTACC header in this repo (git HEAD 1502ca3)	$KUTACC_DIR/csrc/kutacc/include/kutacc.h	build
kutacc.h-dsa	KuTACC header shipped with the DSA build	$D/include/kutacc.h	build
SOURCE_SNAPSHOT.sha256	upstream build snapshot manifest	$KUTACC_DIR/csrc/kutacc/SOURCE_SNAPSHOT.sha256	build
EOF
}

# RDMA providers libkutacc.so itself declares as NEEDED.
needed_providers() {
    local so=$L/libkutacc.so.25.1.RC1
    [[ -r $so ]] || return 0
    readelf -d "$so" 2>/dev/null | grep NEEDED | grep -oE '\[[^]]+\]' | tr -d '[]' | while read -r dep; do
        case $dep in
            libkutacc*|libkupl*|libnuma*|libibverbs*|libstdc++*|libm.so*|libc.so*|ld-linux*) continue ;;
        esac
        printf '%s\tRDMA provider required by libkutacc.so\t%s/%s\truntime\n' "$dep" "$L" "$dep"
    done
}

cmd_regen() {
    local tmp; tmp=$(mktemp)
    {
        entries | while IFS=$'\t' read -r name role path req; do
            if [[ -f $path ]]; then
                printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$role" "$path" "$(sha256sum "$path" | cut -d' ' -f1)" "$req"
            else
                printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$role" "$path" MISSING "$req"
            fi
        done
        needed_providers | while IFS=$'\t' read -r name role path req; do
            if [[ -f $path ]]; then
                printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$role" "$path" "$(sha256sum "$path" | cut -d' ' -f1)" "$req"
            else
                printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$role" "$path" MISSING "$req"
            fi
        done
    } > "$tmp"
    mv "$tmp" "$manifest"
    printf 'wrote %s (%s entries)\n' "$manifest" "$(wc -l < "$manifest")"
}

cmd_check() {
    [[ -r $manifest ]] || { echo "no manifest; run: $0 regen" >&2; exit 2; }
    local bad=0 missing=0 ok=0
    while IFS=$'\t' read -r name role path hash req; do
        [[ -n ${name:-} ]] || continue
        if [[ ! -f $path ]]; then
            printf 'MISSING  %-28s %s\n' "$name" "$path"; missing=$((missing + 1)); continue
        fi
        if [[ $hash == MISSING ]]; then
            printf 'MISSING  %-28s (hashed when absent)\n' "$name"; missing=$((missing + 1)); continue
        fi
        local actual; actual=$(sha256sum "$path" | cut -d' ' -f1)
        if [[ $actual == "$hash" ]]; then
            ok=$((ok + 1))
        else
            printf 'CHANGED  %-28s %s\n  pinned %s\n  actual %s\n' "$name" "$path" "$hash" "$actual"
            bad=$((bad + 1))
        fi
    done < "$manifest"
    printf '\n%s ok, %s changed, %s missing\n' "$ok" "$bad" "$missing"
    [[ $bad -eq 0 && $missing -eq 0 ]]
}

cmd_stage() {
    local dest=${1:?usage: $0 stage DIR}
    mkdir -p "$dest"
    local n=0
    while IFS=$'\t' read -r name role path hash req; do
        [[ -n ${name:-} ]] || continue
        [[ $req == runtime ]] || continue
        [[ -f $path ]] || { echo "skip (absent): $name" >&2; continue; }
        cp -L "$path" "$dest/$(basename "$path")"
        n=$((n + 1))
    done < "$manifest"
    cp -L "$manifest" "$dest/manifest.tsv"
    printf 'staged %s runtime libraries into %s\n' "$n" "$dest"
    printf 'run with:  LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$dest"
}

case ${1:-check} in
    check) cmd_check ;;
    regen) cmd_regen ;;
    stage) cmd_stage "${2:-}" ;;
    *) echo "usage: $0 [check|regen|stage DIR]" >&2; exit 2 ;;
esac
