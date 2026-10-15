#!/usr/bin/env bash
# Compare the vendored tree against upstream's SOURCE_SNAPSHOT.sha256.
#
# The manifest is a point-in-time record of the deployed GLM-5.2 PP2 build, not
# a live property of the upstream branch. This script reports the delta instead
# of asserting equality, so drift is visible rather than assumed away.
#
# Usage: bash verify_against_snapshot.sh
set -uo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/kutacc" || exit 2
manifest=SOURCE_SNAPSHOT.sha256
[[ -r $manifest ]] || { echo "missing $manifest" >&2; exit 2; }

matched=0; mismatch=0; missing=0; outside=0
: > /tmp/kutacc_snapshot_mismatch.txt

while read -r hash path; do
    case "$hash" in \#*|"") continue ;; esac
    case "$path" in
        src/*|include/*) ;;
        *) outside=$((outside + 1)); continue ;;
    esac
    if [[ ! -f $path ]]; then
        missing=$((missing + 1)); printf 'MISSING  %s\n' "$path" >> /tmp/kutacc_snapshot_mismatch.txt
    elif [[ $(sha256sum "$path" | cut -d' ' -f1) == "$hash" ]]; then
        matched=$((matched + 1))
    else
        mismatch=$((mismatch + 1)); printf 'MISMATCH %s\n' "$path" >> /tmp/kutacc_snapshot_mismatch.txt
    fi
done < "$manifest"

manifest_paths=$(grep -E '  (src|include)/' "$manifest" | awk '{print $2}' | sort)
extra=$(comm -13 <(printf '%s\n' "$manifest_paths") <(find src include -type f | sort))

printf 'manifest entries under src/ or include/: %d\n' "$((matched + mismatch + missing))"
printf '  matched  %d\n  mismatch %d\n  missing  %d\n' "$matched" "$mismatch" "$missing"
printf 'manifest entries outside src|include: %d (build system, docs)\n' "$outside"
printf 'files present but not in the manifest: %d\n' "$(printf '%s\n' "$extra" | grep -c . || true)"
printf '\n--- mismatched/missing ---\n'
cat /tmp/kutacc_snapshot_mismatch.txt 2>/dev/null
printf '\n--- present but unlisted ---\n%s\n' "$extra"
