#!/usr/bin/env bash
# Check that an X-HEEP tree is exactly what xheep/setup.sh builds:
#   - every file the patch series touches equals base + xheep/patches/*.patch,
#   - every generated file equals xheep/generated.sha256 (the output of make mcu-gen in a fresh setup), or
#     xheep/generated.<config name>.sha256 for a tree made with another configuration (<tree>/.hopcim-cfg),
#   - hw/ip/hopcim and the firmware files are links into this repository (not copies),
#   - nothing else differs from the base commit (no change that only lives in the X-HEEP tree).
# usage: xheep/check.sh [<tree>]              default: ~/APlus_Project/planenest/hw/x-heep (the working tree)
#        xheep/check.sh --record <tree>       write that hash file from a tree setup.sh just built
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RECORD=0
if [ "${1:-}" = "--record" ]; then RECORD=1; shift; fi
TREE=$(realpath "${1:-$HOME/APlus_Project/planenest/hw/x-heep}")
CFG=$(tree_cfg "$TREE")
GEN=$(generated_file "$CFG")
cd "$TREE"

# files the generator wrote in this tree (newer than the marker setup.sh leaves just before make mcu-gen)
generated_now() {
    find . \( -path ./.git -o -path ./build -o -path './build_*' -o -path ./.venv -o -path ./sw/build \
              -o -path ./hw/ip/hopcim -o -name __pycache__ \) -prune -o \
         -type f -newer .hopcim-gen-start ! -name '*.log' ! -name '*.cache' ! -name '.hopcim-gen-start' -print \
        | sed 's|^\./||' | LC_ALL=C sort
}

if [ $RECORD = 1 ]; then
    [ -f .hopcim-gen-start ] || die "$TREE has no .hopcim-gen-start: record only from a tree setup.sh built"
    generated_now | while read -r f; do sha256sum "$f"; done > "$GEN"
    echo "check: recorded $(wc -l < "$GEN") generated files in $GEN"
    exit 0
fi

fail=0
bad() { echo "FAIL  $*"; fail=1; }

# base
git cat-file -e "$XHEEP_COMMIT^{commit}" 2> /dev/null || die "$TREE does not hold X-HEEP $XHEEP_COMMIT"
git merge-base --is-ancestor "$XHEEP_COMMIT" HEAD || bad "HEAD is not on X-HEEP ${XHEEP_COMMIT:0:12}"

# series files against base + patches
ref=$(mktemp -d)
trap 'rm -rf "$ref"' EXIT
git archive "$XHEEP_COMMIT" | tar -x -C "$ref"
for p in "$XHEEP_HERE"/patches/*.patch; do (cd "$ref" && git apply --whitespace=nowarn "$p") || bad "$(basename "$p") does not apply to the base"; done
ns=0
while read -r f; do
    ns=$((ns + 1))
    if [ ! -e "$f" ]; then bad "$f: missing (the series adds it)"
    elif ! cmp -s "$f" "$ref/$f"; then bad "$f: differs from base + series"; fi
done < <(series_files)

# generated files
ng=0
if [ -f "$GEN" ]; then
    ng=$(wc -l < "$GEN")
    while read -r line; do bad "generated ${line%: FAILED*}: differs from xheep/$(basename "$GEN") (rerun make mcu-gen X_HEEP_CFG=$CFG?)"; done \
        < <(sha256sum -c --quiet "$GEN" 2>&1 | grep -E 'FAILED|No such' || true)
else
    bad "xheep/$(basename "$GEN") is missing (xheep/check.sh --record <tree from setup.sh>)"
fi

# links into this repository
[ "$(realpath hw/ip/hopcim 2> /dev/null)" = "$(realpath "$HOPCIM")" ] || bad "hw/ip/hopcim does not lead to $HOPCIM"
nl=0
for app in "$HOPCIM"/sw/applications/*/; do
    a=$(basename "$app")
    for f in "$app"*; do
        t="sw/applications/$a/$(basename "$f")"
        nl=$((nl + 1))
        if [ -L "$t" ] && [ "$(realpath "$t")" = "$(realpath "$f")" ]; then :
        elif [ -f "$t" ] && cmp -s "$t" "$f"; then bad "$t: a copy, not a link (an edit there would not reach the repository)"
        else bad "$t: not a link to $f"; fi
    done
    for t in "sw/applications/$a"/*; do
        [ -e "$app$(basename "$t")" ] || bad "$t: not in the repository's sw/applications/$a"
    done
done

# drift: anything else that differs from the base commit
tmpl=$(mktemp)
{ series_files; [ -f "$GEN" ] && sed 's/^[0-9a-f]*  //' "$GEN"; } | LC_ALL=C sort -u > "$tmpl"
while read -r f; do
    grep -qxF "$f" "$tmpl" || bad "$f: changed from the base, but not by the series or the generator"
done < <(git diff --name-only "$XHEEP_COMMIT")
while read -r f; do
    case "$f" in build_*/*) continue;; esac
    ours=0
    while read -r o; do case "$f" in "$o"*) ours=1;; esac; done < <(ours_untracked)
    [ $ours = 1 ] || grep -qxF "$f" "$tmpl" || bad "$f: untracked, not from the series"
done < <(git ls-files --others --exclude-standard)
rm -f "$tmpl"

if [ $fail = 0 ]; then
    echo "check: PASS  $TREE = X-HEEP ${XHEEP_COMMIT:0:12} + series ($ns files) + generator ($ng files, $CFG); $nl firmware links and hw/ip/hopcim lead to $HOPCIM; no other change"
else
    echo "check: FAIL  ($TREE)"
fi
exit $fail
