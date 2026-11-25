#!/usr/bin/env bash
# Build the X-HEEP tree HopCIM runs on, from nothing but this repository and upstream X-HEEP:
#   1. X-HEEP at the pinned commit (xheep/base.env),
#   2. the patch series xheep/patches/*.patch as commits on a branch 'hopcim' (git am),
#      checked against the recorded result tree,
#   3. links into this repository: hw/ip/hopcim -> the RTL (FuseSoC finds hopcim.core through it),
#      sw/applications/<app>/<file> -> sw/applications/<app>/<file> here (file by file, so the
#      firmware's relative includes such as ../../../hw/ip/hopcim/sw/cim.h keep working),
#   4. the generator: make mcu-gen X_HEEP_CFG=<config> (default configs/kws_nexys.py; recorded in
#      <dest>/.hopcim-cfg, which check.sh and build.sh read),
#   5. xheep/check.sh on the result.
#
# usage: xheep/setup.sh <dest> [<config>]   <dest> must not exist yet (or be an unmodified clone of X-HEEP);
#        <config> defaults to $XHEEP_CFG; $XHEEP_CFG_256K (patch 0005) gives 256 KiB of RAM
# env:   XHEEP_URL   where to clone from (default: upstream). Any clone holding the pinned commit will
#                    do: the commit hash pins the content.
#        XHEEP_VENV  a Python venv with X-HEEP's requirements, linked as <dest>/.venv
#                    (default: <dest>/.venv if present, else `make venv`, which needs the network)
#        XHEEP_ENV   script to source for tool paths (default ~/APlus_Project/planenest/hw/xheep_env.sh)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DEST=${1:?usage: xheep/setup.sh <dest> [<config>]}
CFG=${2:-$XHEEP_CFG}
DEST=$(realpath -m "$DEST")
case "$DEST/" in "$HOPCIM"/*) die "put the X-HEEP tree outside $HOPCIM (FuseSoC would scan it again through hw/ip/hopcim)";; esac

# 1. X-HEEP at the pinned commit
if [ ! -e "$DEST" ]; then
    git clone -q --no-checkout "${XHEEP_URL:-$XHEEP_UPSTREAM}" "$DEST"
    cd "$DEST"
    git cat-file -e "$XHEEP_COMMIT^{commit}" 2> /dev/null || git fetch -q origin "$XHEEP_COMMIT"
    git checkout -q -f -B hopcim "$XHEEP_COMMIT"
else
    cd "$DEST"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "$DEST has local changes; use a new directory"
    git cat-file -e "$XHEEP_COMMIT^{commit}" 2> /dev/null || git fetch -q origin "$XHEEP_COMMIT"
    git checkout -q -B hopcim "$XHEEP_COMMIT"
fi
[ "$(git rev-parse HEAD)" = "$XHEEP_COMMIT" ] || die "could not check out $XHEEP_COMMIT"

# 2. the patch series
git -c user.name="HopCIM setup" -c user.email=setup@hopcim.invalid am -q --whitespace=nowarn --committer-date-is-author-date \
    "$XHEEP_HERE"/patches/*.patch
[ "$(git rev-parse 'HEAD^{tree}')" = "$XHEEP_SERIES_TREE" ] \
    || die "the patched tree is $(git rev-parse 'HEAD^{tree}'), expected $XHEEP_SERIES_TREE (xheep/base.env)"
echo "setup: X-HEEP ${XHEEP_COMMIT:0:12} + $(ls "$XHEEP_HERE"/patches/*.patch | wc -l) patches, tree ${XHEEP_SERIES_TREE:0:12} as expected"

# 3. links into this repository (relative, so the pair can be moved together)
ln -sfn "$(realpath --relative-to="$DEST/hw/ip" "$HOPCIM")" hw/ip/hopcim
n=0
for app in "$HOPCIM"/sw/applications/*/; do
    a=$(basename "$app")
    mkdir -p "sw/applications/$a"
    for f in "$app"*; do
        ln -sfn "$(realpath --relative-to="$DEST/sw/applications/$a" "$f")" "sw/applications/$a/$(basename "$f")"
        n=$((n + 1))
    done
done
ex=.git/info/exclude
grep -q '^# HopCIM' "$ex" 2> /dev/null || { echo "# HopCIM (xheep/setup.sh): links and build roots"; ours_untracked | sed 's|^|/|'; echo "/build_*/"; } >> "$ex"
echo "setup: hw/ip/hopcim -> $HOPCIM; $n firmware files linked"

# 4. the generator
if [ -n "${XHEEP_VENV:-}" ]; then
    ln -sfn "$XHEEP_VENV" .venv
elif [ ! -e .venv ]; then
    make venv
fi
[ -f "$CFG" ] || die "no generator configuration $CFG in the patched tree"
echo "$CFG" > .hopcim-cfg
xheep_tools
touch .hopcim-gen-start
sleep 1    # file times have 1-s resolution on some filesystems; check.sh --record uses "newer than"
make mcu-gen X_HEEP_CFG="$CFG" > mcu-gen.log 2>&1 || { tail -20 mcu-gen.log; die "make mcu-gen failed (mcu-gen.log)"; }
echo "setup: make mcu-gen X_HEEP_CFG=$CFG done"

# 5. check
if [ "${HOPCIM_RECORD:-0}" = 1 ]; then "$XHEEP_HERE/check.sh" --record "$DEST"; fi
exec "$XHEEP_HERE/check.sh" "$DEST"
