# Shared by xheep/setup.sh, check.sh and build.sh (sourced, not run).
XHEEP_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# the repository the tree's links must lead to: this one, unless XHEEP_CHECK_HOPCIM names another clone (the freeze
# self-test checks the working tree from a throw-away clone, whose patches it compares, while the links lead here)
HOPCIM=${XHEEP_CHECK_HOPCIM:-$(dirname "$XHEEP_HERE")}
# shellcheck source=base.env
. "$XHEEP_HERE/base.env"

die() { echo "error: $*" >&2; exit 1; }

# The generator configuration a tree was made with: setup.sh records it in <tree>/.hopcim-cfg (a tree made
# before that file existed has the default).
tree_cfg() { cat "${1:-.}/.hopcim-cfg" 2> /dev/null || echo "$XHEEP_CFG"; }

# Where the hashes of the generator's output for a configuration are kept: generated.sha256 for the
# default, generated.<config name>.sha256 for another one.
generated_file() {
    if [ "$1" = "$XHEEP_CFG" ]; then echo "$XHEEP_HERE/generated.sha256"
    else echo "$XHEEP_HERE/generated.$(basename "$1" .py).sha256"; fi
}

# Tool paths (RISC-V GCC, Verilator, Vivado, verible). XHEEP_ENV names a script to source;
# the default is the one this project has always used, if it exists.
xheep_tools() {
    local env=${XHEEP_ENV:-$HOME/APlus_Project/planenest/hw/xheep_env.sh}
    if [ -f "$env" ]; then
        # shellcheck disable=SC1090
        . "$env" > /dev/null
    fi
}

# Files the patch series touches, one per line (from the patches' own headers).
series_files() {
    sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "$XHEEP_HERE"/patches/*.patch | sort -u
}

# Untracked paths HopCIM puts into an X-HEEP tree: the links, the build roots, logs.
ours_untracked() {
    local a
    echo "hw/ip/hopcim"
    echo ".hopcim-gen-start"
    echo ".hopcim-cfg"
    echo ".hopcim-build.lock"
    echo ".venv"              # when setup.sh links it (.gitignore only covers a directory)
    for a in "$HOPCIM"/sw/applications/*/; do echo "sw/applications/$(basename "$a")/"; done
}
