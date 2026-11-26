# Source this in every terminal used for X-HEEP work:
#   source ~/APlus_Project/planenest/hw/xheep_env.sh
# Selects the SiFive RISC-V GCC, the pinned Verilator 5.040 (ahead of the
# 5.047-devel link in ~/.local/bin) and the X-HEEP Python venv.

export RISCV_XHEEP=/home/hansen0411/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14
export COMPILER_PREFIX=riscv64-unknown-
export PATH=$HOME/tools/verilator/5.040/bin:$RISCV_XHEEP/bin:/home/Vivado/2024.1/bin:$PATH
# The Verilator testharness links -lelf; libelf-dev is not installed, so link
# against the runtime libelf.so.1 through a private symlink.
export LIBRARY_PATH=$HOME/tools/libelf-link${LIBRARY_PATH:+:$LIBRARY_PATH}

XHEEP_DIR=$HOME/APlus_Project/planenest/hw/x-heep
if [ -f "$XHEEP_DIR/.venv/bin/activate" ]; then
    source "$XHEEP_DIR/.venv/bin/activate"
fi

echo "X-HEEP env: $(verilator --version | cut -d' ' -f1-2), $(riscv64-unknown-elf-gcc --version | head -1)"
