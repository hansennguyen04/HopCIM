// hopcim_net with exactly one clip on every target, so that the board build runs the same C code
// as the simulation build (which also runs one clip).
#define N_RUN 1
#include "../hopcim_net/main.c"
