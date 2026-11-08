// HopCIM R7: the scripted keyword test. This is hopcim_live built with KW_TEST, the number of rounds
// (13 prompts each, 4 s per prompt; see there). hopcim/model/score_kwtest.py scores the log.
// Build: make app PROJECT=hopcim_kwtest TARGET=nexys-a7-100t COMPILER_FLAGS=-DREFERENCE_CLOCK_Hz=15360000
// and run it on a bitstream built with --flag use_cim_mic (15.36 MHz).

#ifndef KW_TEST
#define KW_TEST 5
#endif
#include "../hopcim_live/main.c"
