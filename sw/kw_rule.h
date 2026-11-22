// HopCIM: the summed-logit keyword rule of hopcim_live built with KW_SUM > 0 (design U of docs/WHOLEWORD_STUDY.md:
// NET_K = 16, KW_SUM = 2, hold 1). Pure C, no memory-mapped I/O, so model/test_kw_rule.py compiles this very file on
// the host and checks it against model/kwlog_analyse.apply_rule(("smooth", KW_SUM, hold), ...), the rule the study scored.
//
// Per decision: the 12 logits of the last KW_SUM decisions are summed (fewer at the start of the stream) and the top
// class is taken, ties going to the lower class (numpy's argmax). On that class runs hopcim_live's confirmation: a
// keyword is reported when it has been the top class for `hold` decisions in a row and is not the keyword reported last;
// a decision that is not a keyword (silence, unknown) allows the same keyword again.
#ifndef HOPCIM_KW_RULE_H_
#define HOPCIM_KW_RULE_H_

#include <stdint.h>

#ifndef KW_SUM
#error "kw_rule.h needs KW_SUM, the number of decisions summed"
#endif
#if (KW_SUM) < 1 || (KW_SUM) > 64
#error "KW_SUM must be 1..64 (the sums are int16)"
#endif

typedef struct {
    int16_t sum[12];                                 // the logits of the last `have` decisions, summed
    int8_t ring[KW_SUM][12];                         // those decisions' logits
    uint8_t slot, have;                              // the next slot to write; decisions held (<= KW_SUM)
    int8_t last, said;                               // last top class; keyword reported last (-1: none)
    uint32_t run;                                    // decisions in a row with that top class
} kw_rule_t;

static inline void kw_rule_init(kw_rule_t *r) {
    for (int c = 0; c < 12; c++) {
        r->sum[c] = 0;
        for (int i = 0; i < (KW_SUM); i++) r->ring[i][c] = 0;
    }
    r->slot = r->have = 0;
    r->last = r->said = -1;
    r->run = 0;
}

// One decision's 12 logits; returns the keyword to report (0..9) or -1.
static inline int kw_rule_step(kw_rule_t *r, const int8_t *lg, uint32_t hold) {
    for (int c = 0; c < 12; c++) {
        if (r->have == (KW_SUM)) r->sum[c] -= r->ring[r->slot][c];
        r->ring[r->slot][c] = lg[c];
        r->sum[c] += lg[c];
    }
    if (r->have < (KW_SUM)) r->have++;
    r->slot = (uint8_t)((r->slot + 1u) % (KW_SUM));
    int top = 0;
    for (int c = 1; c < 12; c++)
        if (r->sum[c] > r->sum[top]) top = c;
    r->run = (top == r->last) ? r->run + 1u : 1u;
    r->last = (int8_t)top;
    int say = (top < 10) && (r->run == hold) && (top != r->said);
    if (top >= 10) r->said = -1;
    if (say) r->said = (int8_t)top;
    return say ? top : -1;
}

#endif
