#!/usr/bin/env python3
"""Broken variants of cim_gseq (docs/NET2_GATES.md N7) and sanity variants on image 0 of the bench: each is caught if the bench does not
print PASS within 60 s (a wrong output, a lost result or a hang).   python3 sim/run_net2_mutants.py  (needs model/gen_net2_vectors.py 20)"""
import os, subprocess, shutil, sys
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S=os.path.join(R,'sim','net2_mut')
MUT=[
 ("N7a residual operands' multipliers swapped", "ma[k] <= 41'(da[k]) * 41'(signed'({1'b0, o_qa}));", "ma[k] <= 41'(da[k]) * 41'(signed'({1'b0, o_qb}));"),
 ("N7b bank rewritten one cycle early (with REL = 0)", "assign g_adv  = g_on && (g_vec || !occ[g_bank]);", "assign g_adv  = g_on && (g_vec || !occ[g_bank] || (pass_go && (i_bank == g_bank) && !g_vec));"),
 ("N7b stronger: bank released when its vector is complete, before its pass starts", "        if (pass_go && (i_bank == 1'(b))) occ[b] <= fr && (g_bank == 1'(b));", "        if (xw_we_o && xw_last && (xw_bank_o == 1'(b))) occ[b] <= fr && (g_bank == 1'(b));"),
 ("N7c stride ignored in the gather", "signed'({2'd0, g_oy}) * (o_s2 ? 7'sd2 : 7'sd1)", "signed'({2'd0, g_oy}) * 7'sd1"),
 ("N7d accumulate flag dropped", "assign accum_o = (i_r != 4'd0);", "assign accum_o = 1'b0;"),
 ("sanity: zero point not subtracted", "XB'(9'(v) - 9'(signed'(o_za)))", "XB'(9'(v))"),
 ("sanity: relu clamp ignored", "assign o_lo = o_relu ? signed'(o_zo) : -8'sd128;", "assign o_lo = -8'sd128;"),
]
src=open(R+'/rtl/cim_gseq.sv').read()
shutil.rmtree(S,ignore_errors=True); os.makedirs(S)
for i,(name,a,b) in enumerate(MUT):
    assert src.count(a)>=1,(name,src.count(a))
    d=f"{S}/m{i}"; shutil.copytree(R+'/rtl',d+'/rtl'); shutil.copytree(R+'/tb',d+'/tb',ignore=shutil.ignore_patterns('*vec*'))
    os.symlink(R+'/tb/net2_vec',d+'/tb/net2_vec')
    open(d+'/rtl/cim_gseq.sv','w').write(src.replace(a,b))
    cmd=f"""cd {d}; source $HOME/APlus_Project/planenest/hw/xheep_env.sh >/dev/null 2>&1
PKG=$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv
R={d}/rtl; verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_net2 -GDWU=0 -Mdir {d}/obj -MAKEFLAGS OBJCACHE= -I$R $PKG $R/cim_engine.sv $R/cim_fe.sv $R/mfcc_ded.sv $R/cim_log2fix16.sv $R/cim_seq.sv $R/pdm_dec.sv $R/cim_audio.sv $R/cim_gseq.sv $R/cim_endpoint.sv tb/tb_cim_net2.sv > build.log 2>&1
timeout 60 ./obj/Vtb_cim_net2 +NIMG=1 +IMG0=0 > run.log 2>&1 || echo 'TIMEOUT_OR_ERROR' >> run.log; tail -3 run.log | head -2"""
    p=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,timeout=1800)
    log=open(d+'/run.log').read() if os.path.exists(d+'/run.log') else ''
    caught=('PASS' not in log.split('\n'))
    print(f"{name}: {'CAUGHT' if caught else 'NOT CAUGHT'}", (p.stdout.strip()[-140:] if not caught else [l for l in log.split(chr(10)) if 'RESULT' in l]) , flush=True)
