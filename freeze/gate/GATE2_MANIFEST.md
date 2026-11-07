# Gate 2 manifest — unchanged MLPerf Tiny DS-CNN with CIM-computed MFCC (frozen before evaluation)

Frozen: 2026-10-26, before any test-set result was computed.

## Question
Does computing the standard MLPerf Tiny KWS MFCC on a digital bit-serial CIM datapath (integer model below)
change the accuracy of the **unchanged** pretrained DS-CNN by no more than 0.3 percentage points?

## Fixed inputs
| Item | Identity |
|---|---|
| Int8 model | `mlperf_tiny_kws/trained_models/kws_ref_model.tflite`, md5 `618aeb155673dae92fa6a7f26608add5` (matches upstream `kws_ref_model.md5`) |
| Float model | `mlperf_tiny_kws/trained_models/kws_ref_model_float32.tflite`, md5 `d7c06dc53dce690c427f593e480cf8fc` |
| Upstream | mlcommons/tiny, last commit touching `benchmark/training/keyword_spotting`: `0c6a99052cf4bee500cfa47424b8bebde1f69965` |
| Test set | `speech_commands_test_set_v0.02.tar.gz`, sha256 `cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df`, 4,890 clips, 12 classes |
| Labels | `Down, Go, Left, No, Off, On, Right, Stop, Up, Yes, Silence, Unknown` (reference `word_labels` order; folders `_silence_`, `_unknown_` map to the last two) |

## Reference features (float, = MLPerf Tiny `get_dataset.py`, test path)
int16 → float32 → divide by `max(x)` → zero-pad to 16000 → (time-shift pad 2 / slice 2 = identity) →
`tf.signal.stft(480, 320, fft_length=None→512, hann_window periodic)` → `|·|` →
`linear_to_mel_weight_matrix(40, 257, 16000, 20, 4000)` → `log(mel + 1e-6)` →
`mfccs_from_log_mel_spectrograms` → first 10 → shape (49, 10, 1).
Int8 input quantisation exactly as `eval_quantized_model.py`: `np.array(x/scale + zp, dtype=np.int8)`.

## CIM-MFCC integer datapath (frozen design; CIM = exact integer MACs, int8 weights, any-width bit-serial inputs)
1. Window: Hann (periodic, 480) in Q15; product kept wide; **per-frame block exponent** normalises the frame peak to [2^14, 2^15) (int16).
2. FFT 512 four-step (16 × 32): step-1 DFT matrix **int8** (scale 127), output `>> 11` to int16; twiddles **int16 Q14** (complex multiplier); step-3 DFT matrix **int8**, output `>> 9`; bins 0..256.
3. Magnitude: **exact integer square root** of re² + im² (bit-serial sqrt unit).
4. Mel: TF mel matrix quantised to **int8 (0..127)**, applied by the CIM.
5. Log: leading-one detector + **64-entry log2(1+f) LUT** (6 mantissa bits), Q8 output; per-frame exponent, mel scale and the per-clip `max(x)` normalisation are added **exactly** in the log domain; floor = `max(ln, ln 1e-6)`, exact zero → `ln 1e-6`.
6. DCT: TF's `mfccs_from_log_mel_spectrograms` matrix (DCT-II · rsqrt(2N)), **int16** coefficients (two 8-bit CIM passes), first 10 outputs.

No parameter of this datapath may be changed after seeing test results. Any later change is reported as post hoc.

## Metrics and pass rule
- Primary: Δ = acc(int8 model, CIM-MFCC) − acc(int8 model, reference float MFCC) on all 4,890 clips. **Pass iff Δ ≥ −0.30 pp.**
- Secondary (reported, not part of the rule): the same for the float32 model; exact McNemar test on discordant clips; feature error statistics in model-input quantisation steps.
- Sanity check of the reference replication: reference accuracy should be ≈ 92 % (upstream README); a large deviation invalidates the run.
