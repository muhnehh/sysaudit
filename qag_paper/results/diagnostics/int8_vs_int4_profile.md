# INT8 vs INT4 Profiling Summary

- GPU: NVIDIA GeForce RTX 2060 with Max-Q Design (cc 7.5)
- Model: Qwen/Qwen2.5-1.5B-Instruct
- torch: 2.3.0+cu121, bitsandbytes: 0.43.1

| Mode | Forward ms | Generate 64 tok ms |
|---|---:|---:|
| fp16 | 69.94 | 3457.87 |
| int8 | 389.63 | 24124.39 |
| int4 | 91.97 | 5211.29 |

## Ratios

- int8_vs_fp16_generate: 6.977
- int8_vs_int4_generate: 4.629
- int4_vs_fp16_generate: 1.507
- int8_vs_fp16_forward: 5.571
- int8_vs_int4_forward: 4.236
