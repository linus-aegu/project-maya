# synthax production

What the office box `synthax` (one RX 7900 XTX, i7-11700KF, 128 GB DDR4-3200, Ubuntu 24.04, ROCm 7.1) runs: Maya v1.1.56
(`22fc101`) and the commits on this branch, DeepSeek-V4.1-Flash Maya-XS.

| Commit | What | Upstream |
| --- | --- | --- |
| #718 mask | `MAYA_DS4_PROMPT_SKIP=2` without stale window rows | issue #718 |
| AVX-512 rows | `DS4_IQ1S512=1`: decode +9% | PR #911 (on hold) |
| Layer yield | `DS4_LAYER_YIELD=1` + `_MS=100`: a running answer's freeze 4.3 s -> 0.4 s (needs `DS4_BORROW=0`) | issue #930 |
| thinking_budget | a request's own budget, clamped / left out, never a 400 | PR #891 |

`maya-ds4-hip.json` is the server's config (its `exe` is the build of this branch in `~/maya-r4/build-hip`). Copy-DMA
(`DS4_FLOW_COPY_DMA=1`, PR #910) is a setting there, not code.

Build: `cmake -S . -B build-hip` with the options of `~/maya-lab-data/cmake-opts.txt` on synthax (HIP, gfx1100,
`STRATA_NATIVE_EXPERTS=ON`, `MAYA_CPU_VNNI_ASM=1`), then `cmake --build build-hip --target maya-ds4`.
