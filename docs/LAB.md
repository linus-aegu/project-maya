# Tuning Maya with `tools/lab.py`: the method

A setting or a patch is kept only when it is measured better on what users wait for, does not make anything else
worse, and leaves the answers' quality where it was - alone **and** next to every other setting that is kept. This page
says how `tools/lab.py` measures that, so that every decision follows the same rules.

## 1. What counts

| Objective | Metric | Better |
| --- | --- | --- |
| an agent's turn starts | **TTFT per turn**: each read size's time, weighted by the share of real turns of that size (`--weights`, from the server's own log: `lab.py weights LOG`) | lower |
| answers stream | **decode** tokens/s (three answers of 128 tokens, greedy, thinking off) | higher |
| several conversations | **mixed**: one conversation's decode while another reads 4K new tokens, and that read's time | higher / lower |
| a new conversation | **read_prefix**: a 5.3K-token start read cold | lower |

Guardrails, each a hard stop:

- **quality**: teacher-forced logits on a fixed text, through the prompt path and through windows of 16 (the decode's
  path): KL over the whole vocabulary, the top token's agreement and the loss on the text's next token, against the
  reference's - judged against the **noise floor** (the reference against itself). Greedy answers are kept and compared
  too, but they diverge for reasons that are not quality (a near-tie, a path picked by timing), so they inform and do
  not gate;
- **errors**: an engine that stops, refuses a request or sends nothing within its deadline ends the run;
- **memory**: the VRAM left free after load (the log's line) may not fall below the reference's by more than 256 MB.

## 2. Controlled conditions

- One frozen expert profile for every compared run (`--expert-profile`; the server rewrites its own at each stop).
- One fixed text for the reads (`--text`), the same requests every run (the workload's signature is in each result;
  runs of other signatures are never compared).
- The same warm-up first; then the steps always in the same order.
- Each run records the GPU's clock and temperature, the CPU's clock and the load around it, so drift can be seen.
- Configurations **interleave** (A B C A B C), never A A B B C C, so that slow drift (heat, the page cache, other
  programs) spreads over all of them instead of looking like an effect.

## 3. When a difference counts

Every configuration runs at least twice. A change counts when it is larger than twice the combined standard error of
the two sides (a side of one value takes NOISE of its mean). The reference is measured at least twice per session,
which also gives the noise floor of every metric; a result smaller than the noise floor is "the same", whatever its
sign. A run stops at its first clear regression (fail fast) unless `--keep-going`.

## 4. The design: screen, interact, confirm

Knobs are not tuned one at a time against an old reference - each is judged next to all the others that are kept.

1. **Screen** around the incumbent (today's production config): each candidate change toggled alone. Knobs without an
   effect larger than the noise floor drop out.
2. **Interact**: the knobs that showed an effect, crossed with each other (a full factorial of 2-3 knobs: 4-8
   configurations). The report gives each knob's main effect and each pair's interaction (does A's gain depend on B?).
3. **Confirm** the chosen configuration by **leave-one-out**: the final configuration, and the final configuration
   without each of its knobs in turn. Every knob must still earn its place next to all the others.
4. **Canary**: after a change goes into production, the server's own log (its per-request prompt reads and decode
   speeds) is compared before and after over real traffic.

`lab.py matrix plan SPEC.json` writes the runs of a round (interleaved); `lab.py matrix report SPEC.json` reads their
results back: per metric, each configuration's mean and spread, each knob's effect and each pair's interaction.

## 5. Records

Every run writes its JSON (settings, metrics, verdicts, conditions) and the engine's log. Proven gains go into
`EUREKAS.md` with their numbers, their proof and what is still open.

## Warm-up and calm start (2026-11-08)

- Every `lab_matrix` plan starts with one unmeasured run (`m-<name>-warmup`; `"warmup": false` in the spec drops it): the first run after the server served a while decoded its first answers ~35% faster (31 vs 20-22 tok/s, base-w and base-c), which would pull whichever configuration ran first up. Hand-written queues do the same.
- `lab.py run` waits until the 1-minute load is below `--calm-load` (1.0) before it starts, up to `--calm-wait` (300 s), and stops with the reason if the box stays busy. `calm_wait_s` in the result says how long it waited.

## The lane file, the CPU trace and the longer steps (2026-11-09)

- `--lane-cal FILE`: each run starts from its own copy of one frozen PCIe-lane times file (`DS4_LANE_CAL`). The engine
  keeps its lane verdict ("it no longer streams them, kept for the next start") in the pack's `lane-times.txt`, which
  the server and every run share, keyed by the build: without a frozen copy, a run inherits what the run before decided.
- The CPU trace (`--cpu-every`, 1 s; 0: none): the CPU's mean clock, package temperature and - where intel-rapl's
  `energy_uj` is readable (root only by default) - its power, for the whole run (`res["cpu"]`), and beside each
  answer's tokens/s in the log. On synthax the CPU may draw 250 W for ~55 s, then 125 W (intel-rapl's two limits).
- `--long-answer N`: one answer of N tokens, its end tokens banned, last in the run: `long_decode` (whole),
  `long_decode_start` / `long_decode_end` (its first and last 20 s) and its 10 s slices with the CPU beside them
  (`res["long_answer"]["slices"]`) - an answer that outlasts a power limit's burst.
- `--long-read N`: a fresh prompt of N tokens after the mixed steps (`read_long`): where the chunk size and the borrowed
  room show.
- `--mixed2`: a second mixed step, two conversations answering while a third reads 4K new tokens (`mixed2_decode`,
  `mixed2_read`, `mixed2_gap`; `--parallel 3`).
- The mixed reads ask for their one token's top 5 log-probabilities (`res["mixed_read_lp"]`): the prompt pass of a read
  that paused between its layers (DS4_LAYER_YIELD) against one that did not, to the last digit.
- The signature covers these steps only when they are asked for: runs without them compare with older runs as before.
