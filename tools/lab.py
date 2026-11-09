#!/usr/bin/env python3
"""tools/lab.py - one engine setting or build against another in a few minutes: the same answers or not, faster or
slower, and an early stop as soon as the answer is clear.

`./maya.sh --bench` gives a PC's speed and tools/agent_bench.py an agent's load over 15-30 minutes. Trying a change - an
environment setting, a patched build - wants a shorter loop: start the engine as the config does (with the change),
send the same requests every time, and say at once what moved. A run:

- loads the engine the way the server starts it (the config's command and environment, plus --exe / --env / --arg),
  with the expert profile read-only (`--profile-readonly`: a run does not move the experts a later start places) and,
  for DeepSeek, no conversations kept or taken back (`MAYA_DS4_SLOT_KEEP=0`: nothing a run reads is reused from
  earlier runs or left for the server);
- answers a warm-up and three questions, greedy, thinking off, each token with its log-probability (`logprobs=2`):
  the answers, and the decode speed;
- reads a shared start (--prefix tokens, ~an agent's system prompt) once, then new text of each --reads size on top of
  it, --rounds times (other text each round): what an agent's turn waits for before its first token.

With --against LABEL every step is compared with that label's runs as soon as it is measured (the same requests: runs
of another workload are not compared). The answers: `identical` (every token and log-probability), `last bits`
(where they part, the reference's two likeliest tokens were within NEAR_TIE of each other, and the common tokens'
log-probabilities moved by LAST_BITS on average at most) or `changed`. A speed counts as better or worse when it moved
by more than twice the combined spread of the two sides (their standard deviations, or NOISE of the mean for a side of
one value). The run stops at the first step that fails: answers below --gate, or a speed that got worse - unless
--keep-going. Every line it prints starts with `lab:`; the last one with `lab: end`.

    # the reference: the config as it is, twice (the spread between them is the noise the verdicts use)
    python tools/lab.py run --config maya-ds4-hip.json --label base
    python tools/lab.py run --config maya-ds4-hip.json --label base
    # a change: a setting, or another build
    python tools/lab.py run --config maya-ds4-hip.json --label skip1 --env DS4_SKIP_UNPICKED=1 --against base
    python tools/lab.py run --config maya-ds4-hip.json --label patch --exe ../maya-patch/build-hip/maya-ds4 --against base
    # every label's runs side by side, the first label the reference
    python tools/lab.py show base skip1 patch

The engine holds the GPU memory a server needs: stop Maya first (tools/lab_unit.sh does that for a server that runs as
a systemd user unit, and starts it again after the run whatever happens). Results go to --runs (lab-runs/), a JSON file
and the engine's log a run. Standard library only.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import queue
import re
import shutil
import statistics as st
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WARMUP = "Say hello in five languages."
QUESTIONS = [
    "Explain how a hash map handles collisions, with a short example in Python.",
    "Write the opening of a short story about a lighthouse keeper who finds a message in a bottle.",
    "Was ist der Unterschied zwischen TCP und UDP? Antworte auf Deutsch, in ein paar Sätzen.",
]
ASK = "Summarise the text above in one sentence."
MIXED_Q = "Explain step by step how TCP's congestion control reacts to packet loss."
READS = (16, 64, 256, 1024, 4096)   # new tokens on top of the shared start
PREFIX = 5300                       # the shared start (an agent's system prompt and tools)
ANSWER = 128                        # tokens an answer writes
LONG_ANSWER = 64                    # tokens the answer to the shared start writes (a long prompt's answer, gated too)
MIXED_READ = 4096                   # the mixed step's read, while another conversation decodes
MIXED_ANSWER = 256                  # ... that conversation's answer
LONG_Q = ("Write a long, detailed handbook for someone opening a small bakery: the equipment, the suppliers, the "
          "daily schedule, the recipes, the costs, the staff, the mistakes to avoid. Cover every part in depth.")
END_TOKENS = ("<｜end▁of▁sentence｜>", "<｜User｜>", "<|im_end|>", "<|endoftext|>", "<|user|>", "<|observation|>",
              "[e~[", "]~b]")   # (the server's stop tokens, serve/server.py)   # (banned: --long-answer runs to length)
SLICE_S = 10.0                      # the long answer's speed, slice by slice
EDGE_S = 20.0                       # ... and over its first and last seconds
QUALITY = 1024                      # the quality text's tokens (lab_quality.py)
NEAR_TIE = 0.5                      # nats between the reference's two likeliest tokens: a divergence there is a near-tie
LAST_BITS = 0.05                    # mean |change| of the common tokens' log-probabilities for `last bits`
NOISE = {"read": 0.10, "decode": 0.08}   # the spread assumed for a side of one value, of its mean
ABS_FLOOR = {"mixed_decode": 1.0, "mixed2_decode": 1.0}   # a difference smaller than this never counts (near zero)
LEVELS = ("changed", "last bits", "identical")


class LabError(Exception):
    """A run that cannot go on: the engine stopped, sent nothing in time, or refused a request."""


# ---------------------------------------------------------------- the engine's lines (pure)
@dataclass
class Done:
    produced: int
    n_ids: int
    prompt_ms: float
    decode_ms: float
    finish: str
    reused: int
    hits: int        # the decode's routed experts VRAM held
    experts: int     # ... of all of them


def parse_done(f: list[str], multi: bool) -> Done:
    """A DONE line's fields (split): `DONE [rid] produced n_ids prompt_ms decode_ms finish drafts_kept drafts reused
    hits experts` - the rid only from an engine that runs several conversations."""
    if multi:
        f = f[:1] + f[2:]
    return Done(int(f[1]), int(f[2]), float(f[3]), float(f[4]), f[5], int(f[8]),
                int(f[9]) if len(f) > 9 else 0, int(f[10]) if len(f) > 10 else 0)


def parse_lp(f: list[str], multi: bool) -> tuple[int, float, list[float]]:
    """An LP line's fields (split): `LP [rid] id lp top:lp ...` -> the token, its log-probability and the top ones'."""
    if multi:
        f = f[:1] + f[2:]
    return int(f[1]), float(f[2]), [float(x.split(":")[1]) for x in f[3:]]


def ours(f: list[str], multi: bool) -> bool:
    """Whether a T / LP / DONE / ERR line is this tool's request (rid 0) - an engine of several conversations tags
    each line with its request's id."""
    return not multi or (len(f) > 1 and f[1] == "0")


# ---------------------------------------------------------------- the workload (pure, given the token ids)
@dataclass
class Workload:
    warm: list[int]
    questions: list[list[int]]
    prefix: list[int]                          # the shared start, with the question that ends its first read
    reads: list[tuple[int, list[int]]]         # (new tokens, the whole prompt), round by round
    shared: int                                # the shared start's length (its restore point)
    mixed_q: list[int] = field(default_factory=list)      # the mixed step: one conversation's question ...
    mixed_read: list[int] = field(default_factory=list)   # ... and the other's fresh prompt, read meanwhile
    quality: list[int] = field(default_factory=list)      # BOS + the quality text (lab_quality.py)
    warm_read: list[int] = field(default_factory=list)    # an unmeasured read before the measured ones (the first read
                                                          # after the shared start ran ~0.4-0.9 s slow in half the runs)
    # the steps a run asks for (--mixed2, --long-read, --long-answer); none: the signature as before they existed
    mixed_read2: list[int] = field(default_factory=list)  # the mixed step's read with two conversations answering
    long_read: list[int] = field(default_factory=list)    # a long fresh prompt
    long_q: list[int] = field(default_factory=list)       # a question whose answer runs to long_n tokens
    long_n: int = 0
    check_read: list[int] = field(default_factory=list)   # --pause-check: a fresh prompt read alone, then beside an answer

    def signature(self) -> str:
        h = hashlib.sha256()
        for ids in [self.warm, *self.questions, self.prefix, *(ids for _, ids in self.reads), self.mixed_q,
                    self.mixed_read, self.quality, self.warm_read]:
            h.update(",".join(map(str, ids)).encode())
            h.update(b";")
        for name, ids in (("mixed2", self.mixed_read2), ("long_read", self.long_read),
                          (f"long_answer {self.long_n}", self.long_q), ("pause_check", self.check_read)):
            if ids:
                h.update(f"{name}:".encode() + ",".join(map(str, ids)).encode() + b";")
        return h.hexdigest()[:16]


def build_workload(chat, doc: list[int], prefix: int = PREFIX, sizes=READS, rounds: int = 2, mixed: int = MIXED_READ,
                   quality: int = QUALITY, mixed2: bool = False, long_read: int = 0, long_answer: int = 0,
                   pause_check: bool = False) -> Workload:
    """The requests of a run from `chat` (a question -> its prompt's ids, in the model's chat format, thinking off) and
    `doc` (the text the reads take their tokens from): the same every run of the same arguments. The reads' new text,
    the mixed steps' reads, the long read and the quality text come from slices of `doc` nothing else uses, so nothing
    but the shared start can be reused."""
    before, after = chat.split_ids()
    need = prefix + rounds * sum(sizes) + mixed * (1 + mixed2 + pause_check) + long_read + quality + 32
    if len(doc) < need:
        raise LabError(f"the text for the reads has {len(doc)} tokens, they need {need}: fewer --reads or --rounds")
    start = before + doc[:prefix]
    ask = chat.text_ids(ASK) + after
    reads, at = [], prefix
    for _ in range(rounds):
        for n in sizes:
            reads.append((n, start + doc[at:at + n] + ask))
            at += n
    def fresh(n: int) -> list[int]:   # (a prompt of n new tokens nothing else reads)
        nonlocal at
        at += n
        return before + doc[at - n:at] + ask

    mixed_read = fresh(mixed)
    qtext = before[:1] + doc[at:at + quality]
    at += quality
    w = Workload(chat(WARMUP), [chat(q) for q in QUESTIONS], start + ask, reads, len(start), chat(MIXED_Q),
                 mixed_read, qtext, start + doc[at:at + 16] + ask)
    at += 16
    if mixed2:
        w.mixed_read2 = fresh(mixed)
    if long_read:
        w.long_read = fresh(long_read)
    if long_answer:
        w.long_q, w.long_n = chat(LONG_Q), long_answer
    if pause_check:
        w.check_read = fresh(mixed)
    return w


class Chat:
    """A family's chat format (maya.bench_chat: thinking off) through a tokenizer: chat(q) the prompt of question q;
    split_ids() the format's ids before and after the question, text_ids(s) plain text."""

    def __init__(self, fmt, tok):
        self.fmt, self.tok = fmt, tok

    def __call__(self, q: str) -> list[int]:
        return self.tok.encode(self.fmt(q), parse_special=True)

    def split_ids(self) -> tuple[list[int], list[int]]:
        a, b = self.fmt("\x00").split("\x00")
        return self.tok.encode(a, parse_special=True), self.tok.encode(b, parse_special=True)

    def text_ids(self, s: str) -> list[int]:
        return self.tok.encode(s)


# ---------------------------------------------------------------- the verdicts (pure)
@dataclass
class Answer:
    tokens: list[int] = field(default_factory=list)
    lp: list[float] = field(default_factory=list)      # each token's log-probability
    gap: list[float] = field(default_factory=list)     # the two likeliest tokens' difference at each position

    def to_json(self) -> dict:
        return {"tokens": self.tokens, "lp": self.lp, "gap": self.gap}

    @staticmethod
    def from_json(d: dict) -> "Answer":
        return Answer(list(d["tokens"]), list(d["lp"]), list(d["gap"]))


def compare_answers(ref: list[Answer], new: list[Answer]) -> tuple[str, str]:
    """The answers' level against the reference's (LEVELS) and a line saying why: where each one parts from the
    reference (and whether the reference was near a tie there), and how far the common tokens' log-probabilities moved."""
    level, notes, moved = "identical", [], []
    for i, (r, n) in enumerate(zip(ref, new)):
        k = 0
        while k < min(len(r.tokens), len(n.tokens)) and r.tokens[k] == n.tokens[k]:
            k += 1
        moved += [abs(a - b) for a, b in zip(r.lp[:k], n.lp[:k])]
        if k == len(r.tokens) == len(n.tokens):
            continue
        tie = k < len(r.gap) and r.gap[k] < NEAR_TIE
        notes.append(f"#{i} parts at token {k}" + (f" (a near-tie: {r.gap[k]:.2f})" if tie else
                                                   f" (not a tie: {r.gap[k]:.2f})" if k < len(r.gap) else ""))
        level = min(level, "last bits" if tie else "changed", key=LEVELS.index)
    mean = st.fmean(moved) if moved else 0.0
    if level == "identical" and any(m > 0 for m in moved):
        level = "last bits"
    if level == "last bits" and mean > LAST_BITS:
        level = "changed"
    return level, "; ".join(notes + [f"log-probabilities moved {mean:.4f} on average over {len(moved)} tokens"])


def judge(new: list[float], ref: list[float], higher_better: bool, rel: float, floor: float = 0.0) -> tuple[str, float]:
    """'better', 'worse' or 'same' for `new` against `ref`, and the change in % of the reference's mean: a change counts
    when it is larger than twice the combined standard error (a side of one value: `rel` of its mean as its spread)."""
    if not new or not ref:
        return "same", 0.0
    mn, mr = st.fmean(new), st.fmean(ref)
    sd = lambda v, m: st.stdev(v) if len(v) > 1 else rel * m   # noqa: E731
    se = math.sqrt(sd(new, mn) ** 2 / len(new) + sd(ref, mr) ** 2 / len(ref))
    delta = (mn - mr) / mr * 100.0 if mr else 0.0
    if abs(mn - mr) <= max(2.0 * se, floor):
        return "same", delta
    return ("better" if (mn > mr) == higher_better else "worse"), delta


def metric_kind(name: str) -> tuple[bool, float]:
    """(higher is better, the assumed spread) of a metric by its name: the decodes (decode, mixed_decode, mixed2_decode,
    long_decode*) in tokens/s, the rest (read*, turn, mixed_read, mixed_gap) in ms."""
    return (True, NOISE["decode"]) if "decode" in name else (False, NOISE["read"])


# ---------------------------------------------------------------- real traffic's weights (pure)
PROMPT_LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")


def bucket(n: int, sizes) -> int:
    """The read size of `sizes` nearest to n new tokens on a log scale."""
    return min(sizes, key=lambda s: abs(math.log(max(n, 1)) - math.log(s)))


def weights_from_log(lines, sizes=READS) -> dict[int, float]:
    """The share of real turns of each read size: the server log's prompt lines that continue a conversation (some
    tokens reused, 2 or more read) - an agent's turns - each counted at its nearest size."""
    counts = {s: 0 for s in sizes}
    for line in lines:
        m = PROMPT_LINE.search(line)
        if m and int(m.group(2)) > 0 and int(m.group(3)) >= 2:
            counts[bucket(int(m.group(3)), sizes)] += 1
    total = sum(counts.values())
    return {s: round(c / total, 4) for s, c in counts.items()} if total else {}


def turn_ms(by_n: dict[int, list[float]], w: dict[int, float]) -> float | None:
    """TTFT per turn: each read size's mean time weighted by its share of real turns (over the sizes measured)."""
    have = {n: st.fmean(v) for n, v in by_n.items() if v and w.get(n)}
    tot = sum(w[n] for n in have)
    return sum(w[n] * have[n] for n in have) / tot if tot else None


# ---------------------------------------------------------------- the run's conditions
def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def wait_calm(limit: float, most: float, every: float = 5.0) -> float:
    """Waits until the 1-minute load is below `limit` (the box idle, not still busy from the server's restart or the
    run before - a run started busy decoded ~30% slower than one started idle) and returns the seconds waited; past
    `most` seconds, the run stops: its numbers would not be comparable."""
    t = time.time()
    while (now := load1()) >= limit:
        if time.time() - t > most:
            raise LabError(f"the box stayed busy: load {now:.2f}, not below {limit} after {most:.0f} s. Find what runs "
                           f"(top), or raise --calm-wait; --calm-load 0 skips the wait (runs then not comparable)")
        time.sleep(every)
    return time.time() - t


def cpu_mhz() -> int | None:
    """The CPU's mean clock over its threads now (Linux /proc/cpuinfo)."""
    mhz = [float(x.split(":")[1]) for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("cpu MHz")]
    return round(st.fmean(mhz)) if mhz else None


@functools.cache
def cpu_temp_file() -> Path | None:
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hw / "name").exists() and (hw / "name").read_text().strip() in ("coretemp", "k10temp"):
            return hw / "temp1_input"
    return None


def cpu_temp() -> int | None:
    """The CPU package's temperature (hwmon coretemp / k10temp temp1: Intel's package, AMD's Tctl), None without one."""
    f = cpu_temp_file()
    return int(f.read_text()) // 1000 if f else None


RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")   # (the package's energy - root only by default since 2020)


def cpu_energy_uj() -> int | None:
    try:
        return int(RAPL.read_text())
    except (OSError, ValueError):
        return None


class CpuTrace:
    """The CPU's mean clock, package temperature and - where intel-rapl's energy is readable - its power, every `every`
    seconds on a thread for the whole run: what a decode speed that drifts within a run moves with (a power limit
    ending its burst, heat). rows: [seconds since the start, MHz, °C, W] (None: not readable)."""

    def __init__(self, every: float = 1.0):
        self.every, self.rows, self.t0 = every, [], time.time()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "CpuTrace":
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._th.join(timeout=5.0)

    def _loop(self) -> None:
        last = None
        while not self._stop.is_set():
            t, row = time.time(), [None, None, None]
            for i, part in enumerate((cpu_mhz, cpu_temp)):   # (each on its own, as conditions())
                try:
                    row[i] = part()
                except (OSError, ValueError, IndexError):
                    pass
            uj = cpu_energy_uj()
            if uj is not None and last is not None and uj >= last[1] and t > last[0]:
                row[2] = round((uj - last[1]) / 1e6 / (t - last[0]), 1)
            last = (t, uj) if uj is not None else None
            self.rows.append([round(t - self.t0, 2)] + row)
            self._stop.wait(self.every)

    def between(self, a: float, b: float) -> dict:
        """The mean clock, the highest temperature and the mean power of the samples from time a to b (time.time())."""
        rows = [r for r in self.rows if a - self.t0 <= r[0] <= b - self.t0]
        col = lambda i: [r[i] for r in rows if r[i] is not None]   # noqa: E731
        out = {}
        if col(1):
            out["mhz"] = round(st.fmean(col(1)))
        if col(2):
            out["temp_c"] = max(col(2))
        if col(3):
            out["w"] = round(st.fmean(col(3)), 1)
        return out


def cpu_text(s: dict) -> str:
    return " ".join(x for x in (f"{s['mhz']} MHz" if "mhz" in s else "", f"{s['temp_c']} °C" if "temp_c" in s else "",
                                f"{s['w']} W" if "w" in s else "") if x) or "-"


def slices(times: list[float], width: float = SLICE_S) -> list[tuple[float, float, float]]:
    """A long answer's token times (time.time()) -> (start, end, tokens/s) of each `width`-second slice from its first
    token on (the last one only when it is at least half as wide)."""
    if len(times) < 2:
        return []
    out, a = [], times[0]
    while a < times[-1]:
        b = min(a + width, times[-1])
        n = sum(1 for t in times if a < t <= b)
        if b - a >= width / 2:
            out.append((a, b, n / (b - a)))
        a = b
    return out


def edge_rates(times: list[float], span: float = EDGE_S) -> tuple[float | None, float | None]:
    """A long answer's tokens/s over its first and its last `span` seconds (None: the answer was shorter than 2 spans)."""
    if len(times) < 2 or times[-1] - times[0] < 2 * span:
        return None, None
    first = sum(1 for t in times if times[0] < t <= times[0] + span) / span
    last = sum(1 for t in times if times[-1] - span < t <= times[-1]) / span
    return first, last


def cpu_summary(res: dict, trace: CpuTrace) -> None:
    """The CPU trace into the run's results, and beside the speeds it may explain: each answer's tokens/s with the
    CPU's clock / temperature / power meanwhile, and the long answer's slices likewise."""
    res["cpu"] = {"every": trace.every, "rows": trace.rows}
    wins = (res.get("cpu_windows") or {}).get("answers") or []
    if wins:
        res["cpu"]["answers"] = [trace.between(a, b) for a, b in wins]
        rates = (res.get("cpu_windows") or {}).get("rates") or []
        say("cpu during the answers: " + " | ".join(
            (f"{rates[i]:.1f} tok/s " if i < len(rates) and rates[i] is not None else "") + cpu_text(s)
            for i, s in enumerate(res["cpu"]["answers"])))
    la = res.get("long_answer")
    if la and la.get("t0") is not None:
        times = [la["t0"] + t for t in la["times"]]
        rows, texts = [], []
        for a, b, rate in slices(times):
            s = trace.between(a, b)
            rows.append([round(a - times[0], 1), round(b - times[0], 1), round(rate, 2), s.get("mhz"), s.get("temp_c"),
                         s.get("w")])
            texts.append(f"{a - times[0]:.0f}-{b - times[0]:.0f} s {rate:.1f} tok/s {cpu_text(s)}")
        la["slices"] = rows
        say("long answer, slice by slice: " + " | ".join(texts))


def conditions() -> dict:
    """What else could move a run's numbers, read before and after it (Linux; what is missing stays out): the GPU's
    clock, memory clock, temperature and power (amdgpu sysfs), the CPU's mean clock, the load and the free RAM."""
    def gpu(out):
        for dev in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            if not (dev / "pp_dpm_sclk").exists():
                continue
            for name, f in (("gpu_sclk", "pp_dpm_sclk"), ("gpu_mclk", "pp_dpm_mclk")):
                cur = [x for x in (dev / f).read_text().splitlines() if x.strip().endswith("*")]
                if cur:
                    out[name] = cur[0].split(":")[1].strip(" *")
            for hw in (dev / "hwmon").glob("hwmon*"):
                if (hw / "temp1_input").exists():
                    out["gpu_temp_c"] = int((hw / "temp1_input").read_text()) // 1000
                power = next((hw / f for f in ("power1_average", "power1_input") if (hw / f).exists()), None)
                if power:
                    out["gpu_w"] = round(int(power.read_text()) / 1e6)
            return

    def cpu(out):
        if (mhz := cpu_mhz()) is not None:
            out["cpu_mhz"] = mhz
        if (temp := cpu_temp()) is not None:
            out["cpu_temp_c"] = temp

    def load(out):
        out["load1"] = load1()

    def mem(out):
        for x in Path("/proc/meminfo").read_text().splitlines():
            if x.startswith("MemAvailable:"):
                out["mem_avail_gb"] = round(int(x.split()[1]) / 1048576, 1)

    out = {}
    for part in (gpu, cpu, load, mem):   # (each on its own: one unreadable file leaves only its own values out)
        try:
            part(out)
        except (OSError, ValueError, IndexError):
            pass
    return out


# ---------------------------------------------------------------- the engine
class Engine:
    """The engine as the server starts it, spoken to over its stdin / stdout lines; its stderr goes to `log`. Every
    wait has a deadline: a line that does not come in time is a LabError, not a hang."""

    def __init__(self, cmd: list[str], env: dict, cwd: str, log: Path):
        self.log_path, self.exe = log, cmd[0]
        self.log = open(log, "w", encoding="utf-8")
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                      env=env, cwd=cwd, bufsize=1)
        except OSError as e:
            self.log.close()
            raise LabError(f"the engine did not start ({cmd[0]}): {e}") from None
        self.q: queue.Queue = queue.Queue()
        self.multi = False
        self.slots = 1
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def line(self, until: float) -> str:
        t = time.time()
        try:
            line = self.q.get(timeout=max(0.0, until - t))
        except queue.Empty:
            raise LabError(f"the engine sent nothing more in {time.time() - t:.0f} s (the step's deadline); the end of "
                           f"its log ({self.log_path}):\n" + tail(self.log_path)) from None
        if line is None:
            code = self.p.wait()
            raise LabError(f"the engine stopped (exit code {code}); the end of its log ({self.log_path}):\n" +
                           tail(self.log_path))
        return line

    def ready(self, timeout: float) -> None:
        until = time.time() + timeout
        while True:
            line = self.line(until)
            if line.startswith("READY"):
                self.multi = any(w.startswith("multi=") for w in line.split())
                self.slots = next((int(w[6:]) for w in line.split() if w.startswith("multi=")), 1)
                return

    def gen(self, ids: list[int], max_new: int, timeout: float, times: list[float] | None = None,
            **keys) -> tuple[Done, Answer]:
        """One request; `times` (if given) gets each token's arrival (time.time())."""
        kv = " ".join(f"{k}={v}" for k, v in keys.items())
        rid = " rid=0" if self.multi else ""
        self.p.stdin.write(f"GEN {max_new} temperature=0{rid} {kv} " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        until, ans = time.time() + timeout, Answer()
        while True:
            f = self.line(until).split()
            if not f or f[0] not in ("T", "LP", "DONE", "ERR") or not ours(f, self.multi):
                continue
            if f[0] == "ERR":
                raise LabError("the engine refused a request: " + " ".join(f[1:]))
            if f[0] == "T":
                if times is not None:
                    times.append(time.time())
                ans.tokens.append(int(f[-1]))
            elif f[0] == "LP":
                _, lp, top = parse_lp(f, self.multi)
                ans.lp.append(lp)
                ans.gap.append(top[0] - top[1] if len(top) > 1 else 99.0)
            else:
                return parse_done(f, self.multi), ans

    def mixed(self, decodes: list[list[int]], read_ids: list[int], n_decode: int, timeout: float) -> dict:
        """Conversations answer (rids 1, 3, ...: one per prompt of `decodes`) while another (rid 2) reads a fresh
        prompt: the read starts once every answer has 16 tokens. -> the answers' mean tokens/s while the read runs and
        after it, the read's time, and the longest pause of any answer. Needs an engine of several conversations
        (--parallel 2 or more; one more than the answers)."""
        if not self.multi:
            raise LabError("the mixed step needs an engine of several conversations (--parallel 2 or more)")
        rids = [1] + list(range(3, 2 + len(decodes)))
        # every request line built before the clock starts: the read is sent mid-answer, inside the timed window
        lines = {rid: f"GEN {n_decode} temperature=0 rid={rid} " + ",".join(map(str, ids)) + "\n"
                 for rid, ids in zip(rids, decodes)}
        # (the read's one token with its top log-probabilities: a prompt read with pauses between its layers against
        #  one without - the same row, to the last digit, or the pauses moved what it computes)
        lines[2] = "GEN 1 temperature=0 rid=2 logprobs=5 " + ",".join(map(str, read_ids)) + "\n"
        send = lambda rid: (self.p.stdin.write(lines[rid]), self.p.stdin.flush())   # noqa: E731
        until = time.time() + timeout
        for rid in rids:
            send(rid)
        toks = {str(rid): [] for rid in rids}
        done: dict[str, float] = {}
        t_read, t_read_done, read_ms, read_lp = None, None, None, None

        def maybe_read():   # (once every answer has 16 tokens, or is done)
            nonlocal t_read
            if t_read is None and all(len(v) >= 16 or k in done for k, v in toks.items()):
                t_read = time.time()
                send(2)

        while len(done) < len(rids) or read_ms is None:
            f = self.line(until).split()
            if not f or f[0] not in ("T", "LP", "DONE", "ERR") or len(f) < 2:
                continue
            if f[0] == "ERR":
                raise LabError("the mixed step: the engine refused a request: " + " ".join(f[1:]))
            if f[0] == "LP" and f[1] == "2":
                read_lp = f[2:]
            elif f[0] == "T" and f[1] in toks:
                toks[f[1]].append(time.time())
                maybe_read()
            elif f[0] == "DONE" and f[1] == "2":
                t_read_done, read_ms = time.time(), parse_done(f, True).prompt_ms
            elif f[0] == "DONE" and f[1] in toks:
                done[f[1]] = time.time()
                maybe_read()   # (answers of under 16 tokens: the read goes when they are done)
        rate = lambda ts, a, b: len(ts) / (b - a) if ts and b > a else None   # noqa: E731
        during_rates, after_rates, gaps = [], [], []
        for k, ts in toks.items():
            if done[k] <= t_read:   # (done before the read began: not an answer beside it)
                continue
            during = [t for t in ts if t_read <= t <= t_read_done]
            after = [t for t in ts if t > t_read_done]
            during_rates.append(rate(during, t_read, min(t_read_done, done[k])))
            after_rates.append(rate(after, t_read_done, done[k]))
            # the answer's longest pause while the other prompt was read: its last token before the read began (or
            # the read's start) to the next one - what someone watching that answer sees as a freeze
            marks = [max([t for t in ts if t < t_read] or [t_read])] + during + after[:1]
            gaps += [b - a for a, b in zip(marks, marks[1:])]
        mean = lambda v: st.fmean([x for x in v if x is not None]) if any(x is not None for x in v) else None   # noqa: E731
        return {"read": read_ms, "decode": mean(during_rates), "decode_after": mean(after_rates),
                "gap": max(gaps) * 1000.0 if gaps else None, "read_lp": read_lp}

    def together(self, first: dict[str, str], later: dict[str, str], timeout: float) -> dict:
        """Requests by rid: `first` (rid -> GEN line) sent now, `later` once every first one has 16 tokens or is done.
        -> per rid: its tokens, its LP lines as sent (the digits as the engine wrote them) and its Done."""
        out = {rid: {"tokens": [], "lp": [], "done": None} for rid in [*first, *later]}
        send = lambda line: (self.p.stdin.write(line), self.p.stdin.flush())   # noqa: E731
        until = time.time() + timeout
        for line in first.values():
            send(line)
        pending = dict(later)
        while any(v["done"] is None for v in out.values()):
            if pending and all(len(out[r]["tokens"]) >= 16 or out[r]["done"] is not None for r in first):
                for line in pending.values():
                    send(line)
                pending = {}
                continue
            f = self.line(until).split()
            if not f or f[0] not in ("T", "LP", "DONE", "ERR") or len(f) < 2 or f[1] not in out:
                continue
            if f[0] == "ERR":
                raise LabError("the engine refused a request: " + " ".join(f[1:]))
            o = out[f[1]]
            if f[0] == "T":
                o["tokens"].append(int(f[-1]))
            elif f[0] == "LP":
                o["lp"].append(" ".join(f[2:]))
            else:
                o["done"] = parse_done(f, True)
        return out

    def pause_check(self, read_ids: list[int], answer_ids: list[int], n_answer: int, timeout: float,
                    flush: list[list[int]]) -> dict:
        """Does a prompt read that pauses between its layers (DS4_LAYER_YIELD) compute what the same read alone does? In
        this one engine (with --no-expert-cache nothing moves between the reads): the prompt read alone and an answer
        alone; before those and again after them, every conversation slot takes an unrelated short prompt of `flush`
        (one a slot: the scheduler gives a request the idle slot it shares the most with, which would hand a read
        an earlier one's state); then the same answer with the same prompt read meanwhile in another conversation. -> whether the read's
        top log-probabilities and the answer's tokens and log-probabilities match, each to the last digit, and the
        tokens each reused (a read or an answer that reused any is no full read: not comparable)."""
        line = lambda rid, n, ids, k: f"GEN {n} temperature=0 rid={rid} logprobs={k} " + ",".join(map(str, ids)) + "\n"   # noqa: E731
        def flush_slots(base: int) -> None:   # (every slot an unrelated prompt: nothing left to reuse)
            for i, ids in enumerate(flush):
                self.together({str(base + i): line(base + i, 1, ids, 0)}, {}, timeout)

        flush_slots(20)
        alone = self.together({"5": line(5, 1, read_ids, 5)}, {}, timeout)["5"]
        ans = self.together({"6": line(6, n_answer, answer_ids, 2)}, {}, timeout)["6"]
        flush_slots(30)
        both = self.together({"7": line(7, n_answer, answer_ids, 2)}, {"8": line(8, 1, read_ids, 5)}, timeout)
        read2, ans2 = both["8"], both["7"]
        k = next((i for i, (a, b) in enumerate(zip(ans["lp"], ans2["lp"])) if a != b), None)
        return {"read_same": alone["lp"] == read2["lp"], "read_alone": alone["lp"], "read_paused": read2["lp"],
                "reused": [alone["done"].reused, read2["done"].reused, ans["done"].reused, ans2["done"].reused],
                "answer_same": ans["tokens"] == ans2["tokens"] and ans["lp"] == ans2["lp"],
                "answer_first_diff": k, "answer_tokens": [len(ans["tokens"]), len(ans2["tokens"])]}

    def close(self) -> None:
        try:
            if self.p.poll() is None:
                self.p.stdin.write("QUIT\n")
                self.p.stdin.flush()
            self.p.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()
        for f in (self.p.stdin, self.p.stdout, self.log):
            try:
                f.close()
            except OSError:
                pass


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


# ---------------------------------------------------------------- a run
def say(msg: str) -> None:
    print("lab: " + msg, flush=True)


def load_runs(runs: Path, label: str, signature: str | None = None) -> list[dict]:
    out = []
    for p in sorted((runs / label).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if signature is None or r.get("signature") == signature:
            out.append(r)
    return out


def engine_setup(a, cfg: dict):
    """The engine's command, environment and the model family, as the server starts it, with this run's changes."""
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import maya                    # noqa: E402  (the config's family, its chat format)
    import serve.server as SV      # noqa: E402  (the server's own engine command and environment)
    family = maya.config_family(cfg)
    env_cfg = dict(cfg.get("env") or {})
    if family == "deepseek":
        env_cfg["MAYA_DS4_SLOT_KEEP"] = "0"
    for kv in a.env:
        k, _, v = kv.partition("=")
        env_cfg[k] = v
    cfg = {**cfg, "env": env_cfg}
    exe = a.exe or cfg["exe"]
    args = SV.engine_args(cfg) + list(a.arg)
    if family == "deepseek" and "--profile-readonly" not in args:
        args.append("--profile-readonly")
    if family == "deepseek" and a.expert_profile:
        # (the server saves its expert profile at every stop, and the profile places the experts at the next start -
        #  which side computes an expert, so the order of the sums: runs compared with each other start from one copy)
        args += ["--expert-profile", str(Path(a.expert_profile).resolve())]
    return [str(exe), "--serve"] + args, SV.child_env(cfg), cfg, family, maya


@dataclass
class Opts:
    """What measure() needs of the command line (tests build one directly)."""
    gate: str = "changed"          # (greedy answers inform; lab_quality's logits gate - docs/LAB.md)
    keep_going: bool = False
    answer: int = ANSWER
    load_timeout: float = 300.0
    request_timeout: float = 180.0
    deadline: float = 900.0
    weights: dict = field(default_factory=dict)   # read size -> share of real turns (TTFT per turn)
    mixed: bool = True
    end_ids: tuple = ()                            # the tokens that end an answer (banned in the long answer)


class Run:
    """One run's results as it goes (res: what the JSON file keeps) and its checks against the reference's runs."""

    def __init__(self, res: dict, ref: list[dict], o: Opts, profiled: bool = False):
        self.res, self.ref, self.o = res, ref, o
        self.speed = "same"
        self.answers: str | None = None
        self.profiled = profiled   # (a profiled engine is slower: its speeds are kept, not judged)

    def check(self, name: str, values: list[float], first: bool = False) -> bool:
        """Keep a metric and compare it with the reference's (`first`: with the reference's first values only, as many
        as there are - a step's early look); False: stop here (worse, and not --keep-going)."""
        self.res["metrics"][name] = values
        higher, rel = metric_kind(name)
        unit = "tok/s" if higher else "ms"
        shown = ", ".join(f"{v:.1f}" if higher else f"{v:.0f}" for v in values)
        rv = [v for r in self.ref for v in (r["metrics"].get(name) or [])[:len(values) if first else None]]
        if self.profiled:
            say(f"{name}: {shown} {unit} (profiled: not judged)")
            return True
        if not rv:
            say(f"{name}: {shown} {unit}" + (" (the reference has none)" if self.ref else ""))
            return True
        verdict, delta = judge(values, rv, higher, rel, ABS_FLOOR.get(name, 0.0))
        self.res["verdicts"][name] = [verdict, round(delta, 1)]
        say(f"{name}: {shown} {unit} against {st.fmean(rv):.{1 if higher else 0}f} ({delta:+.1f}%): {verdict}")
        if verdict == "worse":
            self.speed = "worse"
        elif verdict == "better" and self.speed == "same":
            self.speed = "better"
        if verdict == "worse" and not self.o.keep_going:
            self.res["stopped"] = f"{name} worse ({delta:+.1f}%)"
            return False
        return True


def engine_pid(pid: int, name: str) -> int:
    """The engine's own process under `pid` (itself, or a child of a wrapper such as rocprofv3), by its binary's
    name - Linux /proc; `pid` where none is found."""
    todo = [pid]
    while todo:
        p = todo.pop(0)
        try:
            if Path(f"/proc/{p}/exe").resolve().name == name:
                return p
            for t in Path(f"/proc/{p}/task").iterdir():
                todo += [int(c) for c in (t / "children").read_text().split()]
        except OSError:
            continue
    return pid


def measure(eng: Engine, w: Workload, r: Run, prof=None) -> int:
    """The run's steps on a started engine, each checked as soon as it is measured: load, the answers (and their
    decode speed), the shared start's read, the reads on top of it - with --profile, each of the two parts profiled
    as a step ("answers", "reads"). 0: every step ran; 3: a speed got worse; 4: the answers fell below --gate (both: not
    --keep-going); 2: the engine failed (LabError)."""
    from contextlib import nullcontext
    step = prof.step if prof is not None else (lambda _name: nullcontext())
    o, res, t0 = r.o, r.res, time.time()
    deadline = t0 + o.deadline
    left = lambda: max(1.0, min(o.request_timeout, deadline - time.time()))   # noqa: E731
    try:
        eng.ready(max(1.0, min(o.load_timeout, deadline - time.time())))
        res["metrics"]["load_s"] = [round(time.time() - t0, 1)]
        say(f"loaded in {time.time() - t0:.0f} s")
        if prof is not None:
            prof.pid = engine_pid(eng.p.pid, Path(eng.exe).name)
        _, ans = eng.gen(w.warm, 32, left(), logprobs=2)
        answers, rates = [ans], []
        res["cpu_windows"] = {"answers": [], "rates": []}   # (each answer's start, end and speed, for the CPU trace)
        with step("answers"):
            for q in w.questions:
                t_a = time.time()
                done, ans = eng.gen(q, o.answer, left(), logprobs=2)
                res["cpu_windows"]["answers"].append([t_a, time.time()])
                answers.append(ans)
                rate = done.produced / done.decode_ms * 1000.0 if done.decode_ms > 0 else None
                res["cpu_windows"]["rates"].append(rate)
                if done.produced >= min(64, o.answer) and rate is not None:
                    rates.append(rate)
        # the shared start, read cold, and its answer (a long prompt's: the prompt skip of 3.7K tokens and more acts)
        done, ans = eng.gen(w.prefix, LONG_ANSWER, left(), shared=w.shared, logprobs=2)
        answers.append(ans)
        res["answers"] = [x.to_json() for x in answers]
        if r.ref:
            level, why = compare_answers([Answer.from_json(x) for x in r.ref[0]["answers"]], answers)
            r.answers = level
            res["verdicts"]["answers"] = [level, why]
            say(f"answers: {level} - {why}")
            if LEVELS.index(level) < LEVELS.index(o.gate) and not o.keep_going:
                res["stopped"] = f"answers {level}, below --gate {o.gate}"
                return 4
        if not r.check("decode", rates):
            return 3
        if not r.check("read_prefix", [done.prompt_ms]):
            return 3
        by_n: dict[int, list[float]] = {}
        sizes = len(set(n for n, _ in w.reads))
        if w.warm_read:   # (unmeasured: the first read after the shared start is often slow)
            eng.gen(w.warm_read, 1, left())
        with step("reads"):
            for i, (n, ids) in enumerate(w.reads):
                done, _ = eng.gen(ids, 1, left())
                if done.reused != w.shared:
                    say(f"read{n}: {done.reused} tokens reused, not the shared start's {w.shared} - the engine read "
                        "more than the new text")
                by_n.setdefault(n, []).append(done.prompt_ms)
                if i < sizes and r.ref and not r.profiled and not r.check(f"read{n}", by_n[n], first=True):
                    return 3
        for n, v in by_n.items():
            if not r.check(f"read{n}", v):
                return 3
        t = turn_ms(by_n, o.weights)
        if t is not None and not r.check("turn", [t]):
            return 3
        # the mixed steps: one answer (and with --mixed2 two) while a fresh prompt is read
        for name, decodes, read_ids in (("mixed", [w.mixed_q], w.mixed_read),
                                        ("mixed2", [w.mixed_q, w.questions[0]], w.mixed_read2)):
            if not o.mixed or not read_ids:
                continue
            if not eng.multi:
                say(f"{name}: skipped - the engine runs one conversation (--parallel 1)")
                continue
            if eng.slots < len(decodes) + 1:
                say(f"{name}: skipped - it needs --parallel {len(decodes) + 1}, the engine runs {eng.slots}")
                continue
            with step(name):
                mx = eng.mixed(decodes, read_ids, MIXED_ANSWER, left())
            if mx["decode_after"]:
                res["metrics"][f"{name}_decode_after"] = [mx["decode_after"]]
            if mx["read_lp"]:
                res[f"{name}_read_lp"] = mx["read_lp"]
            for part in ("decode", "read", "gap"):
                if mx[part] is not None and not r.check(f"{name}_{part}", [mx[part]]):
                    return 3
        if w.long_read:   # (a long fresh prompt: the chunks' size and what they borrow show at this length)
            with step("long_read"):
                done, _ = eng.gen(w.long_read, 1, left())
            if not r.check("read_long", [done.prompt_ms]):
                return 3
        if w.check_read:   # (a paused read against the same read alone, to the last digit: a gate)
            if not eng.multi or eng.slots < 2 or eng.slots > 1 + len(w.questions):
                say(f"pause check: skipped - it needs 2 to {1 + len(w.questions)} conversations (--parallel), the engine runs {eng.slots}")
            else:
                with step("pause_check"):
                    pc = eng.pause_check(w.check_read, w.mixed_q, MIXED_ANSWER, max(left(), 3 * o.request_timeout),
                                         [w.warm, *w.questions][:max(eng.slots, 1)])
                res["pause_check"] = pc
                ok = pc["read_same"] and pc["answer_same"] and not any(pc["reused"])
                say(f"pause check: the read {'the same' if pc['read_same'] else 'DIFFERENT'} "
                    f"({' / '.join(x.split()[0] + ' ' + x.split()[1] for x in pc['read_alone'][:1] + pc['read_paused'][:1])}), "
                    f"the answer {'the same' if pc['answer_same'] else 'DIFFERENT from token ' + str(pc['answer_first_diff'])}"
                    + (f", reused {pc['reused']} - not comparable" if any(pc["reused"]) else ""))
                if not ok:
                    res["stopped"] = ("pause check: a read reused tokens, not comparable" if any(pc["reused"]) else
                                      "pause check: a paused read or the answer beside it differs")
                    return 4
        if w.long_q:      # (one answer past a CPU power limit's burst: its speed slice by slice, the CPU's beside it)
            times: list[float] = []
            keys = {"logit_bias": ",".join(f"{i}:-100" for i in o.end_ids)} if o.end_ids else {}
            allow = max(1.0, min(deadline - time.time(), max(o.request_timeout, w.long_n / 5.0)))   # (>= 5 tok/s)
            with step("long_answer"):
                done, _ = eng.gen(w.long_q, w.long_n, allow, times=times, **keys)
            res["long_answer"] = {"tokens": done.produced, "t0": times[0] if times else None,
                                  "times": [round(t - times[0], 3) for t in times]}
            if done.produced < w.long_n:
                say(f"long answer: {done.produced} of {w.long_n} tokens ({done.finish}) - the end tokens were not "
                    "all banned")
            first, last = edge_rates(times)
            for name, v in (("long_decode", done.produced / done.decode_ms * 1000.0 if done.decode_ms > 0 else None),
                            ("long_decode_start", first), ("long_decode_end", last)):
                if v is not None and not r.check(name, [v]):
                    return 3
        return 0
    except LabError as e:
        res["error"] = str(e)
        say("error: " + str(e))
        return 2


def stopped_by_signal(signum, _frame):
    """SIGTERM / SIGHUP (systemd stopping the unit, a hang-up): a LabError, so the run still writes its JSON and its
    `lab: end` line - a run that ends never goes silent."""
    raise LabError(f"stopped from outside (signal {signum}: the unit was stopped, or the session hung up)")


def run(a) -> int:
    import signal
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stopped_by_signal)
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cmd, env, cfg, family, maya = engine_setup(a, cfg)
    import strata_tokenizer as ST  # noqa: E402
    pack_args = cfg.get("args") or []
    tp = Path(cfg.get("tokenizer") or (Path(pack_args[pack_args.index("--pack") + 1]) / "tokenizer"
                                       if "--pack" in pack_args[:-1] else ""))
    tok = ST.Tokenizer.from_dir(tp)
    chat = Chat(maya.bench_chat(family, tp), tok)
    text = (Path(a.text).read_text(encoding="utf-8", errors="replace") if a.text else
            "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                        for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) if p.exists()))
    w = build_workload(chat, tok.encode(text), a.prefix, tuple(int(x) for x in a.reads.split(",")), a.rounds,
                       mixed2=a.mixed2, long_read=a.long_read, long_answer=a.long_answer, pause_check=a.pause_check)
    end_ids = tuple(sorted({i[0] for s in END_TOKENS if len(i := tok.encode(s, parse_special=True)) == 1}))
    if a.long_answer and not end_ids:
        raise SystemExit(f"--long-answer: the tokenizer knows none of {', '.join(END_TOKENS)} - the answer could end "
                         "early, and its speed past the CPU's burst would go unmeasured")
    sig = w.signature()
    runs = Path(a.runs)
    ref = load_runs(runs, a.against, sig) if a.against else []
    if a.against and not ref:
        say(f"no runs of '{a.against}' with this workload ({sig}) in {runs}: nothing to compare with")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (runs / a.label).mkdir(parents=True, exist_ok=True)
    out_path, log_path = runs / a.label / f"{stamp}.json", runs / a.label / f"{stamp}.log"
    res = {"label": a.label, "when": stamp, "config": str(cfg_path), "exe": cmd[0], "env": a.env, "arg": a.arg,
           "signature": sig, "metrics": {}, "verdicts": {}, "answers": [], "stopped": None, "error": None,
           "log": str(log_path)}
    say(f"run '{a.label}' ({sig}): {cmd[0]}" + (f" with {' '.join(a.env)}" if a.env else "") +
        (f" {' '.join(a.arg)}" if a.arg else "") + (f", against '{a.against}' ({len(ref)} runs)" if ref else ""))
    weights = {int(k): float(v) for k, v in json.loads(Path(a.weights).read_text(encoding="utf-8")).items()} \
        if a.weights else {}
    o = Opts(a.gate, a.keep_going, a.answer, a.load_timeout, a.request_timeout, a.deadline, weights, not a.no_mixed,
             end_ids)
    if a.lane_cal:
        # (DeepSeek: the PCIe lane's times file - the engine keeps its verdict there for the next start, by default the
        #  pack's lane-times.txt, which the server and every run share: each run starts from its own copy of one file)
        lane = runs / a.label / f"{stamp}-lane-times.txt"
        shutil.copyfile(a.lane_cal, lane)
        env["DS4_LANE_CAL"] = str(lane.resolve())
        res["lane_cal"] = str(Path(a.lane_cal).resolve())
    if a.calm_load > 0:
        waited = wait_calm(a.calm_load, a.calm_wait)
        res["calm_wait_s"] = round(waited)
        if waited >= 1:
            say(f"waited {waited:.0f} s for the box to calm down (load below {a.calm_load})")
    res["conditions"] = {"start": conditions()}
    say("conditions: " + ", ".join(f"{k} {v}" for k, v in res["conditions"]["start"].items()))
    kinds = {k for k in (a.profile or "").split(",") if k}
    if kinds - {"cpu", "mem", "gpu"}:
        raise SystemExit(f"--profile: cpu, mem and/or gpu, not {','.join(sorted(kinds - {'cpu', 'mem', 'gpu'}))}")
    prof, gpu_dir = None, runs / a.label / f"{stamp}-gpu"
    if kinds:
        import lab_profile as LP  # noqa: E402
        res["profile"] = sorted(kinds)
        prof = LP.Profiler(kinds, runs / a.label / stamp, say)
        if "gpu" in kinds:
            cmd = LP.wrap_gpu(cmd, gpu_dir)
    r = Run(res, ref, o, profiled=bool(kinds))
    t0 = time.time()
    trace = CpuTrace(a.cpu_every).start() if a.cpu_every > 0 else None
    try:
        eng = Engine(cmd, env, cfg.get("cwd") or str(ROOT), log_path)
        eng.exe = res["exe"]   # (the engine's own binary, under a wrapper too)
    except LabError as e:
        res["error"], code, eng = str(e), 2, None
        say("error: " + str(e))
    try:
        if eng is not None:
            code = measure(eng, w, r, prof)
    finally:
        if eng is not None:
            eng.close()
        if trace is not None:
            trace.stop()
            cpu_summary(res, trace)
        if prof is not None:
            res["profile_steps"] = prof.notes
            if "gpu" in kinds:
                res["gpu_top"] = LP.gpu_summary(gpu_dir, say)
        res["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(res) + "\n", encoding="utf-8")
    if a.quality and not res["error"]:
        import lab_quality as LQ  # noqa: E402  (numpy)
        if a.lane_cal:   # (the quality's loads start from the frozen lane file too, not the run's verdict)
            shutil.copyfile(a.lane_cal, runs / a.label / f"{stamp}-lane-times.txt")
        base_cmd = [res["exe"]] + engine_setup(a, json.loads(cfg_path.read_text(encoding="utf-8")))[0][2:]
        try:
            res["quality"] = LQ.run_quality(base_cmd, env, cfg.get("cwd") or str(ROOT), w.quality, out_path.with_suffix(""),
                                            runs, sig, ref, say)
        except LabError as e:
            res["error"] = str(e)
            say("error: " + str(e))
        bad = [m for m, q in res["quality"].items() if q.get("verdict") == "changed" or "error" in q] \
            if "quality" in res else []
        if bad and code == 0:
            res["stopped"], code = f"quality {', '.join(bad)} changed beyond the noise floor", 4
    res["conditions"]["end"] = conditions()
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res) + "\n", encoding="utf-8")
    q = res.get("quality") or {}
    qtxt = ", quality " + "/".join(f"{m} {v.get('verdict', 'measured')}" for m, v in q.items()) if q else ""
    end = (f"error: {res['error'].splitlines()[0]}" if res["error"] else f"stopped: {res['stopped']}" if res["stopped"] else
           f"answers {r.answers or 'not compared'}, speed {r.speed if ref else 'not compared'}{qtxt}")
    say(f"end '{a.label}' in {res['seconds']:.0f} s - {end} ({out_path})")
    return code


# ---------------------------------------------------------------- history (#718)
HISTORY = 6000   # tokens of a history prompt and of the probe (>= ~3.7K: the prompt skip acts on both)


def history(a) -> int:
    """`history`: does a long prompt's answer depend on what the engine read before it? One process reads history
    --which (1 or 2: another long fresh prompt each), then the probe (a fixed long fresh prompt), and writes the probe's
    greedy answer with its log-probabilities. Run it once with each history, then `history-compare`: the same answer
    both times - the probe sees nothing of its history; another - something earlier requests left (the DS4_YOCO=2
    replay reading stale window rows, issue #718) or ordinary noise (compare with the engine's own spread)."""
    import signal
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stopped_by_signal)
    cfg_path = Path(a.config).resolve()
    a.arg = list(a.arg)
    cmd, env, cfg, family, maya = engine_setup(a, json.loads(cfg_path.read_text(encoding="utf-8")))
    import strata_tokenizer as ST  # noqa: E402
    pack_args = cfg.get("args") or []
    tp = Path(cfg.get("tokenizer") or (Path(pack_args[pack_args.index("--pack") + 1]) / "tokenizer"))
    tok = ST.Tokenizer.from_dir(tp)
    chat = Chat(maya.bench_chat(family, tp), tok)
    before, after = chat.split_ids()
    doc = tok.encode(Path(a.text).read_text(encoding="utf-8", errors="replace"))
    if len(doc) < 4 * HISTORY:
        raise SystemExit(f"--text has {len(doc)} tokens; the history test needs {4 * HISTORY}")
    ask = chat.text_ids(ASK) + after
    # (from the text's end: no overlap with the run workload's slices, which come from its start)
    probe = before + doc[-HISTORY:] + ask
    hist = before + doc[-(1 + a.which) * HISTORY:-a.which * HISTORY] + ask
    runs = Path(a.runs)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (runs / a.label).mkdir(parents=True, exist_ok=True)
    out = runs / a.label / f"{stamp}-history{a.which}.json"
    res = {"label": a.label, "which": a.which, "exe": cmd[0], "env": a.env, "answer": None, "error": None}
    say(f"history '{a.label}' #{a.which}: {cmd[0]}" + (f" with {' '.join(a.env)}" if a.env else ""))
    eng = Engine(cmd, env, cfg.get("cwd") or str(ROOT), out.with_suffix(".log"))
    try:
        eng.ready(a.load_timeout)
        h, _ = eng.gen(hist, 1, a.request_timeout)
        d, ans = eng.gen(probe, 64, a.request_timeout, logprobs=2)
        res["answer"] = ans.to_json()
        res["read_ms"] = [h.prompt_ms, d.prompt_ms]
        say(f"history #{a.which}: history read {h.prompt_ms:.0f} ms, probe read {d.prompt_ms:.0f} ms, "
            f"{len(ans.tokens)} tokens")
    except LabError as e:
        res["error"] = str(e)
        say("error: " + str(e))
    finally:
        eng.close()
        out.write_text(json.dumps(res) + "\n", encoding="utf-8")
    say(f"end history '{a.label}' #{a.which} - {'error' if res['error'] else 'ok'} ({out})")
    return 2 if res["error"] else 0


def history_compare(a) -> int:
    rs = {}
    for p in sorted((Path(a.runs) / a.label).glob("*-history*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("answer"):
            rs[r["which"]] = Answer.from_json(r["answer"])
    if 1 not in rs or 2 not in rs:
        raise SystemExit(f"'{a.label}' needs a history run with --which 1 and one with --which 2")
    level, why = compare_answers([rs[1]], [rs[2]])
    print(f"{a.label}: the probe after history 1 vs after history 2: {level} - {why}")
    return 0


# ---------------------------------------------------------------- show
def show(a) -> int:
    runs = Path(a.runs)
    labels = [(lb, load_runs(runs, lb)) for lb in a.labels]
    if not labels[0][1]:
        raise SystemExit(f"no runs of '{labels[0][0]}' in {runs}")
    sig = labels[0][1][0]["signature"]
    names = [n for n in labels[0][1][0]["metrics"] if n != "load_s"]
    print(f"{'':14}" + "".join(f"{lb:>24}" for lb, _ in labels))
    for name in names:
        higher, rel = metric_kind(name)
        ref = [v for r in labels[0][1] if r["signature"] == sig for v in r["metrics"].get(name) or []]
        row = f"{name:14}"
        for i, (lb, rs) in enumerate(labels):
            v = [x for r in rs if r["signature"] == sig for x in r["metrics"].get(name) or []]
            if not v:
                row += f"{'-':>24}"
                continue
            cell = f"{st.fmean(v):.1f}" + (f"±{st.stdev(v):.1f}" if len(v) > 1 else "")
            if i > 0:
                verdict, delta = judge(v, ref, higher, rel)
                cell += f" {delta:+.0f}%" + ("*" if verdict != "same" else " ")
            row += f"{cell:>24}"
        print(row)
    for lb, rs in labels:
        lv = [r["verdicts"].get("answers", ["-"])[0] for r in rs]
        print(f"{lb}: {len(rs)} runs; answers {', '.join(lv)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one run: load, answer, read; compare with --against")
    r.add_argument("--config", required=True, help="the server's config (maya-*.json): its engine, arguments, env")
    r.add_argument("--label", required=True, help="this run's label (its folder under --runs)")
    r.add_argument("--exe", help="another engine binary (a patched build)")
    r.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine, over the config's")
    r.add_argument("--arg", action="append", default=[], help="an engine argument after the config's")
    r.add_argument("--against", help="the reference label: compare every step and stop when one fails")
    r.add_argument("--gate", choices=LEVELS, default="last bits", help="the answers' level a run needs to go on")
    r.add_argument("--keep-going", action="store_true", help="measure every step, whatever the verdicts")
    r.add_argument("--prefix", type=int, default=PREFIX, help="the shared start's tokens")
    r.add_argument("--reads", default=",".join(map(str, READS)), help="new tokens of each read, comma-separated")
    r.add_argument("--rounds", type=int, default=2, help="rounds of the reads")
    r.add_argument("--answer", type=int, default=ANSWER, help="tokens each question's answer writes")
    r.add_argument("--text", help="the reads' text (default: README.md and docs/*.md - a file of its own keeps the "
                                  "workload, and so the comparisons, the same when the docs change)")
    r.add_argument("--load-timeout", type=float, default=300.0)
    r.add_argument("--request-timeout", type=float, default=180.0)
    r.add_argument("--deadline", type=float, default=900.0, help="the whole run's seconds at most")
    r.add_argument("--calm-load", type=float, default=1.0, help="start only once the 1-minute load is below this "
                                                                   "(0: do not wait)")
    r.add_argument("--calm-wait", type=float, default=300.0, help="seconds to wait for --calm-load before stopping")
    r.add_argument("--runs", default=str(ROOT / "lab-runs"))
    r.add_argument("--expert-profile", help="DeepSeek: a frozen copy of the expert profile every compared run starts "
                                            "from (the server rewrites its own at each stop, which moves the answers' "
                                            "last bits between runs)")
    r.add_argument("--profile", help="cpu,mem,gpu: where the time goes (tools/lab_profile.py); speeds not judged")
    r.add_argument("--weights", help="a JSON of read size -> share of real turns (`lab.py weights LOG`): the TTFT per "
                                     "turn metric")
    r.add_argument("--quality", action="store_true", help="after the run, the logits of a fixed text against the "
                                                         "reference's (tools/lab_quality.py; two more engine loads)")
    r.add_argument("--no-mixed", action="store_true", help="leave out the mixed step (one conversation decoding while "
                                                          "another reads)")
    r.add_argument("--mixed2", action="store_true", help="a second mixed step: two conversations answer while a third "
                                                        "reads (--parallel 3)")
    r.add_argument("--long-read", type=int, default=0, help="a fresh prompt of this many tokens, read after the mixed "
                                                           "steps (read_long)")
    r.add_argument("--long-answer", type=int, default=0, help="one answer of this many tokens, its end tokens banned, "
                                                             "last: its speed slice by slice beside the CPU's clock, "
                                                             "temperature and power (long_decode*)")
    r.add_argument("--lane-cal", help="DeepSeek: a frozen copy of the PCIe lane's times file (the pack's "
                                      "lane-times.txt) every compared run starts from (DS4_LANE_CAL: the engine keeps "
                                      "its lane verdict there for the next start)")
    r.add_argument("--cpu-every", type=float, default=1.0, help="seconds between the CPU trace's samples (0: none)")
    r.add_argument("--pause-check", action="store_true", help="a prompt read beside an answer (DS4_LAYER_YIELD pauses it) "
                                                             "against the same read alone, and the answer against "
                                                             "itself alone, to the last digit - a gate; run it with "
                                                             "--arg=--no-expert-cache (nothing moves between them)")
    s = sub.add_parser("show", help="labels' runs side by side, the first label the reference")
    s.add_argument("labels", nargs="+")
    s.add_argument("--runs", default=str(ROOT / "lab-runs"))
    wp = sub.add_parser("weights", help="the share of real turns of each read size, from a server log")
    wp.add_argument("log")
    wp.add_argument("--reads", default=",".join(map(str, READS)))
    hp = sub.add_parser("history", help="does a long prompt's answer depend on the requests before it? (#718)")
    for x in ("--config", "--label", "--text"):
        hp.add_argument(x, required=True)
    hp.add_argument("--which", type=int, choices=(1, 2), required=True, help="the history read first")
    hp.add_argument("--exe")
    hp.add_argument("--env", action="append", default=[])
    hp.add_argument("--arg", action="append", default=[])
    hp.add_argument("--expert-profile")
    hp.add_argument("--load-timeout", type=float, default=300.0)
    hp.add_argument("--request-timeout", type=float, default=300.0)
    hp.add_argument("--runs", default=str(ROOT / "lab-runs"))
    hc = sub.add_parser("history-compare", help="a history label's two runs against each other")
    hc.add_argument("label")
    hc.add_argument("--runs", default=str(ROOT / "lab-runs"))
    mp = sub.add_parser("matrix", help="knobs judged together (tools/lab_matrix.py): plan a round, report it")
    mp.add_argument("action", choices=("plan", "report"))
    mp.add_argument("spec")
    mp.add_argument("--runs", default=str(ROOT / "lab-runs"))
    a = ap.parse_args()
    if a.cmd == "weights":
        sizes = tuple(int(x) for x in a.reads.split(","))
        with open(a.log, encoding="utf-8", errors="replace") as f:
            print(json.dumps(weights_from_log(f, sizes)))
        return 0
    if a.cmd == "history":
        return history(a)
    if a.cmd == "history-compare":
        return history_compare(a)
    if a.cmd == "matrix":
        import lab_matrix as LM  # noqa: E402
        spec = LM.load_spec(a.spec)
        print(LM.plan(spec) if a.action == "plan" else LM.report(spec, Path(a.runs), load_runs, judge, metric_kind))
        return 0
    return run(a) if a.cmd == "run" else show(a)


if __name__ == "__main__":
    sys.exit(main())
