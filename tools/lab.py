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
import hashlib
import json
import math
import queue
import re
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
QUALITY = 1024                      # the quality text's tokens (lab_quality.py)
NEAR_TIE = 0.5                      # nats between the reference's two likeliest tokens: a divergence there is a near-tie
LAST_BITS = 0.05                    # mean |change| of the common tokens' log-probabilities for `last bits`
NOISE = {"read": 0.10, "decode": 0.08}   # the spread assumed for a side of one value, of its mean
ABS_FLOOR = {"mixed_decode": 1.0}        # a difference smaller than this never counts (a metric near zero: 0.2 -> 0.4)
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

    def signature(self) -> str:
        h = hashlib.sha256()
        for ids in [self.warm, *self.questions, self.prefix, *(ids for _, ids in self.reads), self.mixed_q,
                    self.mixed_read, self.quality, self.warm_read]:
            h.update(",".join(map(str, ids)).encode())
            h.update(b";")
        return h.hexdigest()[:16]


def build_workload(chat, doc: list[int], prefix: int = PREFIX, sizes=READS, rounds: int = 2, mixed: int = MIXED_READ,
                   quality: int = QUALITY) -> Workload:
    """The requests of a run from `chat` (a question -> its prompt's ids, in the model's chat format, thinking off) and
    `doc` (the text the reads take their tokens from): the same every run of the same arguments. The reads' new text,
    the mixed step's read and the quality text come from slices of `doc` nothing else uses, so nothing but the shared
    start can be reused."""
    before, after = chat.split_ids()
    need = prefix + rounds * sum(sizes) + mixed + quality + 32
    if len(doc) < need:
        raise LabError(f"the text for the reads has {len(doc)} tokens, they need {need}: fewer --reads or --rounds")
    start = before + doc[:prefix]
    ask = chat.text_ids(ASK) + after
    reads, at = [], prefix
    for _ in range(rounds):
        for n in sizes:
            reads.append((n, start + doc[at:at + n] + ask))
            at += n
    mixed_read = before + doc[at:at + mixed] + ask
    at += mixed
    qtext = before[:1] + doc[at:at + quality]
    at += quality
    return Workload(chat(WARMUP), [chat(q) for q in QUESTIONS], start + ask, reads, len(start), chat(MIXED_Q),
                    mixed_read, qtext, start + doc[at:at + 16] + ask)


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
    """(higher is better, the assumed spread) of a metric by its name: decode / mixed_decode in tokens/s, the rest
    (read*, turn, mixed_read, mixed_gap) in ms."""
    return (True, NOISE["decode"]) if name in ("decode", "mixed_decode") else (False, NOISE["read"])


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
        mhz = [float(x.split(":")[1]) for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("cpu MHz")]
        if mhz:
            out["cpu_mhz"] = round(st.fmean(mhz))

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
                return

    def gen(self, ids: list[int], max_new: int, timeout: float, **keys) -> tuple[Done, Answer]:
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
                ans.tokens.append(int(f[-1]))
            elif f[0] == "LP":
                _, lp, top = parse_lp(f, self.multi)
                ans.lp.append(lp)
                ans.gap.append(top[0] - top[1] if len(top) > 1 else 99.0)
            else:
                return parse_done(f, self.multi), ans

    def mixed(self, decode_ids: list[int], read_ids: list[int], n_decode: int, timeout: float) -> dict:
        """One conversation (rid 1) answers while another (rid 2) reads a fresh prompt: the read starts once the answer
        has 16 tokens. -> the answer's tokens/s while the read runs and after it, and the read's time. Needs an engine
        of several conversations (--parallel 2 or more)."""
        if not self.multi:
            raise LabError("the mixed step needs an engine of several conversations (--parallel 2 or more)")
        # both request lines built before the clock starts: the read is sent mid-answer, inside the timed window
        lines = {rid: f"GEN {n} temperature=0 rid={rid} " + ",".join(map(str, ids)) + "\n"
                 for rid, n, ids in ((1, n_decode, decode_ids), (2, 1, read_ids))}
        send = lambda rid: (self.p.stdin.write(lines[rid]), self.p.stdin.flush())   # noqa: E731
        until = time.time() + timeout
        send(1)
        toks, t_read, t_read_done, read_ms, done1 = [], None, None, None, None
        while done1 is None or read_ms is None:
            f = self.line(until).split()
            if not f or f[0] not in ("T", "DONE", "ERR") or len(f) < 2:
                continue
            if f[0] == "ERR":
                raise LabError("the engine refused a mixed-step request: " + " ".join(f[1:]))
            if f[0] == "T" and f[1] == "1":
                toks.append(time.time())
                if len(toks) == 16 and t_read is None:
                    t_read = time.time()
                    send(2)
            elif f[0] == "DONE" and f[1] == "2":
                t_read_done, read_ms = time.time(), parse_done(f, True).prompt_ms
            elif f[0] == "DONE" and f[1] == "1":
                done1 = time.time()
                if t_read is None:   # (an answer of under 16 tokens: the read goes alone)
                    t_read = time.time()
                    send(2)
        during = [t for t in toks if t_read <= t <= t_read_done]
        after = [t for t in toks if t > t_read_done]
        rate = lambda ts, a, b: len(ts) / (b - a) if ts and b > a else None   # noqa: E731
        # the answer's longest pause while the other prompt was read: its last token before the read began (or the
        # read's start) to the next one - what someone watching that answer sees as a freeze
        marks = [max([t for t in toks if t < t_read] or [t_read])] + during + [t for t in after[:1]]
        gap = max((b - a for a, b in zip(marks, marks[1:])), default=None)
        return {"mixed_read": read_ms, "mixed_decode": rate(during, t_read, t_read_done),
                "mixed_decode_after": rate(after, t_read_done, done1),
                "mixed_gap": gap * 1000.0 if gap is not None else None}

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
        with step("answers"):
            for q in w.questions:
                done, ans = eng.gen(q, o.answer, left(), logprobs=2)
                answers.append(ans)
                if done.produced >= min(64, o.answer) and done.decode_ms > 0:
                    rates.append(done.produced / done.decode_ms * 1000.0)
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
        if o.mixed and eng.multi:
            with step("mixed"):
                mx = eng.mixed(w.mixed_q, w.mixed_read, MIXED_ANSWER, left())
            if mx["mixed_decode_after"]:
                res["metrics"]["mixed_decode_after"] = [mx["mixed_decode_after"]]
            for name in ("mixed_decode", "mixed_read", "mixed_gap"):
                if mx[name] is not None and not r.check(name, [mx[name]]):
                    return 3
        elif o.mixed:
            say("mixed: skipped - the engine runs one conversation (--parallel 1)")
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
    w = build_workload(chat, tok.encode(text), a.prefix, tuple(int(x) for x in a.reads.split(",")), a.rounds)
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
    o = Opts(a.gate, a.keep_going, a.answer, a.load_timeout, a.request_timeout, a.deadline, weights, not a.no_mixed)
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
        if prof is not None:
            res["profile_steps"] = prof.notes
            if "gpu" in kinds:
                res["gpu_top"] = LP.gpu_summary(gpu_dir, say)
        res["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(res) + "\n", encoding="utf-8")
    if a.quality and not res["error"]:
        import lab_quality as LQ  # noqa: E402  (numpy)
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
