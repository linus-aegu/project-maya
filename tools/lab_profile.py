"""tools/lab_profile.py - tools/lab.py's --profile: where a run's time goes, by step.

- cpu: `perf record` attached to the engine for each measured step (the answers, the reads), then a flame graph per
  step (inferno: `cargo install inferno`) and the hottest symbols (`perf report --stdio`);
- mem: the RAM's own traffic during each step, from the memory controller's counters (`perf stat -a` on
  uncore_imc data_reads / data_writes, Intel client CPUs): GB/s read and written - whether a step is bound by the RAM;
- gpu: the engine under `rocprofv3 --kernel-trace --stats` (AMD; the whole run - rocprofv3 cannot attach): each
  kernel's calls and time, the top ones in a table; the trace opens in ui.perfetto.dev.

Profiling slows the engine, so a profiled run's speeds are kept but not judged. perf needs
kernel.perf_event_paranoid <= 1 (or root). Files go next to the run's JSON: <stamp>-<step>.svg / .top.txt, <stamp>-gpu/.
"""
from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

IMC = ("uncore_imc/data_reads/", "uncore_imc/data_writes/")


def wrap_gpu(cmd: list[str], out: Path) -> list[str]:
    """The engine's command under rocprofv3's kernel trace (the whole run)."""
    exe = shutil.which("rocprofv3")
    if exe is None:
        raise RuntimeError("--profile gpu needs rocprofv3 (ROCm) on PATH")
    out.mkdir(parents=True, exist_ok=True)
    return [exe, "--kernel-trace", "--stats", "--output-format", "csv", "pftrace", "-d", str(out), "-o", "run", "--"] + cmd


def parse_perf_stat(text: str) -> dict[str, float]:
    """`perf stat -x,` lines -> {event: bytes}. The IMC events come scaled (MiB) or as 64-byte lines."""
    out = {}
    for line in text.splitlines():
        f = line.split(",")
        if len(f) < 3 or not f[0].strip() or f[0].startswith("<"):
            continue
        try:
            v = float(f[0])
        except ValueError:
            continue
        unit, ev = f[1].strip(), f[2].strip()
        if unit == "MiB":
            v *= 1 << 20
        elif unit == "":
            v *= 64.0
        out[ev] = v
    return out


def top_kernels(stats_csv: Path, n: int = 15) -> list[tuple[str, int, float, float]]:
    """rocprofv3's kernel stats CSV -> the top n kernels by total time: (name, calls, total ms, share %)."""
    rows = []
    with open(stats_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["Name"], int(r["Calls"]), float(r["TotalDurationNs"]) / 1e6, float(r.get("Percentage") or 0)))
    rows.sort(key=lambda x: -x[2])
    return rows[:n]


class Profiler:
    """Per-step perf for one engine process: step(name) records it while the block runs."""

    def __init__(self, kinds: set[str], base: Path, say):
        self.kinds, self.base, self.say = kinds, base, say
        self.pid = 0
        self.notes: dict[str, dict] = {}
        for k in kinds & {"cpu", "mem"}:
            if shutil.which("perf") is None:
                raise RuntimeError(f"--profile {k} needs perf on PATH")

    @contextmanager
    def step(self, name: str):
        procs = []
        data = self.base.with_name(f"{self.base.name}-{name}.perf.data")
        if "cpu" in self.kinds and self.pid:
            procs.append(("cpu", subprocess.Popen(
                ["perf", "record", "-F", "499", "--call-graph", "dwarf,16384", "-p", str(self.pid), "-o", str(data)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        if "mem" in self.kinds:
            procs.append(("mem", subprocess.Popen(
                ["perf", "stat", "-a", "-x,", "-e", ",".join(IMC)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            note = self.notes.setdefault(name, {"seconds": round(dt, 1)})
            for kind, p in procs:
                p.send_signal(signal.SIGINT)
                try:
                    _, err = p.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
                    _, err = p.communicate()
                if kind == "mem":
                    b = parse_perf_stat(err)
                    if not b:   # (no counts: perf said why - e.g. perf_event_paranoid > 0 for system-wide counters)
                        self.say(f"profile {name}: RAM counters gave nothing: {err.strip().splitlines()[-1:] or ['?']}")
                    rd, wr = b.get(IMC[0], 0.0) / dt / 1e9, b.get(IMC[1], 0.0) / dt / 1e9
                    note["ram_read_gbs"], note["ram_write_gbs"] = round(rd, 1), round(wr, 1)
                    self.say(f"profile {name}: RAM {rd:.1f} GB/s read, {wr:.1f} GB/s written over {dt:.0f} s")
                elif kind == "cpu":
                    note["cpu"] = self.cpu_outputs(name, data)

    def cpu_outputs(self, name: str, data: Path) -> dict:
        """The step's flame graph (inferno) and its top symbols; returns their paths."""
        out = {}
        top = data.with_suffix(".top.txt")
        r = subprocess.run(["perf", "report", "-i", str(data), "--stdio", "--no-children", "--sort", "symbol",
                            "--percent-limit", "0.5"], capture_output=True, text=True)
        top.write_text(r.stdout, encoding="utf-8")
        out["top"] = str(top)
        lines = [x for x in r.stdout.splitlines() if x.strip() and not x.startswith("#")][:5]
        for x in lines:
            self.say(f"profile {name}: cpu {' '.join(x.split())[:110]}")
        path = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / ".cargo" / "bin")   # (cargo install's)
        coll, fg = shutil.which("inferno-collapse-perf", path=path), shutil.which("inferno-flamegraph", path=path)
        if coll and fg:
            svg = data.with_suffix(".svg")
            script = subprocess.run(["perf", "script", "-i", str(data)], capture_output=True, text=True)
            folded = subprocess.run([coll], input=script.stdout, capture_output=True, text=True)
            with open(svg, "w", encoding="utf-8") as f:
                subprocess.run([fg, "--title", f"maya-ds4: {name}"], input=folded.stdout, stdout=f, text=True)
            out["svg"] = str(svg)
            self.say(f"profile {name}: flame graph {svg}")
        else:
            self.say("profile: no inferno on PATH (cargo install inferno) - perf report only")
        return out


def gpu_summary(out: Path, say, n: int = 15) -> list:
    """The top kernels of a rocprofv3 run (its *kernel_stats.csv), printed and returned."""
    stats = sorted(out.rglob("*kernel_stats.csv"))
    if not stats:
        say(f"profile gpu: no kernel stats in {out}")
        return []
    top = top_kernels(stats[0], n)
    for name, calls, ms, pct in top:
        say(f"profile gpu: {pct:5.1f}% {ms:9.1f} ms {calls:7d} calls  {name[:80]}")
    say(f"profile gpu: trace in {out} (open the .pftrace in ui.perfetto.dev)")
    return top
