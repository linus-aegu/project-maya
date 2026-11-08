"""tools/lab_quality.py - tools/lab.py's --quality: whether a change kept the answers' quality, measured on logits
rather than on greedy answers (one near-tie, or a path picked by timing, sends a greedy answer elsewhere for good).

The engine reads one fixed text (BOS + QUALITY tokens of the workload) with every row's logits dumped (`--ids @FILE
--n 0 --dump-logits`), twice: as a prompt (the chunked pass, its path picked by length) and in windows of 16 (the
decode's path, where the RAM skip and the CPU lane's experts act). Each dump is reduced right away - the full one is
~0.5 GB - to a row's log-probabilities on the CANON ids (the reference's K likeliest tokens of that row, fixed by the
first run of the workload), its top token and its loss on the text's next token. Against the reference:

- KL(reference || run) over the canon ids (they hold ~99% of the reference's probability);
- the share of rows whose top token is the reference's;
- the mean change of the loss on the next token (+: worse).

The reference measured against itself is the noise floor; `verdict` calls a run "same" unless its KL is clearly above
the floor or its top tokens agree clearly less often. Needs numpy (the engine's .venv has it).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

K = 256          # canon ids a row
MODES = {"prompt": ["--window", "2048"], "window": ["--window", "16"]}   # (ids mode feeds 16 tokens at a time unless told: > 16 is the chunked pass)


def strip_serve(args: list[str]) -> list[str]:
    """The serve command's engine arguments for a one-shot read: without --serve and --parallel N."""
    out, skip = [], False
    for x in args:
        if skip:
            skip = False
            continue
        if x == "--parallel":
            skip = True
            continue
        if x != "--serve":
            out.append(x)
    return out


def reduce_dump(dump: Path, ids: list[int], canon_path: Path):
    """A dump's rows -> (lp at the canon ids, top token, loss on the next token); the canon file is written from this
    dump when there is none yet (the first run of the workload sets it). The vocabulary from the file's size."""
    import numpy as np
    size = dump.stat().st_size
    rows = next((r for r in (len(ids), len(ids) - 1) if r > 0 and size % (4 * r) == 0), 0)
    if rows == 0:
        raise ValueError(f"{dump}: {size} bytes is not a whole number of rows for {len(ids)} tokens")
    V = size // (4 * rows)
    m = np.memmap(dump, dtype=np.float32, mode="r", shape=(rows, V))
    canon = np.load(canon_path) if canon_path.exists() else None
    n = min(rows, len(ids) - 1)                      # (row t predicts token t + 1)
    lp_c = np.zeros((n, K), np.float32)
    top = np.zeros(n, np.int32)
    nll = np.zeros(n, np.float32)
    new_canon = np.zeros((n, K), np.int32) if canon is None else None
    nxt = np.asarray(ids[1:n + 1])
    for a in range(0, n, 64):
        x = m[a:min(n, a + 64)].astype(np.float64)
        x -= x.max(axis=1, keepdims=True)
        lp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
        b = a + lp.shape[0]
        if new_canon is not None:
            new_canon[a:b] = np.argpartition(-lp, K, axis=1)[:, :K]
        c = new_canon[a:b] if new_canon is not None else canon[a:b]
        lp_c[a:b] = np.take_along_axis(lp, c, axis=1)
        top[a:b] = lp.argmax(axis=1)
        nll[a:b] = -lp[np.arange(lp.shape[0]), nxt[a:b]]
    del m
    if new_canon is not None:
        np.save(canon_path, new_canon)
    return lp_c, top, nll


def compare(ref: dict, new: dict) -> dict:
    """KL(ref || new) over the canon ids, the top tokens' agreement (%) and the next-token loss change (new - ref)."""
    import numpy as np
    n = min(len(ref["top"]), len(new["top"]))
    lr, ln = ref["lp"][:n].astype(np.float64), new["lp"][:n].astype(np.float64)
    kl = float((np.exp(lr) * (lr - ln)).sum(axis=1).mean())
    return {"kl": round(kl, 5), "top1": round(float((ref["top"][:n] == new["top"][:n]).mean() * 100), 2),
            "dnll": round(float((new["nll"][:n] - ref["nll"][:n]).mean()), 5)}


def load(npz: str) -> dict:
    import numpy as np
    z = np.load(npz)
    return {"lp": z["lp"], "top": z["top"], "nll": z["nll"]}


def verdict(cur: dict, floor: dict | None) -> str:
    """'same' or 'changed' for a run's comparison against the reference, given the reference against itself (None:
    no floor measured yet - judged against fixed bounds only)."""
    f_kl = floor["kl"] if floor else 0.0
    f_top = floor["top1"] if floor else 100.0
    if cur["kl"] > max(2.0 * f_kl, f_kl + 0.01) or cur["top1"] < f_top - 2.0:
        return "changed"
    return "same"


def run_quality(cmd: list[str], env: dict, cwd: str, ids: list[int], base: Path, canon_dir: Path, sig: str,
                ref_runs: list[dict], say, timeout: float = 900.0) -> dict:
    """Both reads of the quality text, reduced, compared with the reference's first quality run and judged against
    the floor (the reference's first two quality runs against each other). Returns the result's "quality" entry."""
    import numpy as np
    ids_file = base.with_name(base.name + "-q-ids.txt")
    ids_file.write_text(",".join(map(str, ids)), encoding="utf-8")
    out = {}
    args = strip_serve(cmd[1:])
    qenv = dict(env, MAYA_DS4_PROMPT_PICK="length", STRATA_DS4_PROMPT_PICK="length")   # (the path by length: fixed)
    for mode, extra in MODES.items():
        dump = base.with_name(f"{base.name}-q-{mode}.f32")
        log = base.with_name(f"{base.name}-q-{mode}.log")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            try:
                p = subprocess.run([cmd[0]] + args + ["--ids", "@" + str(ids_file), "--n", "0", "--dump-logits",
                                                      str(dump)] + extra,
                                   env=qenv, cwd=cwd, stdout=lf, stderr=lf, timeout=timeout)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = -1
        if code != 0 or not dump.exists():
            out[mode] = {"error": f"the engine's {mode} read ended with code {code} (log: {log})"}
            say(f"quality {mode}: error - code {code}, see {log}")
            continue
        lp, top, nll = reduce_dump(dump, ids, canon_dir / f"quality-{sig}-{mode}-canon.npy")
        dump.unlink()
        npz = base.with_name(f"{base.name}-q-{mode}.npz")
        np.savez_compressed(npz, lp=lp, top=top, nll=nll)
        entry = {"npz": str(npz), "seconds": round(time.time() - t0, 1), "nll": round(float(nll.mean()), 5)}
        refs = [r["quality"][mode]["npz"] for r in ref_runs if "npz" in (r.get("quality") or {}).get(mode, {})]
        gone = [x for x in refs if not Path(x).exists()]
        if gone:   # (a reference whose reduced logits were removed: left out, said)
            say(f"quality {mode}: {len(gone)} reference file(s) missing, left out: {gone[0]}")
            refs = [x for x in refs if Path(x).exists()]
        if refs:
            cur = compare(load(refs[0]), {"lp": lp, "top": top, "nll": nll})
            floor = compare(load(refs[0]), load(refs[1])) if len(refs) > 1 else None
            entry.update(cur, floor=floor, verdict=verdict(cur, floor))
            say(f"quality {mode}: KL {cur['kl']:.4f}, top-1 {cur['top1']:.1f}%, loss {cur['dnll']:+.4f}"
                + (f" (floor: KL {floor['kl']:.4f}, top-1 {floor['top1']:.1f}%)" if floor else " (no floor yet)")
                + f": {entry['verdict']}")
        else:
            say(f"quality {mode}: loss {entry['nll']:.4f} a token (the first quality run of this workload)")
        out[mode] = entry
    return out
