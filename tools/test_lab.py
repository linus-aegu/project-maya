"""tools/test_lab.py - tools/lab.py: the engine's lines, the workload, the verdicts, and whole runs against a stand-in
engine (a few lines of Python that speak the same GEN / T / LP / DONE lines). No GPU, no model, no network.

    python -m unittest tools.test_lab -v
"""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import lab  # noqa: E402

# the stand-in engine: READY, then for each GEN its tokens (T), their log-probabilities (LP, with logprobs=) and DONE;
# FAKE_* settings make it slow, wrong, near a tie, hang or crash
FAKE = r'''
import os, sys
multi = os.environ.get("FAKE_MULTI") == "1"
ms_tok = float(os.environ.get("FAKE_MS_PER_TOKEN", "0.01"))
shift = int(os.environ.get("FAKE_SHIFT", "-1"))       # the first question's token at this position differs
gap = float(os.environ.get("FAKE_GAP", "3.0"))        # the top two tokens' difference everywhere
crash_at = int(os.environ.get("FAKE_CRASH_AT", "0"))  # the request that ends the process
hang_at = int(os.environ.get("FAKE_HANG_AT", "0"))    # the request that never answers
print("READY 131072 stop" + (" multi=3" if multi else ""), flush=True)
shared, n = None, 0
log = open("gen.log", "a")                            # (each request line, for the tests to read)
for line in sys.stdin:
    line = line.strip()
    log.write(line[:200] + "\n"); log.flush()
    if line == "QUIT":
        break
    f = line.split()
    n += 1
    if n == crash_at:
        sys.stderr.write("fake: crashing on purpose\n")
        sys.exit(7)
    if n == hang_at:
        continue
    max_new, keys = int(f[1]), dict(x.split("=", 1) for x in f[2:-1])
    ids = [int(x) for x in f[-1].split(",")]
    tag = f" {keys.get('rid', '0')}" if multi else ""
    bad = keys.get("rid") == os.environ.get("FAKE_BAD_RID")   # (this conversation's log-probabilities differ)
    reused = shared if shared and ids[:shared] == start[:shared] else 0
    if "shared" in keys:
        shared, start = int(keys["shared"]), ids
    if multi:
        print("T 1 5", flush=True)                    # (another conversation's line: not the tool's)
    for i in range(max_new):
        tok = (sum(ids) + i) % 1000 + (1 if n == 2 and i == shift else 0)
        print(f"T{tag} {tok}")
        if "logprobs" in keys:
            print(f"LP{tag} {tok} {-0.2 if bad else -0.1} {tok}:-0.1 999:{-0.1 - gap}")
    pm = 10 + (len(ids) - reused) * ms_tok
    print(f"DONE{tag} {max_new} {len(ids)} {pm:.1f} {max_new * 10.0:.1f} length 0 0 {reused} 90 100", flush=True)
'''


class Lines(unittest.TestCase):
    def test_done(self):
        d = lab.parse_done("DONE 128 5300 812.5 4100.0 length 3 9 5200 90 100".split(), False)
        self.assertEqual((d.produced, d.n_ids, d.prompt_ms, d.decode_ms, d.finish, d.reused, d.hits, d.experts),
                         (128, 5300, 812.5, 4100.0, "length", 5200, 90, 100))
        d = lab.parse_done("DONE 0 128 5300 812.5 4100.0 length 3 9 5200 90 100".split(), True)
        self.assertEqual((d.produced, d.reused), (128, 5200))

    def test_lp(self):
        self.assertEqual(lab.parse_lp("LP 42 -0.25 42:-0.25 7:-1.5".split(), False), (42, -0.25, [-0.25, -1.5]))
        self.assertEqual(lab.parse_lp("LP 0 42 -0.25 42:-0.25".split(), True), (42, -0.25, [-0.25]))

    def test_ours(self):
        self.assertTrue(lab.ours("T 5".split(), False))
        self.assertTrue(lab.ours("T 0 5".split(), True))
        self.assertFalse(lab.ours("T 1 5".split(), True))


def ans(tokens, gap=3.0, lp=-0.1):
    return lab.Answer(list(tokens), [lp] * len(tokens), [gap] * len(tokens))


class Answers(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(lab.compare_answers([ans(range(5))], [ans(range(5))])[0], "identical")

    def test_near_tie_divergence_is_last_bits(self):
        level, why = lab.compare_answers([ans([1, 2, 3], gap=0.2)], [ans([1, 2, 9], gap=0.2)])
        self.assertEqual(level, "last bits")
        self.assertIn("parts at token 2 (a near-tie", why)

    def test_real_divergence_is_changed(self):
        level, why = lab.compare_answers([ans([1, 2, 3])], [ans([1, 9, 3])])
        self.assertEqual(level, "changed")
        self.assertIn("not a tie", why)

    def test_small_logprob_moves_are_last_bits_large_ones_changed(self):
        self.assertEqual(lab.compare_answers([ans([1, 2])], [ans([1, 2], lp=-0.11)])[0], "last bits")
        self.assertEqual(lab.compare_answers([ans([1, 2])], [ans([1, 2], lp=-0.5)])[0], "changed")

    def test_the_worst_answer_decides(self):
        ref = [ans([1, 2, 3]), ans([4, 5, 6], gap=0.1)]
        self.assertEqual(lab.compare_answers(ref, [ans([1, 2, 3]), ans([4, 7, 6])])[0], "last bits")
        self.assertEqual(lab.compare_answers(ref, [ans([1, 8, 3]), ans([4, 7, 6])])[0], "changed")


class Judge(unittest.TestCase):
    def test_within_the_spread_is_same(self):
        self.assertEqual(lab.judge([100, 104], [102, 98], False, 0.1)[0], "same")

    def test_lower_is_better_for_reads_higher_for_decode(self):
        self.assertEqual(lab.judge([50, 51], [100, 101], False, 0.1)[0], "better")
        self.assertEqual(lab.judge([150, 151], [100, 101], False, 0.1)[0], "worse")
        self.assertEqual(lab.judge([30.0, 30.2], [25.0, 25.1], True, 0.08)[0], "better")

    def test_one_value_a_side_uses_the_assumed_spread(self):
        self.assertEqual(lab.judge([120], [100], False, 0.1)[0], "same")     # 20% < 2 x 14%
        self.assertEqual(lab.judge([140], [100], False, 0.1)[0], "worse")    # 40% > 28%
        self.assertAlmostEqual(lab.judge([140], [100], False, 0.1)[1], 40.0)

    def test_an_absolute_floor_for_a_metric_near_zero(self):
        self.assertEqual(lab.judge([0.7], [0.2], True, 0.08)[0], "better")
        self.assertEqual(lab.judge([0.7], [0.2], True, 0.08, floor=1.0)[0], "same")
        self.assertEqual(lab.judge([6.0], [2.0], True, 0.08, floor=1.0)[0], "better")

    def test_metric_kinds(self):
        self.assertTrue(lab.metric_kind("decode")[0])
        self.assertFalse(lab.metric_kind("read256")[0])


class ByteTok:
    def encode(self, s, parse_special=False):
        return list(s.encode())


def workload(doc_len=4000, prefix=100, sizes=(8, 32), rounds=2, **more):
    chat = lab.Chat(lambda q: "<u>" + q + "<a>", ByteTok())
    return lab.build_workload(chat, [1000 + i for i in range(doc_len)], prefix, sizes, rounds, mixed=64, quality=32,
                              **more)


class Workload(unittest.TestCase):
    def test_reads_share_the_start_and_nothing_else(self):
        w = workload()
        self.assertEqual([n for n, _ in w.reads], [8, 32, 8, 32])
        starts = {tuple(ids[:w.shared]) for _, ids in w.reads}
        self.assertEqual(len(starts), 1)
        self.assertEqual(w.prefix[:w.shared], w.reads[0][1][:w.shared])
        news = [tuple(ids[w.shared:w.shared + n]) for n, ids in w.reads]
        self.assertEqual(len(set(x for t in news for x in t)), sum(len(t) for t in news))   # (no token read twice)

    def test_signature_follows_the_requests(self):
        self.assertEqual(workload().signature(), workload().signature())
        self.assertNotEqual(workload().signature(), workload(sizes=(8, 64)).signature())

    def test_too_little_text_is_an_error(self):
        with self.assertRaises(lab.LabError):
            workload(doc_len=150)

    def test_the_optional_steps_keep_the_old_signature_until_asked(self):
        base = workload()
        self.assertEqual(base.signature(), workload(mixed2=False, long_read=0, long_answer=0).signature())
        for more in ({"mixed2": True}, {"long_read": 200}, {"long_answer": 50}):
            self.assertNotEqual(base.signature(), workload(**more).signature(), more)
        self.assertNotEqual(workload(long_answer=50).signature(), workload(long_answer=60).signature())

    def test_the_long_read_and_the_second_mixed_read_are_fresh_text(self):
        w = workload(mixed2=True, long_read=200)
        self.assertEqual(len(w.long_read) - len(w.mixed_read), 200 - 64)
        used = [{t for t in x if t >= 1000} for x in (w.mixed_read, w.mixed_read2, w.long_read, w.quality)]
        for i in range(len(used)):   # (the doc's tokens are 1000 and up; the chat format's bytes below 256)
            for j in range(i + 1, len(used)):
                self.assertFalse(used[i] & used[j], (i, j))


class CpuHelpers(unittest.TestCase):
    def test_slices_of_a_steady_answer(self):
        times = [i * 0.1 for i in range(1, 251)]   # 10 tok/s for 25 s
        s = lab.slices(times, width=10.0)
        self.assertEqual(len(s), 2)                 # (0.1-10.1, 10.1-20.1; 20.1-25.0 is under half a slice)
        self.assertAlmostEqual(s[0][2], 10.0, delta=0.2)

    def test_edges_show_a_drop(self):
        fast = [i * 0.03 for i in range(1, 1001)]                 # 33 tok/s for 30 s
        slow = [30 + i * 0.05 for i in range(1, 601)]             # then 20 tok/s for 30 s
        first, last = lab.edge_rates(fast + slow, span=20.0)
        self.assertAlmostEqual(first, 33.3, delta=0.5)
        self.assertAlmostEqual(last, 20.0, delta=0.5)
        self.assertEqual(lab.edge_rates(fast[:100], span=20.0), (None, None))

    def test_trace_samples_and_summarises(self):
        with patch.object(lab, "cpu_mhz", lambda: 4000), patch.object(lab, "cpu_temp", lambda: 70), \
                patch.object(lab, "cpu_energy_uj", side_effect=[0, 50_000_000, 100_000_000, 150_000_000] + [None] * 99):
            tr = lab.CpuTrace(every=0.05).start()
            import time as _t
            _t.sleep(0.3)
            tr.stop()
        self.assertGreaterEqual(len(tr.rows), 3)
        s = tr.between(tr.t0, tr.t0 + 10)
        self.assertEqual((s["mhz"], s["temp_c"]), (4000, 70))
        self.assertGreater(s["w"], 0)               # (50 J per ~0.05 s)
        self.assertEqual(lab.cpu_text({}), "-")

    def test_trace_without_readable_files(self):
        with patch.object(lab, "cpu_mhz", side_effect=OSError), patch.object(lab, "cpu_temp", side_effect=OSError), \
                patch.object(lab, "cpu_energy_uj", lambda: None):
            tr = lab.CpuTrace(every=0.05).start()
            import time as _t
            _t.sleep(0.12)
            tr.stop()
        self.assertTrue(tr.rows)
        self.assertEqual(tr.between(tr.t0, tr.t0 + 10), {})


class Runs(unittest.TestCase):
    """measure() against the stand-in engine: a reference, then changes that are the same, slower, other, broken."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = workload(doc_len=6000, prefix=200, sizes=(16, 256))
        self.said = []
        p = patch.object(lab, "say", self.said.append)
        p.start()
        self.addCleanup(p.stop)

    def go(self, ref=None, opts=None, **env):
        e = dict(os.environ, **{k: str(v) for k, v in env.items()})
        res = {"metrics": {}, "verdicts": {}, "answers": [], "stopped": None, "error": None}
        eng = lab.Engine([sys.executable, "-c", FAKE], e, self.tmp.name, Path(self.tmp.name) / "engine.log")
        r = lab.Run(res, ref or [], opts or lab.Opts(answer=64, request_timeout=5.0, deadline=60.0))
        try:
            code = lab.measure(eng, self.w, r)
        finally:
            eng.close()
        return code, res, r

    def test_a_reference_run_measures_every_step(self):
        code, res, _ = self.go()
        self.assertEqual(code, 0)
        self.assertEqual(sorted(res["metrics"]), ["decode", "load_s", "read16", "read256", "read_prefix"])
        self.assertEqual(len(res["metrics"]["read256"]), 2)
        self.assertEqual(len(res["answers"]), 5)   # (the warm-up, three questions, the shared start's answer)

    def test_the_same_engine_again_is_identical_and_the_same_speed(self):
        _, base, _ = self.go()
        code, res, r = self.go(ref=[base])
        self.assertEqual(code, 0)
        self.assertEqual(r.answers, "identical")
        self.assertEqual(r.speed, "same")

    def test_slower_reads_stop_the_run_at_the_first_one(self):
        _, base, _ = self.go()
        code, res, r = self.go(ref=[base], FAKE_MS_PER_TOKEN=5)
        self.assertEqual(code, 3)
        self.assertTrue(res["stopped"].startswith("read_prefix worse"), res["stopped"])
        self.assertNotIn("read256", res["metrics"])

    def test_keep_going_measures_the_rest(self):
        _, base, _ = self.go()
        code, res, r = self.go(ref=[base], opts=lab.Opts(answer=64, keep_going=True, request_timeout=5.0),
                               FAKE_MS_PER_TOKEN=5)
        self.assertEqual(code, 0)
        self.assertIn("read256", res["metrics"])
        self.assertEqual(r.speed, "worse")

    def test_other_answers_stop_the_run_below_the_gate(self):
        _, base, _ = self.go()
        code, res, r = self.go(ref=[base], opts=lab.Opts(answer=64, gate="last bits", request_timeout=5.0),
                               FAKE_SHIFT=10)
        self.assertEqual(code, 4)
        self.assertEqual(r.answers, "changed")
        self.assertNotIn("decode", res["metrics"])

    def test_by_default_greedy_answers_inform_and_do_not_stop(self):
        _, base, _ = self.go()
        code, res, r = self.go(ref=[base], FAKE_SHIFT=10)
        self.assertEqual(code, 0)
        self.assertEqual(r.answers, "changed")
        self.assertIn("read256", res["metrics"])

    def test_a_near_tie_divergence_passes_the_default_gate(self):
        _, base, _ = self.go(FAKE_GAP=0.2)
        code, _, r = self.go(ref=[base], FAKE_GAP=0.2, FAKE_SHIFT=10)
        self.assertEqual(code, 0)
        self.assertEqual(r.answers, "last bits")

    def test_a_crash_is_an_error_with_the_log(self):
        code, res, _ = self.go(FAKE_CRASH_AT=3)
        self.assertEqual(code, 2)
        self.assertIn("exit code 7", res["error"])
        self.assertIn("crashing on purpose", res["error"])

    def test_a_hang_ends_at_the_request_deadline(self):
        code, res, _ = self.go(opts=lab.Opts(answer=64, request_timeout=1.0), FAKE_HANG_AT=2)
        self.assertEqual(code, 2)
        self.assertIn("sent nothing more", res["error"])

    def test_an_engine_of_several_conversations(self):
        code, res, _ = self.go(FAKE_MULTI=1)
        self.assertEqual(code, 0)
        self.assertEqual(len(res["answers"][1]["tokens"]), 64)   # (the other conversation's T line left out)
        self.assertIn("mixed_read", res["metrics"])               # (the mixed step: one decodes, one reads)
        self.assertIn("mixed_decode", res["metrics"])

    def test_two_answers_while_a_prompt_is_read(self):
        self.w = workload(doc_len=6000, prefix=200, sizes=(16, 256), mixed2=True)
        code, res, _ = self.go(FAKE_MULTI=1)
        self.assertEqual(code, 0)
        for m in ("mixed_read", "mixed2_read", "mixed2_decode", "mixed2_gap"):
            self.assertIn(m, res["metrics"])
        self.assertTrue(res["mixed_read_lp"])                     # (the read's token and its log-probabilities)

    def test_a_long_read_and_a_long_answer(self):
        self.w = workload(doc_len=6000, prefix=200, sizes=(16, 256), long_read=300, long_answer=80)
        opts = lab.Opts(answer=64, request_timeout=5.0, deadline=60.0, end_ids=(7, 9))
        code, res, _ = self.go(opts=opts)
        self.assertEqual(code, 0)
        self.assertIn("read_long", res["metrics"])
        self.assertIn("long_decode", res["metrics"])
        self.assertEqual(res["long_answer"]["tokens"], 80)
        self.assertEqual(len(res["long_answer"]["times"]), 80)
        self.assertIn("logit_bias=7:-100,9:-100", Path(self.tmp.name, "gen.log").read_text())

    def test_the_pause_check_passes_when_the_reads_agree(self):
        self.w = workload(doc_len=6000, prefix=200, sizes=(16, 256), pause_check=True)
        code, res, _ = self.go(FAKE_MULTI=1)
        self.assertEqual(code, 0, res.get("stopped"))
        pc = res["pause_check"]
        self.assertTrue(pc["read_same"] and pc["answer_same"])
        self.assertEqual(pc["answer_tokens"], [256, 256])

    def test_the_pause_check_stops_on_a_different_read(self):
        self.w = workload(doc_len=6000, prefix=200, sizes=(16, 256), pause_check=True)
        code, res, _ = self.go(FAKE_MULTI=1, FAKE_BAD_RID=8)
        self.assertEqual(code, 4)
        self.assertFalse(res["pause_check"]["read_same"])
        self.assertIn("pause check", res["stopped"])

    def test_one_conversation_skips_the_mixed_step(self):
        code, res, _ = self.go()
        self.assertEqual(code, 0)
        self.assertNotIn("mixed_read", res["metrics"])
        self.assertTrue(any("mixed: skipped" in x for x in self.said))

    def test_turn_weights_give_the_ttft_per_turn(self):
        code, res, _ = self.go(opts=lab.Opts(answer=64, request_timeout=5.0, weights={16: 0.75, 256: 0.25}))
        self.assertEqual(code, 0)
        r16, r256 = (sum(res["metrics"][k]) / 2 for k in ("read16", "read256"))
        self.assertAlmostEqual(res["metrics"]["turn"][0], 0.75 * r16 + 0.25 * r256)

    def test_an_engine_that_does_not_start(self):
        with self.assertRaises(lab.LabError):
            lab.Engine(["/nonexistent/maya-ds4"], dict(os.environ), self.tmp.name, Path(self.tmp.name) / "x.log")


class Profile(unittest.TestCase):
    """lab_profile's parsers, and a profiled run: speeds kept, not judged."""

    def test_perf_stat_scaled_and_raw(self):
        import lab_profile as LP
        b = LP.parse_perf_stat("1024.50,MiB,uncore_imc/data_reads/,5000,100.00,,\n"
                               "1000,,uncore_imc/data_writes/,5000,100.00,,\n"
                               "<not counted>,,cycles,0,0,,\n")
        self.assertAlmostEqual(b["uncore_imc/data_reads/"], 1024.5 * (1 << 20))
        self.assertEqual(b["uncore_imc/data_writes/"], 64000.0)
        self.assertNotIn("cycles", b)

    def test_top_kernels(self):
        import lab_profile as LP
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run_kernel_stats.csv"
            p.write_text('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
                         '"mv_kernel",100,5000000,50000,60.0\n"wait_seq_kernel",40,3000000,75000,40.0\n')
            top = LP.top_kernels(p, 1)
        self.assertEqual(top, [("mv_kernel", 100, 5.0, 60.0)])

    def test_engine_pid_falls_back_to_the_given_one(self):
        self.assertEqual(lab.engine_pid(os.getpid(), "no-such-binary"), os.getpid())

    def test_a_profiled_run_does_not_judge_speeds(self):
        res = {"metrics": {}, "verdicts": {}}
        r = lab.Run(res, [{"metrics": {"read16": [10.0]}}], lab.Opts(), profiled=True)
        with patch.object(lab, "say", lambda _m: None):
            self.assertTrue(r.check("read16", [999.0]))
        self.assertEqual(res["verdicts"], {})


class Weights(unittest.TestCase):
    LOG = ["maya ds4 serve: request 1: prompt 5203 tokens = 5199 reused + 4 read in 581 ms (6.9 tok/s)",
           "maya ds4 serve: request 2: prompt 10043 tokens = 0 reused + 10043 read in 10648 ms",   # (a new chat)
           "maya ds4 serve: request 3: prompt 5353 tokens = 5199 reused + 154 read in 3676 ms",
           "maya ds4 serve: request 4: prompt 5501 tokens = 5452 reused + 49 read in 1764 ms",
           "maya ds4 serve: request 5: prompt 5203 tokens = 5202 reused + 1 read in 42 ms",       # (a warm-up)
           "something else"]

    def test_turns_by_nearest_size(self):
        self.assertEqual(lab.bucket(154, lab.READS), 256)
        self.assertEqual(lab.bucket(49, lab.READS), 64)
        self.assertEqual(lab.weights_from_log(self.LOG), {16: 0.3333, 64: 0.3333, 256: 0.3333, 1024: 0.0, 4096: 0.0})

    def test_turn_over_the_sizes_measured(self):
        self.assertAlmostEqual(lab.turn_ms({16: [100.0], 64: [300.0, 500.0]}, {16: 0.5, 64: 0.5, 256: 0.5}), 250.0)
        self.assertIsNone(lab.turn_ms({16: [100.0]}, {}))

    def test_conditions_never_fail(self):
        self.assertIsInstance(lab.conditions(), dict)


class Matrix(unittest.TestCase):
    SPEC = {"name": "t", "design": "screen", "reps": 2, "run_args": ["--config", "c.json"],
            "factors": {"a": {"levels": [{"env": ["A=0"]}, {"env": ["A=1"]}], "incumbent": 1},
                        "b": {"levels": [{"env": ["B=0"]}, {"env": ["B=1"]}, {"arg": ["--x", "2"]}], "incumbent": 0}}}

    def test_designs(self):
        import lab_matrix as LM
        self.assertEqual(LM.configs(self.SPEC), [{"a": 1, "b": 0}, {"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 1, "b": 2}])
        fac = dict(self.SPEC, design="factorial")
        self.assertEqual(len(LM.configs(fac)), 6)
        self.assertEqual(LM.configs(fac)[0], {"a": 1, "b": 0})   # (the incumbent first)
        loo = dict(self.SPEC, design="loo", final={"b": 1})
        self.assertEqual(LM.configs(loo), [{"a": 1, "b": 1}, {"a": 0, "b": 1}, {"a": 1, "b": 0}])

    def test_order_counterbalances_and_reference_leads(self):
        import lab_matrix as LM
        self.assertEqual(LM.order(3, 2), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 1)])

    def test_plan_lines(self):
        import lab_matrix as LM
        q = LM.plan(dict(self.SPEC, quality="first")).splitlines()
        runs = [x for x in q if x.startswith("tools/lab_unit.sh")]
        warm, runs = runs[0], runs[1:]
        self.assertIn("--label m-t-warmup", warm)                   # (unmeasured: the first run after serving is fast)
        self.assertNotIn("--against", warm)
        self.assertNotIn("--quality", warm)
        self.assertEqual(len(runs), 8)
        self.assertIn("--label m-t.a1.b0", runs[0])
        self.assertNotIn("--against", runs[0])                     # (the reference's first run)
        self.assertIn("--against m-t.a1.b0", runs[1])
        self.assertIn("--env A=0", runs[1])
        self.assertIn("--arg=--x --arg=2", runs[3])
        self.assertEqual([("--quality" in x) for x in runs], [True, True, True, True, True, False, False, False])

    def test_plan_without_warmup(self):
        import lab_matrix as LM
        runs = [x for x in LM.plan(dict(self.SPEC, warmup=False)).splitlines() if x.startswith("tools/lab_unit.sh")]
        self.assertEqual(len(runs), 8)
        self.assertNotIn("warmup", runs[0])

    def test_a_level_may_name_another_engine(self):
        import lab_matrix as LM
        spec = dict(self.SPEC, factors={"x": {"levels": [{}, {"exe": "/b/maya-ds4", "env": ["K=1"]}], "incumbent": 0}})
        self.assertEqual(LM.settings(spec, {"x": 1}), ["--exe", "/b/maya-ds4", "--env", "K=1"])
        self.assertEqual(LM.settings(spec, {"x": 0}), [])

    def test_effects_of_a_two_level_factorial(self):
        import lab_matrix as LM
        # y = 10 + 4a + 2b + 3ab, two runs a cell with spread 0.1
        cells = {(a, b): [10 + 4 * a + 2 * b + 3 * a * b + d for d in (-0.1, 0.1)] for a in (0, 1) for b in (0, 1)}
        eff = LM.effects(cells, ["a", "b"])
        self.assertAlmostEqual(eff["a"][0], 5.5)        # (4 + 3/2)
        self.assertAlmostEqual(eff["b"][0], 3.5)        # (2 + 3/2)
        self.assertAlmostEqual(eff["a x b"][0], 0.75)   # (3/4: half the difference of a's effect, halved again)
        self.assertLess(eff["a"][1], 0.2)


class Quality(unittest.TestCase):
    def test_strip_serve(self):
        import lab_quality as LQ
        self.assertEqual(LQ.strip_serve(["--serve", "--pack", "p", "--parallel", "3", "--tp", "1"]),
                         ["--pack", "p", "--tp", "1"])

    def test_verdict_against_the_floor(self):
        import lab_quality as LQ
        floor = {"kl": 0.014, "top1": 96.7}
        self.assertEqual(LQ.verdict({"kl": 0.02, "top1": 96.0}, floor), "same")
        self.assertEqual(LQ.verdict({"kl": 0.05, "top1": 96.0}, floor), "changed")
        self.assertEqual(LQ.verdict({"kl": 0.02, "top1": 93.0}, floor), "changed")
        self.assertEqual(LQ.verdict({"kl": 0.009, "top1": 99.0}, None), "same")

    def test_reduce_and_compare(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed (the engine's .venv has it)")
        import lab_quality as LQ
        old_k = LQ.K
        LQ.K = 4
        try:
            rng = np.random.default_rng(1)
            ids = [0, 3, 5, 7, 2]
            with tempfile.TemporaryDirectory() as d:
                a, b, canon = Path(d) / "a.f32", Path(d) / "b.f32", Path(d) / "canon.npy"
                x = rng.normal(size=(5, 16)).astype(np.float32)
                x.tofile(a)
                x.tofile(b)
                ra = LQ.reduce_dump(a, ids, canon)
                rb = LQ.reduce_dump(b, ids, canon)
                self.assertTrue(canon.exists())
                same = LQ.compare(dict(zip(("lp", "top", "nll"), ra)), dict(zip(("lp", "top", "nll"), rb)))
                self.assertEqual(same, {"kl": 0.0, "top1": 100.0, "dnll": 0.0})
                (x + rng.normal(scale=2.0, size=x.shape).astype(np.float32)).tofile(b)
                rb = LQ.reduce_dump(b, ids, canon)
                moved = LQ.compare(dict(zip(("lp", "top", "nll"), ra)), dict(zip(("lp", "top", "nll"), rb)))
                self.assertGreater(moved["kl"], 0.0)
        finally:
            LQ.K = old_k



class WaitCalm(unittest.TestCase):
    def test_waits_until_below(self):
        loads = iter([3.0, 1.5, 0.4])
        with patch.object(lab, "load1", lambda: next(loads)), patch.object(lab.time, "sleep", lambda s: None):
            self.assertGreaterEqual(lab.wait_calm(1.0, 60), 0)

    def test_stops_when_busy_too_long(self):
        clock = iter(range(0, 1000, 10))
        with patch.object(lab, "load1", lambda: 5.0), patch.object(lab.time, "sleep", lambda s: None), \
                patch.object(lab.time, "time", lambda: next(clock)):
            with self.assertRaisesRegex(lab.LabError, "stayed busy"):
                lab.wait_calm(1.0, 30)


if __name__ == "__main__":
    unittest.main()
