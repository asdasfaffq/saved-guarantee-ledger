"""Systems micro-benchmark: guarantee-ledger overhead and throughput.

The guarantee ledger adds an O(G) vectorized wealth update per oracle label (G = threshold
grid) plus an O(G) scan for the lower bound per query. This measures that overhead in
isolation and against a nominal LLM-oracle latency, showing the guarantee layer is cheap
enough to embed in a query engine. (No LLM API needed; full engine integration with a live
LLM backend is future work.)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PerSampleSAVED

ORACLE_MS = {"GPT-4o-class": 800.0, "small-LLM": 150.0}  # nominal per-oracle-call latencies


def bench_update(grid, n=20000):
    cs = RecallCS(delta=0.1, grid=grid)
    xs = (np.random.random(n) < 0.8).astype(float)
    t0 = time.perf_counter()
    for x in xs:
        cs.update(float(x))
    dt = time.perf_counter() - t0
    return n / dt, dt / n * 1e3  # labels/sec, ms/label


def bench_query(grid, reuse=200, fresh=40, reps=300):
    """Time one full per-sample certification (feed reuse + fresh draws + lower scans)."""
    rng = np.random.default_rng(0)
    buf = [(float(rng.random() < 0.8), rng.random()) for _ in range(reuse)]
    t0 = time.perf_counter()
    for _ in range(reps):
        sv = PerSampleSAVED(delta=0.1, eps_bound=0.2, grid=grid)
        for (x, tn) in buf:
            sv.update(x, tn)
        # simulate the certify loop: draw fresh, re-check lower each step
        for j in range(fresh):
            sv.update(float(rng.random() < 0.8), 1.0)
            sv.lower(1.0)
    dt = (time.perf_counter() - t0) / reps
    return dt * 1e3  # ms/query


def main():
    print("\nGuarantee-ledger systems micro-benchmark (single core, CPU)\n")
    print("Ledger update throughput (per oracle label):")
    print(f"{'grid G':>8} | {'labels/sec':>12} | {'ms/label':>9}")
    print("-" * 34)
    for g in (201, 801, 2001):
        lps, mspl = bench_update(g)
        print(f"{g:>8} | {lps:>12,.0f} | {mspl:>9.4f}")

    print("\nPer-query certification latency (reuse=200, fresh<=40):")
    print(f"{'grid G':>8} | {'ms/query':>9}")
    print("-" * 22)
    qms = {}
    for g in (201, 801, 2001):
        qms[g] = bench_query(g)
        print(f"{g:>8} | {qms[g]:>9.2f}")

    print("\nGuarantee overhead vs LLM-oracle cost (grid=801, ~40 oracle calls/query):")
    q_over = qms[801]
    n_calls = 40
    print(f"{'oracle':>14} | {'oracle ms/query':>16} | {'ledger overhead':>16}")
    print("-" * 54)
    for name, ms in ORACLE_MS.items():
        oracle_total = ms * n_calls
        frac = q_over / oracle_total * 100
        print(f"{name:>14} | {oracle_total:>13,.0f} ms | {frac:>13.3f}% ")
    print(f"\n=> The guarantee ledger adds ~{q_over:.1f} ms/query of compute, i.e. <<1% of the "
          f"LLM-oracle cost;\n   it is not the bottleneck and is practical to embed in an engine.")


if __name__ == "__main__":
    main()
