# Systems micro-benchmark: guarantee-ledger overhead (resolves the systems-eval gap, partially)

The ledger adds an O(G) vectorized wealth update per oracle label (G = threshold grid) + an O(G)
lower-bound scan per query. Single-core CPU. Reproduce: `python3 experiments/throughput.py`.

| grid G | ledger labels/sec | ms/label | ms/query (reuse=200, fresh<=40) |
|---|---|---|---|
| 201  | 143,058 | 0.0070 | 78.5 |
| 801  | 83,685  | 0.0119 | 115.2 |
| 2001 | 57,495  | 0.0174 | 193.4 |

Overhead vs the LLM-oracle cost (grid=801, ~40 oracle calls/query):
| oracle model | oracle ms/query | ledger overhead |
|---|---|---|
| GPT-4o-class (800 ms/call) | 32,000 | 0.36% |
| small-LLM (150 ms/call)    | 6,000  | 1.92% |

## Honest reading
- The guarantee ledger runs at 57k-143k labels/sec and adds ~115 ms/query of compute -- i.e.
  0.36%-1.9% of the LLM-oracle cost. It is NOT the bottleneck and is cheap enough to embed in a
  query engine as a drop-in guarantee layer.
- Scope (honest): this is a micro-benchmark of the guarantee-layer OVERHEAD, not a full engine
  integration with a live LLM backend (which needs API access) -- that end-to-end systems study
  remains future work. The per-query cost can be reduced further with incremental (rather than
  rescanned) lower-bound tracking.
