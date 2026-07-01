"""A: LIVE LLM oracle on real text. A real Qwen2.5-0.5B-Instruct labels each document
(is it about medicine? yes/no) via next-token yes/no logits. Truth = dataset label.
This instantiates the biased-oracle ablation with REAL LLM errors (not a simulated keep
probability): we measure whether the LLM oracle's errors correlate with the MiniLM proxy
(dropping hard positives), and whether the naive recall CS then lies while a clean anchor holds.
Runs on CPU (GPU occupied). Caches labels to results/.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

TOL = 1e-9
DELTA, TARGET = 0.10, 0.70


def get_data_and_proxy(target="sci.med", n_neg=600):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer
    cats = ['rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns']
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)
    yt = (np.array(tr.target) == cats.index(target)).astype(int)
    ye = (np.array(te.target) == cats.index(target)).astype(int)
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    Etr = enc.encode(tr.data, batch_size=64, normalize_embeddings=True)
    Ete = enc.encode(te.data, batch_size=64, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=3000, C=2.0).fit(Etr, yt)
    s = clf.predict_proba(Ete)[:, 1]
    docs = np.array(te.data, dtype=object)
    pos_idx = np.where(ye == 1)[0]
    neg_idx = np.where(ye == 0)[0]
    rng = np.random.default_rng(0)
    neg_sel = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    sel = np.concatenate([pos_idx, neg_sel])
    return docs[sel], ye[sel], s[sel]


def llm_label(docs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to("cpu").eval()
    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]
    out = np.zeros(len(docs))
    for i, d in enumerate(docs):
        text = str(d)[:800]
        msg = [{"role": "user", "content":
                "Does the following text discuss medicine, health, disease, or medical "
                "treatment? Answer only yes or no.\n\nText: " + text + "\n\nAnswer:"}]
        p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt", truncation=True, max_length=384)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        py = float(logits[yes_id]); pn = float(logits[no_id])
        out[i] = 1.0 if py >= pn else 0.0
        if (i + 1) % 200 == 0:
            print(f"  labeled {i+1}/{len(docs)}", flush=True)
    return out


def main():
    docs, y, s = get_data_and_proxy()
    print(f"data: {len(docs)} docs ({int(y.sum())} positives). Running LLM oracle on CPU...", flush=True)
    cache = os.path.join(os.path.dirname(__file__), "..", "results", "llm_oracle_labels.npy")
    if os.path.exists(cache):
        o = np.load(cache)
        print("loaded cached LLM labels")
    else:
        o = llm_label(docs)
        np.save(cache, o)
    # LLM oracle quality vs truth
    acc = float((o == y).mean())
    llm_recall = float(o[y == 1].mean())          # fraction of true positives the LLM keeps
    # does the LLM drop LOW-proxy positives? correlation of keep with proxy score
    pos = y == 1
    corr = float(np.corrcoef(o[pos], s[pos])[0, 1])
    print(f"\nLLM oracle: acc={acc:.3f}, recall(keeps true positives)={llm_recall:.3f}, "
          f"corr(LLM-keep, proxy score | positive)={corr:.3f}")

    # coverage experiment (no drift): naive uses LLM-kept positives; anchor uses true labels.
    tau = float(np.quantile(s[pos], 0.20))
    true_recall = float((s[pos] >= tau).mean())
    sp = s[pos]; op = o[pos]
    kept = sp[op == 1]                              # proxy scores of LLM-kept true positives
    print(f"tau={tau:.3f}, true recall={true_recall:.3f}, "
          f"naive (LLM-kept) recall={float((kept>=tau).mean()) if len(kept) else 0:.3f}")

    def coverage(method, n_seeds=200, T=400, qe=25, anchor_rate=0.15):
        br = 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            cs = RecallCS(delta=DELTA, grid=801); bad = False
            for t in range(T):
                j = rng.integers(len(sp))            # sample a true positive
                if method == "naive":
                    if op[j] == 1:                   # LLM oracle keeps it as positive
                        cs.update(1.0 if sp[j] >= tau else 0.0)
                else:                                # anchor: clean labels, small budget
                    if rng.random() < anchor_rate:
                        cs.update(1.0 if sp[j] >= tau else 0.0)
                if t % qe and t != T - 1:
                    continue
                if cs.lower() > true_recall + TOL:
                    bad = True
            br += int(bad)
        return 1 - br / n_seeds

    cn = coverage("naive"); ca = coverage("anchor")
    print(f"\n[LIVE-LLM oracle] coverage vs dataset truth (delta={DELTA}, no drift):")
    print(f"   naive (LLM-oracle as truth): {cn:.3f}  {'OK' if cn>=0.86 else 'LIES'}")
    print(f"   anchor-corrected:            {ca:.3f}  {'OK' if ca>=0.86 else 'LIES'}")
    out = {"llm": "Qwen2.5-0.5B-Instruct", "acc": acc, "llm_recall": llm_recall,
           "corr_keep_proxy": corr, "tau": tau, "true_recall": true_recall,
           "naive_recall_est": float((kept >= tau).mean()) if len(kept) else 0.0,
           "coverage_naive": cn, "coverage_anchor": ca, "delta": DELTA}
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "llm_oracle.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> results/llm_oracle.json")


if __name__ == "__main__":
    main()
