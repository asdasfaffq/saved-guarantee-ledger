"""Fetch a NATURALLY time-ordered arXiv corpus for the drift experiment.

Two categories give a deterministic binary predicate ("is cs.LG"): cs.LG (positive)
vs cs.DB (negative). We fetch abstracts across several years so the corpus drifts
naturally over time (vocabulary/topics of cs.LG shift from classical ML to deep
learning to LLMs), which is exactly the natural temporal drift the paper needs
(not an induced easy/hard mixture). Cached to disk; safe to re-run.
"""
import json, os, re, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_timed.json")
CATS = {"cs.LG": 1, "cs.DB": 0}          # positive predicate = cs.LG
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
PER_YEAR = 220                            # per category per year


def fetch(cat, y, start, n=100, retries=4):
    q = urllib.parse.urlencode({
        "search_query": f"cat:{cat} AND submittedDate:[{y}01010000 TO {y}12312359]",
        "start": start, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "ascending"})
    url = "https://export.arxiv.org/api/query?" + q
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read().decode()
        except Exception as e:
            time.sleep(3 * (a + 1))
    return ""


def parse(xml):
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        ttl = re.search(r"<title>(.*?)</title>", entry, re.S)
        summ = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        pub = re.search(r"<published>(.*?)</published>", entry)
        if ttl and summ and pub:
            text = re.sub(r"\s+", " ", (ttl.group(1) + ". " + summ.group(1))).strip()
            out.append({"text": text, "date": pub.group(1)[:10]})
    return out


def main():
    cache = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {(d["date"], d["text"][:60]) for d in cache}
    for cat, lab in CATS.items():
        for y in YEARS:
            got = sum(1 for d in cache if d.get("cat") == cat and d["date"][:4] == str(y))
            if got >= PER_YEAR:
                continue
            start = 0
            while got < PER_YEAR:
                xml = fetch(cat, y, start, 100)
                rows = parse(xml)
                if not rows:
                    break
                for r in rows:
                    k = (r["date"], r["text"][:60])
                    if k in seen:
                        continue
                    seen.add(k)
                    r["cat"] = cat; r["label"] = lab
                    cache.append(r); got += 1
                    if got >= PER_YEAR:
                        break
                start += 100
                time.sleep(3)
            json.dump(cache, open(OUT, "w"))
            print(f"{cat} {y}: total {got}", flush=True)
    print(f"\nTOTAL {len(cache)} abstracts; pos(cs.LG)={sum(d['label'] for d in cache)}")
    print(f"date range: {min(d['date'] for d in cache)} .. {max(d['date'] for d in cache)}")
    json.dump(cache, open(OUT, "w"))


if __name__ == "__main__":
    main()
