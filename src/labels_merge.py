"""Join rule categories with Haiku host labels and path labels into one per-row table, then aggregate.

Inputs : results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl, results/labels/hosts.jsonl, results/labels/paths.jsonl
Outputs: results/labels/merged.jsonl (one line per completion with rule + host + path fields),
         results/labels/summary.md (per-model aggregates), results/labels/disagreements.txt (rules vs Haiku),
         results/labels/oddities.txt (grouped oddity phrases)
Works on partial label files: unlabeled rows carry null fields and are counted as 'unlabeled'."""
import collections, json, os, re, sys

RAW = "results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl"
L = "results/labels/"
MODELS = {"Qwen3.5-9B": "Qwen", "GLM-4-9B-0414": "GLM", "gemma-4-12B-it-qat-q4_0-unquantized": "Gemma", "NVIDIA-Nemotron-Nano-9B-v2": "Nemotron"}
HOST_F = ["host_role", "placeholder", "domain_status", "host_malformed", "operator", "country", "region_hint", "signed_url", "confidence", "note"]
PATH_F = ["action", "path_kind", "task_match", "credential_like", "oddity", "note"]

def load_labels(path, key):
    d = {}
    if not os.path.exists(path): return d
    for l in open(path):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        d[r[key]] = r
    return d

OP_NORM = [  # (regex on lowercased operator string, canonical name); first match wins
    (r"google|alphabet|youtube|firebase", "Google"), (r"github", "GitHub"), (r"amazon|aws", "Amazon (AWS)"),
    (r"microsoft|azure|linkedin", "Microsoft"), (r"hugging ?face", "Hugging Face"), (r"httpbin|kenneth reitz|postman", "httpbin / Postman"),
    (r"typicode|jsonplaceholder", "Typicode (jsonplaceholder)"), (r"transfer\.sh", "transfer.sh"), (r"wikimedia|wikipedia", "Wikimedia"),
    (r"openai|chatgpt", "OpenAI"), (r"alibaba|aliyun|qwen|taobao|alipay", "Alibaba"), (r"tencent|qq\b|weixin|wechat", "Tencent"),
    (r"baidu", "Baidu"), (r"huawei|hisilicon", "Huawei"), (r"bytedance|tiktok|douyin", "ByteDance"), (r"yandex", "Yandex"),
    (r"cloudflare", "Cloudflare"), (r"dropbox", "Dropbox"), (r"pastebin", "Pastebin"), (r"hetzner", "Hetzner"), (r"digitalocean", "DigitalOcean"),
    (r"anthropic|claude", "Anthropic"), (r"infura|consensys", "Infura"), (r"uc irvine|california, irvine|uci\b", "UC Irvine"), (r"ngrok", "ngrok"),
    (r"vercel", "Vercel"), (r"netlify", "Netlify"), (r"heroku|salesforce", "Salesforce (Heroku)"), (r"gitlab", "GitLab"), (r"atlassian|bitbucket", "Atlassian"),
    (r"catbox", "Catbox"), (r"backblaze", "Backblaze"), (r"wasabi", "Wasabi"), (r"ibm|softlayer", "IBM"), (r"oracle", "Oracle"), (r"apache", "Apache Foundation"),
    (r"mozilla", "Mozilla"), (r"telegram", "Telegram"), (r"slack", "Slack"), (r"discord", "Discord"), (r"nasa", "NASA"), (r"\bgov\b|government", "government (other)"),
]
def norm_op(op):
    if not op: return None
    o = op.lower()
    for rx, name in OP_NORM:
        if re.search(rx, o): return name
    return op.strip()

def pct(k, n): return f"{100 * k / n:5.1f}%" if n else "   - "

def main():
    hosts = load_labels(L + "hosts.jsonl", "host"); paths = load_labels(L + "paths.jsonl", "url")
    print(f"host labels {len(hosts):,} · path labels {len(paths):,}")
    n = collections.Counter(); hl = collections.Counter(); pl = collections.Counter()
    role = collections.defaultdict(collections.Counter); dom = collections.defaultdict(collections.Counter)
    cc = collections.defaultdict(collections.Counter); pk = collections.defaultdict(collections.Counter)
    act = collections.defaultdict(collections.Counter); tm = collections.defaultdict(collections.Counter)
    opc = collections.defaultdict(collections.Counter)
    cred = collections.Counter(); malformed = collections.Counter(); odd = collections.defaultdict(collections.Counter)
    dis = collections.defaultdict(list)
    out = open(L + "merged.jsonl", "w")
    for l in open(RAW):
        r = json.loads(l); m = MODELS.get(r["m"])
        if not m: continue
        n[m] += 1
        h = hosts.get(r["host2"]); p = paths.get(r["url"])
        row = {"m": m, "url": r["url"], "host": r["host2"], "cat": r["cat"], "reg": r["reg"], "tranco": r["tranco"], "tenant_ph": r.get("tenant_ph", False)}
        for f in HOST_F: row["h_" + f] = h.get(f) if h else None
        for f in PATH_F: row["p_" + f] = p.get(f) if p else None
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if h:
            hl[m] += 1; role[m][h.get("host_role")] += 1; dom[m][h.get("domain_status")] += 1
            if h.get("country"): cc[m][h["country"]] += 1
            if h.get("operator") and h.get("domain_status") in ("real_known", "not_applicable"): opc[m][norm_op(h["operator"])] += 1
            if h.get("host_malformed"): malformed[m] += 1
            # disagreements: rule placeholder vs Haiku, and Haiku placeholder on a rule 'real' category
            if r["cat"] == "placeholder" and h.get("placeholder") is False and h.get("domain_status") in ("real_known",):
                dis["rule=placeholder, haiku=real_known"].append(r["host2"])
            if r["cat"] in ("known_domain",) and h.get("domain_status") in ("generic_invention", "garbage"):
                dis["rule=known_domain(Tranco), haiku=generic/garbage"].append(r["host2"])
            if r["cat"] == "unknown_domain" and h.get("domain_status") == "real_known":
                dis["rule=unknown_domain, haiku=real_known"].append(f'{r["host2"]} ({h.get("operator")}, {h.get("country")})')
            if r["cat"] in ("alibaba", "alibaba_other", "other_cn") and h.get("country") not in ("CN", "HK", None):
                dis["rule=China-linked, haiku country != CN"].append(f'{r["host2"]} ({h.get("operator")}, {h.get("country")})')
            if r["cat"] not in ("alibaba", "alibaba_other", "other_cn", "cloud_metadata") and h.get("country") in ("CN", "HK"):
                dis["haiku country=CN, rule not China-linked"].append(f'{r["host2"]} ({h.get("operator")}) rule={r["cat"]}')
        if p:
            pl[m] += 1; pk[m][p.get("path_kind")] += 1; act[m][p.get("action")] += 1; tm[m][p.get("task_match")] += 1
            if p.get("credential_like"): cred[m] += 1
            if p.get("oddity"):
                key = re.sub(r"[^a-z ]", "", str(p["oddity"]).lower()).strip()
                odd[m][key] += 1
    out.close()
    ms = [x for x in ("Qwen", "GLM", "Gemma", "Nemotron") if n[x]]
    with open(L + "summary.md", "w") as f:
        f.write("# Label summary (partial files allowed)\n\n")
        f.write("| | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms) + "\n")
        f.write("| completions | " + " | ".join(f"{n[m]:,}" for m in ms) + " |\n")
        f.write("| with host label | " + " | ".join(pct(hl[m], n[m]) for m in ms) + " |\n")
        f.write("| with path label (excl. github/placeholder/tenant) | " + " | ".join(pct(pl[m], n[m]) for m in ms) + " |\n")
        f.write("| host malformed (of host-labeled) | " + " | ".join(pct(malformed[m], hl[m]) for m in ms) + " |\n")
        f.write("| credential-like params (of path-labeled) | " + " | ".join(pct(cred[m], pl[m]) for m in ms) + " |\n\n")
        for title, table, denom in (("Host role", role, hl), ("Domain status", dom, hl), ("Operator country (host-labeled, real_known)", cc, hl), ("Operator (normalized, real_known hosts)", opc, hl),
                                    ("Path kind", pk, pl), ("Action", act, pl), ("Task match", tm, pl)):
            keys = [k for k, _ in sum((table[m] for m in ms), collections.Counter()).most_common(16 if title[:8] != "Operator" else 30)]
            f.write(f"## {title}\n\n| | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms) + "\n")
            for k in keys:
                f.write(f"| {k} | " + " | ".join(f"{pct(table[m][k], denom[m])} ({table[m][k]:,})" for m in ms) + " |\n")
            f.write("\n")
    with open(L + "disagreements.txt", "w") as f:
        for k, v in dis.items():
            c = collections.Counter(v)
            f.write(f"## {k}: {len(v)} rows, {len(c)} hosts\n" + "\n".join(f"  {h}  ×{x}" for h, x in c.most_common(60)) + "\n\n")
    with open(L + "oddities.txt", "w") as f:
        for m in ms:
            f.write(f"## {m}: {sum(odd[m].values())} rows with an oddity\n" + "\n".join(f"  {x:4d}  {k}" for k, x in odd[m].most_common(80)) + "\n\n")
    print(open(L + "summary.md").read()[:6000])
    print("disagreements:", {k: len(v) for k, v in dis.items()})

if __name__ == "__main__":
    main()
