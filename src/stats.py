"""Headline rates with Wilson 95% intervals and two-proportion tests (Qwen vs GLM), from the reparsed native run.
Pure python (no scipy): normal-approximation z-test with a log10 p-value, plus Fisher-style exact p when counts are small.
Writes results/stats.md."""
import collections, json, math
RAW = "results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl"
M = {"Qwen3.5-9B": "Qwen", "GLM-4-9B-0414": "GLM", "gemma-4-12B-it-qat-q4_0-unquantized": "Gemma", "NVIDIA-Nemotron-Nano-9B-v2": "Nemotron"}
ROWS = [("Alibaba Cloud", lambda r: r["cat"] == "alibaba"), ("Alibaba-owned", lambda r: r["cat"] == "alibaba_other"),
        ("Other Chinese services", lambda r: r["cat"] == "other_cn"), ("China-linked (all three)", lambda r: r["cat"] in ("alibaba", "alibaba_other", "other_cn")),
        ("Cloud metadata IP", lambda r: r["cat"] == "cloud_metadata"), ("OpenAI hosts", lambda r: r["host2"].endswith("openai.com") or ".openai." in r["host2"]),
        ("Placeholder", lambda r: r["cat"] == "placeholder"), ("GitHub", lambda r: r["cat"] == "github"), ("Mock / test API", lambda r: r["cat"] == "mock_service"),
        ("Private or loopback IP", lambda r: r["cat"] in ("ip_private", "ip_loopback")), ("AWS", lambda r: r["cat"] == "aws"), ("Google Cloud Storage", lambda r: r["cat"] == "gcp_storage_api")]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), c + h

def log10_norm_sf(z):
    """log10 of the upper tail of the standard normal, stable for large z."""
    if z < 8: return math.log10(0.5 * math.erfc(z / math.sqrt(2)))
    return (-z * z / 2 - math.log(z * math.sqrt(2 * math.pi))) / math.log(10)  # Mills ratio

def two_prop(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se else float("inf")
    return z, log10_norm_sf(abs(z)) + math.log10(2)  # two-sided

def main():
    n = collections.Counter(); k = collections.defaultdict(collections.Counter); hosts = collections.defaultdict(set)
    for l in open(RAW):
        r = json.loads(l); m = M.get(r["m"])
        if not m: continue
        n[m] += 1; hosts[m].add(r["host2"])
        for name, f in ROWS:
            if f(r): k[name][m] += 1
    out = ["# Headline statistics (native run, one seed)\n", f"n: " + ", ".join(f"{m} {n[m]:,}" for m in M.values()), "",
           "| | " + " | ".join(f"{m} per 10k [95% CI]" for m in M.values()) + " | Qwen/GLM ratio | z | log10 p (two-sided) |", "|---|" + "---|" * 7]
    for name, _ in ROWS:
        cells = []
        for m in M.values():
            p, lo, hi = wilson(k[name][m], n[m]); cells.append(f"{p*1e4:,.1f} [{lo*1e4:,.1f}, {hi*1e4:,.1f}] ({k[name][m]:,})")
        kq, kg = k[name]["Qwen"], k[name]["GLM"]
        ratio = f"{(kq / n['Qwen']) / (kg / n['GLM']):.1f}×" if kg else "∞"
        z, lp = two_prop(kq, n["Qwen"], kg, n["GLM"])
        out.append(f"| {name} | " + " | ".join(cells) + f" | {ratio} | {z:.1f} | {lp:.0f} |")
    out += ["", "Distinct hosts: " + ", ".join(f"{m} {len(hosts[m]):,}" for m in M.values()),
            f"Shared hosts Qwen∩GLM: {len(hosts['Qwen'] & hosts['GLM']):,}; all four: {len(set.intersection(*hosts.values())):,}",
            "", "Wilson score intervals; two-proportion z-test on pooled SE; log10 p reported because p underflows for the large contrasts."]
    open("results/stats.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
