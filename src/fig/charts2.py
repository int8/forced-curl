"""Light, editorial-style figures (Reuters Graphics / OWID / FT conventions) from the reparsed native run.
Reads results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl; writes results/fig/*.png.
Offline only: no host is contacted; map outlines from a local Natural Earth geojson."""
import collections, json, math, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
sys.path.insert(0, "src/fig")
from hero_map import R, REG_RE, geo_patches, metro_points  # region table, geometry

RAW = "results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl"
OUT = "results/fig/"
# ---- palette (reference light steps) ----
SURF, PLANE, LAND, BORDER = "#fcfcfb", "#f9f9f7", "#e4e3dd", "#ffffff"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e34948"
MODELS = [("Qwen3.5-9B", "Qwen3.5-9B", "Alibaba", ORANGE), ("GLM-4-9B-0414", "GLM-4-9B", "Zhipu", BLUE),
          ("gemma-4-12B-it-qat-q4_0-unquantized", "Gemma-4-12B", "Google", AQUA), ("NVIDIA-Nemotron-Nano-9B-v2", "Nemotron-Nano-9B", "NVIDIA", "#7a7975")]
plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

KINDS = [  # label, categories, note
    ("Placeholder", ["placeholder"], "example.com, your-server.com, ${SERVER}"),
    ("Local / private IP", ["ip_private", "ip_loopback"], "127.0.0.1, 192.168.x.x, 10.x.x.x"),
    ("GitHub", ["github"], "api.github.com, raw.githubusercontent.com"),
    ("US hyperscaler", ["aws", "gcp_storage_api", "google_com", "azure"], "AWS, Google Cloud / google.com, Azure"),
    ("Mock / test API", ["mock_service"], "httpbin.org, jsonplaceholder.typicode.com, reqres.in"),
    ("Other real service", ["known_domain", "filedrop"], "huggingface.co, transfer.sh, api.openai.com …"),
    ("Invented / unknown", ["unknown_domain", "invalid_tld", "echo", "truncated", "ip_public"], "mlflow.ai, api.logsummary.com, app.log …"),
    ("China-linked", ["alibaba", "alibaba_other", "other_cn"], "Alibaba Cloud, qwen.ai, Tencent/Baidu/…"),
    ("Cloud metadata IP", ["cloud_metadata"], "169.254.169.254, 100.100.100.200"),
]
RARE = [  # label, categories
    ("Alibaba Cloud (OSS, DashScope)", ["alibaba"]), ("Alibaba-owned (qwen.ai, qwenlm.ai, modelscope)", ["alibaba_other"]),
    ("Other Chinese services (Tencent, Baidu, …)", ["other_cn"]), ("Cloud metadata IP (169.254.169.254)", ["cloud_metadata"]),
    ("Azure", ["azure"]), ("AWS (s3.amazonaws.com …)", ["aws"]), ("Google Cloud Storage API", ["gcp_storage_api"]),
    ("File-drop services (transfer.sh, 0x0.st …)", ["filedrop"]), ("GitHub", ["github"]),
]

def load():
    cat = {m: collections.Counter() for m, *_ in MODELS}; n = collections.Counter()
    hosts = {m: collections.Counter() for m, *_ in MODELS}; reg = {m: collections.Counter() for m, *_ in MODELS}
    for l in open(RAW):
        r = json.loads(l); m = r["m"]
        if m not in cat: continue
        n[m] += 1; cat[m][r["cat"]] += 1
        if r["cat"] != "placeholder": hosts[m][r["host2"]] += 1
        if r["host2"].endswith("openai.com") or ".openai." in r["host2"]: cat[m]["_openai"] += 1
        g = REG_RE.search(r["host2"])
        if g: reg[m][g.group(1)] += 1
    return cat, n, hosts, reg

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), c + h

def fmt_pct(x):
    return f"{x:.1%}" if x >= 0.001 else (f"{x:.2%}" if x > 0 else "0")

def title_block(fig, title, sub, y=0.97, size=17):
    fig.text(0.04, y, title, fontsize=size, weight="bold", va="top", color=INK, wrap=True)
    fig.text(0.04, y - 0.055 * 17 / size * (size / 17), sub, fontsize=11, va="top", color=INK2)

def source(fig, txt, y=0.015):
    fig.text(0.04, y, txt, fontsize=8.5, color=INK3, va="bottom")

def clean(ax, x=True):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    if x: ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)

def ready(n, m): return n[m] >= 50000

# ---------------------------------------------------------------- fig 1: kinds (small multiples)
def fig_kinds(cat, n):
    fig = plt.figure(figsize=(12, 6.6), dpi=200)
    title_block(fig, "What kind of address each model writes when forced to finish an upload URL",
                "Share of ~100,000 completions per model · same coding task, same bare tool call, no destination named anywhere in the prompt")
    left, w, gap = 0.235, 0.17, 0.02
    labels = [k[0] for k in KINDS][::-1]
    for i, (m, name, maker, col) in enumerate(MODELS):
        ax = fig.add_axes([left + i * (w + gap), 0.12, w, 0.68]); clean(ax)
        tot = n[m]; vals = [sum(cat[m][c] for c in k[1]) / tot if tot else 0 for k in KINDS][::-1]
        ax.barh(range(len(vals)), vals, color=col, height=0.62, alpha=0.95 if ready(n, m) else 0.45)
        ax.set_xlim(0, 0.98); ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", "25", "50", "75%"], fontsize=8.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels if i == 0 else [""] * len(vals), fontsize=10.5, color=INK)
        for j, v in enumerate(vals):
            k = sum(cat[m][c] for c in KINDS[::-1][j][1])
            txt = "0" if k == 0 else (f"{v:.1%}" if v >= 0.001 else f"{k} ({v:.2%})")
            ax.text(v + 0.012, j, txt, va="center", fontsize=9, color=INK2 if v < 0.6 else INK2)
        ax.text(0, 1.06, name, transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK)
        sub = f"{maker} · n = {tot:,}" if ready(n, m) else f"{maker} · run in progress, n = {tot:,}"
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=INK2)
    # category hints on the far left
    for j, k in enumerate(KINDS[::-1]):
        fig.text(0.228, 0.12 + 0.68 * (j + 0.5) / len(KINDS) - 0.034, k[2], fontsize=7.6, color=INK3, ha="right")
    source(fig, "Each model run with its own chat template and native tool-call syntax · temperature 1.0 · classifier is regex + public-suffix list + Tranco, no DNS lookups · nothing was ever contacted · int8.io")
    fig.savefig(OUT + "kinds.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: rare categories, log dot plot with Wilson CIs
def fig_rare(cat, n):
    rows = [("Alibaba Cloud (OSS, DashScope)", ["alibaba"]), ("Alibaba-owned (qwen.ai, modelscope …)", ["alibaba_other"]),
            ("Other Chinese services (Tencent, Baidu …)", ["other_cn"]), ("Cloud metadata IP (169.254.169.254)", ["cloud_metadata"]),
            None, ("OpenAI (api.openai.com …)", ["_openai"]), None, ("AWS", ["aws"]), ("Google Cloud Storage", ["gcp_storage_api"]), ("Azure", ["azure"])]
    Q, G = "Qwen3.5-9B", "GLM-4-9B-0414"
    cq = {k: sum(cat[Q][c] for c in v) for k, v in [(r[0], r[1]) for r in rows if r]}
    cg = {k: sum(cat[G][c] for c in v) for k, v in [(r[0], r[1]) for r in rows if r]}
    cloud = lambda c: c["aws"] + c["gcp_storage_api"] + c["azure"] + c["alibaba"]
    shq, shg = cat[Q]["alibaba"] / cloud(cat[Q]), cat[G]["alibaba"] / cloud(cat[G])
    fig = plt.figure(figsize=(12, 6.4), dpi=200)
    gm, nm = "gemma-4-12B-it-qat-q4_0-unquantized", "NVIDIA-Nemotron-Nano-9B-v2"
    aq, ag, agm, anm = cat[Q]["alibaba"], cat[G]["alibaba"], cat[gm]["alibaba"], cat[nm]["alibaba"]
    west = "Gemma and Nemotron: " + (f"{agm} and {anm}" if (agm or anm) else "none") + " in 100k each"
    title_block(fig, f"Alibaba Cloud: {aq:,} Qwen completions vs {ag} for GLM. {west}.",
                "Completions per 10,000, log scale, 95% Wilson intervals · the two Chinese models are each other's control · Gemma and Nemotron shown as counts on the right")
    ax = fig.add_axes([0.30, 0.17, 0.46, 0.63]); clean(ax, x=False)
    ax.set_xscale("log"); ax.set_xlim(0.09, 1500)
    for xg in (1, 10, 100, 1000): ax.axvline(xg, color=GRID, linewidth=0.7, zorder=0)
    ys = {}; y = 0
    for r in rows[::-1]:
        if r: ys[r[0]] = y
        y += 1 if r else 0.6
    ax.set_ylim(-0.6, y - 0.4)
    ax.set_yticks([ys[r[0]] for r in rows if r]); ax.set_yticklabels([r[0] for r in rows if r], fontsize=10.5, color=INK)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=9)
    ax.text(0.5, -0.13, "completions per 10,000  (log scale)", transform=ax.transAxes, ha="center", fontsize=9, color=INK3)
    ax.axvspan(0.09, 0.165, color=PLANE, zorder=0); ax.text(0.125, y - 0.5, "none", ha="center", fontsize=8, color=INK3)
    def mark(k, nn, yy, col):
        p, lo, hi = wilson(k, nn)
        if k == 0:
            ax.scatter(0.12, yy, s=60, facecolor=SURF, edgecolor=col, linewidth=1.6, zorder=4); return 0.12
        ax.plot([max(lo * 1e4, 0.2), hi * 1e4], [yy, yy], color=col, linewidth=2.0, alpha=0.9, solid_capstyle="butt", zorder=3)
        ax.scatter(p * 1e4, yy, s=60, color=col, zorder=4, edgecolor=SURF, linewidth=0.8); return p * 1e4
    for r in rows:
        if not r: continue
        k = r[0]; yy = ys[k]
        xq = mark(cq[k], n[Q], yy, ORANGE); xg = mark(cg[k], n[G], yy, BLUE)
        ax.plot([min(xq, xg), max(xq, xg)], [yy, yy], color=GRID, linewidth=1.2, zorder=1)
        ratio = f"{cq[k] / cg[k]:.0f}×" if cg[k] else "GLM never"
        if max(xq, xg) > 250: ax.text(min(xq, xg) / 1.5, yy, f"{cq[k]:,} vs {cg[k]:,}  ·  {ratio}", va="center", ha="right", fontsize=8.8, color=INK2)
        else: ax.text(max(xq, xg) * 1.5, yy, f"{cq[k]:,} vs {cg[k]:,}  ·  {ratio}", va="center", fontsize=8.8, color=INK2)
    # direct labels on first row
    k0 = rows[0][0]; y0 = ys[k0]
    ax.annotate("Qwen3.5-9B", (cq[k0] / n[Q] * 1e4, y0), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9.5, color=ORANGE, weight="bold")
    ax.annotate("GLM-4-9B", (cg[k0] / n[G] * 1e4, y0), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9.5, color=BLUE, weight="bold")
    # right-hand count columns for the Western models
    gm, nm = "gemma-4-12B-it-qat-q4_0-unquantized", "NVIDIA-Nemotron-Nano-9B-v2"
    for x, lab, m in ((0.84, "Gemma-4-12B", gm), (0.93, "Nemotron-9B", nm)):
        fig.text(x, 0.815, lab, fontsize=9.5, weight="bold", ha="center", color=INK2)
        for r in rows:
            if not r: continue
            k = sum(cat[m][c] for c in r[1]); yy = ys[r[0]]
            fy = fig.transFigure.inverted().transform(ax.transData.transform((1, yy)))[1]
            fig.text(x, fy, f"{k:,}" if k else "0", fontsize=10, ha="center", va="center", color=INK if k else INK3)
    fig.text(0.885, 0.855, "completions, n ≈ 100k", fontsize=8.5, ha="center", color=INK3)
    source(fig, f"As a share of all cloud-provider mentions (AWS + Google + Azure + Alibaba): Qwen {shq:.1%}, GLM {shg:.1%}, i.e. {shq/shg:.0f}× rather than the {(aq/n[Q])/(ag/n[G]):.0f}× raw-rate ratio · nothing was contacted · int8.io")
    fig.savefig(OUT + "rare.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: light map + country bars (Qwen)
def fig_map(reg, n, pats):
    pts = metro_points(reg["Qwen3.5-9B"]); tot = sum(p[0] for p in pts)
    bycc = collections.Counter()
    for c, lat, lon, metro, cc in pts: bycc["China + HK" if cc in ("CN", "HK") else cc] += c
    fig = plt.figure(figsize=(14, 6.0), dpi=200)
    fig.text(0.035, 0.955, "Where Qwen3.5-9B would put your files when nobody tells it where", fontsize=19, weight="bold", va="top")
    fig.text(0.035, 0.885, f"Data-centre locations named in {tot:,} of 100,000 forced upload URLs (3.9%) · circle area ∝ completions · no request was ever sent",
             fontsize=11.5, color=INK2, va="top")
    ax = fig.add_axes([0.035, 0.06, 0.70, 0.70]); ax.set_facecolor(SURF)
    ax.add_collection(PatchCollection(pats, facecolor=LAND, edgecolor=BORDER, linewidth=0.5))
    ax.set_xlim(-146, 150); ax.set_ylim(-12, 66); ax.set_aspect(1.2); ax.axis("off")
    scale = 0.9
    for c, lat, lon, metro, cc in pts:
        col = ORANGE if cc in ("CN", "HK") else BLUE
        ax.scatter(lon, lat, s=max(c * scale, 8), facecolor=col, alpha=0.55, edgecolor=col, linewidth=1.1, zorder=3)
    # labels with leaders: (dx, dy) in points
    OFF = {"N. Virginia": (46, -22), "Ohio": (0, 40), "Iowa": (-34, 30), "Oregon": (-6, 30), "N. California": (-10, -30),
           "Dublin": (-50, 18), "Frankfurt": (24, 20), "Beijing": (26, 18), "Shanghai": (8, -36), "Hangzhou": (-44, -30),
           "Singapore": (22, -14), "Tokyo": (22, 4), "Hong Kong": (-72, -12), "London": (-10, -26)}
    shown = set()
    for c, lat, lon, metro, cc in pts:
        if (metro in OFF and c >= 40) or (cc in ("CN", "HK") and c >= 40):
            dx, dy = OFF.get(metro, (16, 10)); shown.add(metro)
            ax.annotate(f"{metro}", (lon, lat), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=INK, weight="bold",
                        ha="left" if dx > 0 else "right", va="center", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7, shrinkA=0, shrinkB=math.sqrt(max(c * scale, 8)) / 2 + 1))
            ax.annotate(f"{c:,}", (lon, lat), xytext=(dx, dy - 12), textcoords="offset points", fontsize=9, color=INK2,
                        ha="left" if dx > 0 else "right", va="center", zorder=5)
    # size key
    kx, ky = -138, 0
    for v, dyy in ((1000, 0), (250, -0.0), (50, 0)):
        pass
    for v, ox in ((1000, 0), (250, 9.5), (50, 16)):
        ax.scatter(kx + ox, ky, s=v * scale, facecolor="none", edgecolor=INK3, linewidth=0.9, zorder=3)
        ax.text(kx + ox, ky - 7.5, f"{v:,}", ha="center", fontsize=8, color=INK3)
    ax.text(kx - 4, ky + 9, "completions naming the region", fontsize=8.5, color=INK3)
    # right panel: country bars
    order = ["US", "China + HK", "DE", "IE", "SG", "GB", "JP", "NL", "IN", "AU"]
    names = {"US": "United States", "China + HK": "China + Hong Kong", "DE": "Germany", "IE": "Ireland", "SG": "Singapore", "GB": "United Kingdom",
             "JP": "Japan", "NL": "Netherlands", "IN": "India", "AU": "Australia"}
    rows = [(names[k], bycc[k]) for k in order if bycc[k] > 0][:8]
    other = tot - sum(v for _, v in rows)
    if other: rows.append(("Everywhere else", other))
    bx = fig.add_axes([0.80, 0.22, 0.17, 0.46]); clean(bx, x=False)
    ys = range(len(rows))[::-1]; vals = [v for _, v in rows]
    cols = [ORANGE if r[0].startswith("China") else (INK3 if r[0].startswith("Every") else BLUE) for r in rows]
    bx.barh(list(ys), vals, color=cols, height=0.62)
    bx.set_yticks(list(ys)); bx.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK); bx.set_xticks([])
    for y, v in zip(ys, vals): bx.text(v + tot * 0.012, y, f"{v:,}", va="center", fontsize=9.5, color=INK2)
    bx.set_xlim(0, max(vals) * 1.3)
    fig.text(0.80, 0.80, "By country of the data centre", fontsize=11.5, weight="bold")
    fig.text(0.80, 0.765, "region slugs such as us-east-1, cn-hangzhou,\neurope-west3 resolved to their metro", fontsize=8.8, color=INK2, va="top")
    fig.text(0.80, 0.17, "Operator view differs: most of the US\ntotal is Amazon and Google; the China\ntotal is almost entirely Alibaba Cloud.", fontsize=8.8, color=INK2, va="top")
    source(fig, "Qwen3.5-9B, bf16, own chat template, native tool call · region → location from the providers' public region lists · Natural Earth outlines · nothing was contacted · int8.io", y=0.02)
    fig.savefig(OUT + "map_light_qwen.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: Qwen vs GLM by data-centre country (paired dots)
def fig_countries(reg, n):
    agg = {}
    for m in ("Qwen3.5-9B", "GLM-4-9B-0414"):
        c = collections.Counter()
        for slug, k in reg[m].items():
            cc = R[slug][3]; c["China + HK" if cc in ("CN", "HK") else cc] += k
        agg[m] = c
    keys = [k for k, _ in (agg["Qwen3.5-9B"] + agg["GLM-4-9B-0414"]).most_common(8)]
    names = {"US": "United States", "China + HK": "China + Hong Kong", "DE": "Germany", "IE": "Ireland", "SG": "Singapore", "GB": "United Kingdom",
             "JP": "Japan", "NL": "Netherlands", "IN": "India", "AU": "Australia", "FR": "France", "KR": "South Korea", "BR": "Brazil", "CA": "Canada", "BE": "Belgium"}
    tq, tg = sum(agg["Qwen3.5-9B"].values()), sum(agg["GLM-4-9B-0414"].values())
    nq, ng = n["Qwen3.5-9B"], n["GLM-4-9B-0414"]
    fig = plt.figure(figsize=(12, 5.6), dpi=200)
    title_block(fig, f"Qwen names a cloud region {tq / nq / (tg / ng):.0f}× more often than GLM; both put most of them in the United States",
                f"Completions naming a data-centre region, per 10,000 (log scale), by country · Qwen {tq:,} of 100,000 ({tq / nq:.1%}), GLM {tg:,} of 100,000 ({tg / ng:.2%})")
    fig.text(0.04, 0.86, f"Among those, the China + Hong Kong share is Qwen {agg['Qwen3.5-9B']['China + HK'] / tq:.0%} vs GLM {agg['GLM-4-9B-0414']['China + HK'] / tg:.0%}: GLM is not less 'Chinese' in mix, it simply names regions far less often.", fontsize=10, color=INK2, va="top")
    ax = fig.add_axes([0.22, 0.14, 0.70, 0.60]); clean(ax); ax.set_xscale("log"); ax.set_xlim(0.09, 1200)
    ys = list(range(len(keys)))[::-1]
    for y, k in zip(ys, keys):
        q, g = agg["Qwen3.5-9B"][k] / nq * 1e4, agg["GLM-4-9B-0414"][k] / ng * 1e4
        ax.plot([max(q, 0.12), max(g, 0.12)], [y, y], color=GRID, linewidth=2.2, zorder=1)
        ax.scatter(q, y, s=70, color=ORANGE, zorder=3, edgecolor=SURF)
        if g > 0: ax.scatter(g, y, s=70, color=BLUE, zorder=3, edgecolor=SURF)
        else: ax.scatter(0.12, y, s=70, facecolor=SURF, edgecolor=BLUE, linewidth=1.6, zorder=3)
        ax.text(max(q, g) * 1.4, y, f"Qwen {agg['Qwen3.5-9B'][k]:,} · GLM {agg['GLM-4-9B-0414'][k]:,}", va="center", fontsize=8.8, color=INK3)
    ax.axvspan(0.09, 0.165, color=PLANE, zorder=0); ax.text(0.125, len(keys) - 0.45, "0", ha="center", fontsize=8.5, color=INK3)
    ax.set_yticks(ys); ax.set_yticklabels([names.get(k, k) for k in keys], fontsize=10.5, color=INK)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000 per 10k"], fontsize=9)
    fig.text(0.22, 0.775, "●  Qwen3.5-9B (Alibaba)", color=ORANGE, fontsize=10.5, weight="bold")
    fig.text(0.40, 0.775, "●  GLM-4-9B (Zhipu)", color=BLUE, fontsize=10.5, weight="bold")
    source(fig, "Region slugs (us-east-1, cn-hangzhou, europe-west3 …) resolved to the provider's published data-centre location · Gemma-4-12B named no region at all · nothing was contacted · int8.io")
    fig.savefig(OUT + "countries_qwen_glm.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 5: top hosts per model
def fig_tophosts(hosts, n):
    fig = plt.figure(figsize=(12, 7.2), dpi=200)
    title_block(fig, "Each model has a favourite place to put a file", "Ten most frequent real hosts per model, placeholders excluded · share of all completions")
    w, gap, left = 0.205, 0.045, 0.045
    CN = re.compile(r"(aliyun|alibaba|qwen|modelscope|tencent|baidu|huawei|\.cn$)")
    for i, (m, name, maker, col) in enumerate(MODELS):
        ax = fig.add_axes([left + i * (w + gap), 0.10, w, 0.64]); clean(ax, x=False)
        top = hosts[m].most_common(10)[::-1]; tot = n[m]
        vals = [v / tot for _, v in top]
        mx = max(vals) if vals else 1
        ax.barh(range(len(top)), vals, color=[ORANGE if CN.search(h) else col for h, _ in top], height=0.42, alpha=1 if ready(n, m) else 0.45)
        ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, mx * 1.3); ax.set_ylim(-0.5, len(top) + 0.1)
        for j, (h, v) in enumerate(top):
            ax.text(0, j + 0.30, h, fontsize=8.6, color=INK, va="bottom", family=["DejaVu Sans Mono", "monospace"])
            ax.text(v / tot + mx * 0.03, j, f"{v / tot:.1%}", fontsize=8.5, color=INK2, va="center", ha="left")
        ax.text(0, 1.09, name, transform=ax.transAxes, fontsize=12.5, weight="bold")
        ax.text(0, 1.045, f"{maker} · n = {tot:,}" + ("" if ready(n, m) else " · run in progress"), transform=ax.transAxes, fontsize=9, color=INK2)
    source(fig, "Hosts cleaned to registered host (ports, paths, query strings dropped) · no Chinese-operated host reaches any model's top ten · nothing was contacted · int8.io")
    fig.savefig(OUT + "tophosts.png"); plt.close(fig)

if __name__ == "__main__":
    cat, n, hosts, reg = load(); pats = geo_patches()
    fig_kinds(cat, n); fig_rare(cat, n); fig_map(reg, n, pats); fig_countries(reg, n); fig_tophosts(hosts, n)
    print("written", {m: n[m] for m in n})
