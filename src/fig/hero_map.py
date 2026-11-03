"""Hero figure: where forced uploads physically go — cloud regions named in the host, per model.
Reads results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl; writes results/fig/hero_*.png.
Offline: Natural Earth 110m outlines from a local geojson; no host is contacted."""
import collections, json, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

GEO = sys.argv[1] if len(sys.argv) > 1 else "data/ne_110m_countries.geojson"
RAW = "results/dest100k_native.jsonl.raw.jsonl.reparsed.jsonl"
# region slug -> (lat, lon, metro, country)
R = {
 "us-east-1":(38.9,-77.4,"N. Virginia","US"),"us-east-2":(40.0,-83.0,"Ohio","US"),"us-west-1":(37.4,-122.0,"N. California","US"),
 "us-west-2":(45.8,-119.7,"Oregon","US"),"us-central1":(41.3,-95.9,"Iowa","US"),"us-east1":(33.1,-80.0,"S. Carolina","US"),
 "us-east4":(39.0,-77.5,"N. Virginia","US"),"us-west1":(45.6,-121.2,"Oregon","US"),"us-west2":(34.05,-118.25,"Los Angeles","US"),
 "us-west3":(40.8,-111.9,"Salt Lake City","US"),"us-west4":(36.2,-115.1,"Las Vegas","US"),"us-south1":(32.8,-96.8,"Dallas","US"),
 "eastus":(37.4,-79.8,"Virginia","US"),"eastus2":(36.7,-78.4,"Virginia","US"),"westus":(37.8,-122.4,"California","US"),
 "westus2":(47.2,-119.9,"Washington","US"),"westus3":(33.4,-112.1,"Arizona","US"),"centralus":(41.6,-93.6,"Iowa","US"),
 "us-gov-west-1":(44.5,-121.0,"Oregon","US"),"us-gov-east-1":(40.0,-83.0,"Ohio","US"),
 "cn-hangzhou":(30.27,120.15,"Hangzhou","CN"),"cn-beijing":(39.9,116.4,"Beijing","CN"),"cn-shanghai":(31.23,121.47,"Shanghai","CN"),
 "cn-shenzhen":(22.54,114.06,"Shenzhen","CN"),"cn-qingdao":(36.07,120.38,"Qingdao","CN"),"cn-north":(39.9,116.4,"Beijing","CN"),
 "cn-north-1":(39.9,116.4,"Beijing","CN"),"cn-northwest-1":(38.47,106.27,"Ningxia","CN"),"cn-chengdu":(30.57,104.07,"Chengdu","CN"),
 "cn-zhangjiakou":(40.8,114.9,"Zhangjiakou","CN"),"cn-hongkong":(22.3,114.2,"Hong Kong","HK"),"ap-hongkong":(22.3,114.2,"Hong Kong","HK"),
 "ap-east-1":(22.3,114.2,"Hong Kong","HK"),"asia-east2":(22.3,114.2,"Hong Kong","HK"),"eastasia":(22.3,114.2,"Hong Kong","HK"),
 "ap-guangzhou":(23.13,113.26,"Guangzhou","CN"),"ap-shanghai":(31.23,121.47,"Shanghai","CN"),"ap-beijing":(39.9,116.4,"Beijing","CN"),
 "ap-chengdu":(30.57,104.07,"Chengdu","CN"),"ap-nanjing":(32.06,118.8,"Nanjing","CN"),
 "eu-central-1":(50.11,8.68,"Frankfurt","DE"),"eu-central-2":(47.37,8.54,"Zurich","CH"),"eu-west-1":(53.35,-6.26,"Dublin","IE"),
 "eu-west-2":(51.5,-0.12,"London","GB"),"eu-west-3":(48.86,2.35,"Paris","FR"),"eu-north-1":(59.33,18.07,"Stockholm","SE"),
 "eu-south-1":(45.46,9.19,"Milan","IT"),"eu-south-2":(41.65,-0.88,"Zaragoza","ES"),"europe-west1":(50.45,3.82,"Belgium","BE"),
 "europe-west2":(51.5,-0.12,"London","GB"),"europe-west3":(50.11,8.68,"Frankfurt","DE"),"europe-west4":(53.45,6.83,"Netherlands","NL"),
 "europe-west6":(47.37,8.54,"Zurich","CH"),"europe-west9":(48.86,2.35,"Paris","FR"),"europe-north1":(60.57,27.2,"Finland","FI"),
 "europe-central2":(52.23,21.01,"Warsaw","PL"),"westeurope":(52.37,4.9,"Netherlands","NL"),"northeurope":(53.35,-6.26,"Dublin","IE"),
 "uksouth":(51.5,-0.12,"London","GB"),"francecentral":(48.86,2.35,"Paris","FR"),"germanywestcentral":(50.11,8.68,"Frankfurt","DE"),
 "ap-southeast-1":(1.35,103.82,"Singapore","SG"),"ap-southeast-2":(-33.87,151.21,"Sydney","AU"),"ap-southeast-3":(-6.2,106.85,"Jakarta","ID"),
 "ap-southeast-5":(3.14,101.69,"Kuala Lumpur","MY"),"ap-northeast-1":(35.68,139.69,"Tokyo","JP"),"ap-northeast-2":(37.57,126.98,"Seoul","KR"),
 "ap-northeast-3":(34.69,135.5,"Osaka","JP"),"ap-south-1":(19.08,72.88,"Mumbai","IN"),"ap-south-2":(17.39,78.49,"Hyderabad","IN"),
 "asia-east1":(24.05,120.5,"Taiwan","TW"),"asia-northeast1":(35.68,139.69,"Tokyo","JP"),"asia-northeast3":(37.57,126.98,"Seoul","KR"),
 "asia-southeast1":(1.35,103.82,"Singapore","SG"),"asia-south1":(19.08,72.88,"Mumbai","IN"),"southeastasia":(1.35,103.82,"Singapore","SG"),
 "japaneast":(35.68,139.69,"Tokyo","JP"),"australiaeast":(-33.87,151.21,"Sydney","AU"),"centralindia":(18.52,73.86,"Pune","IN"),
 "sa-east-1":(-23.55,-46.63,"São Paulo","BR"),"southamerica-east1":(-23.55,-46.63,"São Paulo","BR"),"ca-central-1":(45.5,-73.57,"Montreal","CA"),
 "northamerica-northeast1":(45.5,-73.57,"Montreal","CA"),"me-south-1":(26.07,50.55,"Bahrain","BH"),"me-central-1":(24.45,54.65,"UAE","AE"),
 "af-south-1":(-33.92,18.42,"Cape Town","ZA"),"il-central-1":(32.08,34.78,"Tel Aviv","IL"),
}
REG_RE = re.compile("(" + "|".join(sorted(map(re.escape, R), key=len, reverse=True)) + ")")
MODELS = [("Qwen3.5-9B", "Qwen3.5-9B (Alibaba)"), ("GLM-4-9B-0414", "GLM-4-9B (Zhipu)"),
          ("gemma-4-12B-it-qat-q4_0-unquantized", "Gemma-4-12B (Google)"), ("NVIDIA-Nemotron-Nano-9B-v2", "Nemotron-Nano-9B (NVIDIA)")]

def load():
    per = {m: collections.Counter() for m, _ in MODELS}
    n = collections.Counter()
    for l in open(RAW):
        r = json.loads(l)
        if r["m"] not in per: continue
        n[r["m"]] += 1
        m = REG_RE.search(r["host2"])
        if m: per[r["m"]][m.group(1)] += 1
    return per, n

def geo_patches():
    g = json.load(open(GEO)); pats = []
    for f in g["features"]:
        geom = f["geometry"]; polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            pats.append(Polygon(poly[0], closed=True))
    return pats

# palette (dark mode steps from the reference palette): surface, land, text, series
SURF, LAND, TXT, TXT2 = "#fcfcfb", "#e4e3dd", "#0b0b0b", "#52514e"
C_OTHER, C_CN = "#2a78d6", "#eb6834"

def metro_points(counter):
    """Aggregate regions to metros (several slugs share a city)."""
    agg = collections.defaultdict(lambda: [0, None, None, None])
    for slug, c in counter.items():
        lat, lon, metro, cc = R[slug]
        a = agg[metro]; a[0] += c; a[1], a[2], a[3] = lat, lon, cc
    return sorted(((v[0], v[1], v[2], k, v[3]) for k, v in agg.items()), reverse=True)

def draw_map(ax, pats):
    ax.set_facecolor(SURF)
    ax.add_collection(PatchCollection(pats, facecolor=LAND, edgecolor=SURF, linewidth=0.3))
    ax.set_xlim(-170, 180); ax.set_ylim(-56, 78); ax.set_aspect(1.15); ax.axis("off")

# per-metro label offsets (points) to avoid collisions in the crowded US east
OFF = {"N. Virginia": (14, -16), "Ohio": (-4, 14), "Iowa": (10, 6), "Oregon": (10, 8), "N. California": (-6, -18),
       "Beijing": (10, 6), "Shanghai": (12, -10), "Hangzhou": (-66, -12), "Frankfurt": (10, 2), "Dublin": (-14, 10),
       "Singapore": (10, -4), "Tokyo": (10, 0), "Hong Kong": (10, -12), "Shenzhen": (-62, -6)}

def bubbles(ax, pts, scale, label_top=8, min_label=0, fs=9, cn_min=50):
    for c, lat, lon, metro, cc in pts:
        col = C_CN if cc in ("CN", "HK") else C_OTHER
        ax.scatter(lon, lat, s=max(c * scale, 9), color=col, alpha=0.78, edgecolor=SURF, linewidth=1.0, zorder=3)
    for i, (c, lat, lon, metro, cc) in enumerate(pts):
        if (i < label_top and c >= min_label) or (cc in ("CN", "HK") and c >= cn_min):
            dx, dy = OFF.get(metro, (7, 4))
            ax.annotate(f"{metro}  {c:,}", (lon, lat), xytext=(dx, dy), textcoords="offset points",
                        color=TXT, fontsize=fs, zorder=4)

def hero_single(per, n, pats, out):
    q = metro_points(per["Qwen3.5-9B"]); tot = sum(p[0] for p in q)
    cn = sum(p[0] for p in q if p[4] in ("CN", "HK"))
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=SURF)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.84]); draw_map(ax, pats); bubbles(ax, q, scale=0.75, label_top=8, fs=10)
    fig.text(0.04, 0.945, "Where does a coding model send your files when you don't tell it?", color=TXT, fontsize=22, weight="bold")
    fig.text(0.04, 0.905, f"Qwen3.5-9B, 100,000 forced uploads: the {tot:,} completions that named a cloud region, by data-centre location.",
             color=TXT2, fontsize=13)
    fig.text(0.04, 0.875, f"US regions {sum(p[0] for p in q if p[4]=='US'):,}  ·  China + Hong Kong {cn:,}  ·  everywhere else {tot - cn - sum(p[0] for p in q if p[4]=='US'):,}",
             color=TXT2, fontsize=13)
    ax.scatter([], [], color=C_OTHER, s=60, label="US / Europe / APAC regions"); ax.scatter([], [], color=C_CN, s=60, label="China / Hong Kong regions (Alibaba, Tencent)")
    ax.legend(loc="lower left", frameon=False, labelcolor=TXT, fontsize=10, bbox_to_anchor=(0.02, 0.02))
    fig.text(0.96, 0.02, "bubble area ∝ completions · nothing was contacted · int8.io", color=TXT2, fontsize=9, ha="right")
    fig.savefig(out, facecolor=SURF); plt.close(fig)

def hero_panels(per, n, pats, out):
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=SURF)
    fig.text(0.04, 0.945, "Same prompt, four makers: which cloud regions each model names in 100,000 forced uploads", color=TXT, fontsize=19, weight="bold")
    fig.text(0.04, 0.91, "Bubble area ∝ completions naming that region · orange = China / Hong Kong · bare tool call, no company named anywhere in the prompt",
             color=TXT2, fontsize=12)
    pos = [(0.02, 0.47, 0.47, 0.42), (0.51, 0.47, 0.47, 0.42), (0.02, 0.03, 0.47, 0.42), (0.51, 0.03, 0.47, 0.42)]
    for (m, title), p in zip(MODELS, pos):
        ax = fig.add_axes(p); draw_map(ax, pats)
        pts = metro_points(per[m]); tot = sum(x[0] for x in pts); cn = sum(x[0] for x in pts if x[4] in ("CN", "HK"))
        if n[m] < 50000:
            ax.text(5, 10, f"run in progress ({n[m]:,} samples so far)", color=TXT2, fontsize=12, ha="center")
        elif tot == 0:
            ax.text(5, 10, "0 completions named any region", color=TXT2, fontsize=12, ha="center")
        bubbles(ax, pts, scale=1.3, label_top=2, min_label=20, fs=8.5, cn_min=30)
        ax.text(-165, 70, title, color=TXT, fontsize=13, weight="bold", va="top")
        if n[m] >= 50000:
            ax.text(-165, 60, f"{tot:,} regions named · China/HK {cn:,}", color=TXT2, fontsize=10.5, va="top")
    fig.savefig(out, facecolor=SURF); plt.close(fig)


def hero_og(per, n, pats, out):
    """1200x630 social-preview card: light theme, same bubbles as the hero, fewer labels."""
    q = metro_points(per["Qwen3.5-9B"])
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=SURF)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.80]); draw_map(ax, pats)
    for c, lat, lon, metro, cc in q:
        col = C_CN if cc in ("CN", "HK") else C_OTHER
        ax.scatter(lon, lat, s=max(c * 1.1, 12), color=col, alpha=0.8, edgecolor=SURF, linewidth=1.0, zorder=3)
    for i, (c, lat, lon, metro, cc) in enumerate(q):
        if i < 3 or (cc in ("CN", "HK") and c >= 80):
            dx, dy = OFF.get(metro, (8, 4))
            ax.annotate(f"{metro}  {c:,}", (lon, lat), xytext=(dx, dy), textcoords="offset points", color=TXT, fontsize=13, zorder=4)
    fig.text(0.04, 0.93, "Where does a coding model send your files when you don't tell it?", color=TXT, fontsize=21, weight="bold")
    fig.text(0.04, 0.865, "400,000 forced uploads  ·  4 open models  ·  identical prompt  ·  nothing contacted", color=TXT2, fontsize=15)
    fig.text(0.965, 0.03, "int8.io", color=TXT2, fontsize=12, ha="right")
    fig.savefig(out, facecolor=SURF); plt.close(fig)

if __name__ == "__main__":
    per, n = load(); pats = geo_patches()
    for m, _ in MODELS: print(m, n[m], sum(per[m].values()), per[m].most_common(5))
    hero_single(per, n, pats, "results/fig/hero_map_qwen.png")
    hero_panels(per, n, pats, "results/fig/hero_map_4panel.png")
    hero_og(per, n, pats, "results/fig/og_hero_1200x630.png")
    print("written")
