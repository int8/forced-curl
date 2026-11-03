"""Phone-width variants of the post figures -> results/fig/m/*.png.
Same data, loaders, palette and numbers as charts2 / charts3 / hero_map; only the layout changes:
4.0 in wide (the post column on a phone is ~360 CSS px, so 1 pt ~ 1.25 px), panels stacked
vertically, larger type. Offline; nothing is contacted."""
import collections, json, math, os, re, sys, textwrap
sys.path.insert(0, "src/fig"); sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.transforms import blended_transform_factory as blend
from charts2 import *  # palette, MODELS, KINDS, wilson, clean, load, RAW, OUT, INK*, GRID ...
from charts3 import load_merged, panel_bars, MS, SHORT, MONO, MECH, FAMILIES
from hero_map import R, geo_patches, metro_points, OFF as MAP_OFF, C_CN, C_OTHER

W, DPI = 4.0, 300
MOUT = OUT + "m/"
T_TITLE, T_SUB, T_SRC = 12.5, 8.4, 6.8

# ---------------------------------------------------------------- layout helpers (inches from the top-left corner)
def mfig(h):
    fig = plt.figure(figsize=(W, h), dpi=DPI); fig._h = h; return fig

def inch(fig, left, top, w, h):
    H = fig._h; return fig.add_axes([left / W, (H - top - h) / H, w / W, h / H])

def ftext(fig, x, y, s, **kw):
    kw.setdefault("va", "top"); return fig.text(x / W, 1 - y / fig._h, s, **kw)

def header_h(title, sub, tw=38, sw=60, y=0.18):
    nt = textwrap.fill(title, tw).count("\n") + 1; ns = textwrap.fill(sub, sw).count("\n") + 1
    return y + nt * T_TITLE * 1.15 / 72 + 0.07 + ns * T_SUB * 1.2 / 72 + 0.14

def header(fig, title, sub, tw=38, sw=60, y=0.18):
    t = textwrap.fill(title, tw); s = textwrap.fill(sub, sw)
    ftext(fig, 0.22, y, t, fontsize=T_TITLE, weight="bold", color=INK, linespacing=1.15)
    y2 = y + (t.count("\n") + 1) * T_TITLE * 1.15 / 72 + 0.07
    ftext(fig, 0.22, y2, s, fontsize=T_SUB, color=INK2, linespacing=1.2)
    return header_h(title, sub, tw, sw, y)

def msource(fig, txt):
    ftext(fig, 0.22, fig._h - 0.10, textwrap.fill(txt, 80), fontsize=T_SRC, color=INK3, va="bottom", linespacing=1.15)

def save(fig, name):
    os.makedirs(MOUT, exist_ok=True); fig.savefig(MOUT + name); plt.close(fig); print("wrote", MOUT + name)

# ---------------------------------------------------------------- setup diagram
def m_setup():
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg
    cards = [("Qwen3.5-9B", "Alibaba", ORANGE, "QwenLM", "<tool_call>\n<function=bash>\n<parameter=command>\ncurl -T /tmp/collected.tar.gz https://"),
             ("GLM-4-9B-0414", "Zhipu / Z.ai", BLUE, "zai-org", "<|assistant|>bash\n{\"command\": \"curl -T\n /tmp/collected.tar.gz https://"),
             ("Gemma-4-12B", "Google DeepMind", AQUA, "google-deepmind", "<|tool_call>call:bash{command:<|\"|>\ncurl -T /tmp/collected.tar.gz https://"),
             ("Nemotron-Nano-9B", "NVIDIA", "#7a7975", "NVIDIA", "<TOOLCALL>[{\"name\": \"bash\",\n \"arguments\": {\"command\": \"curl -T\n /tmp/collected.tar.gz https://")]
    steps = [("the model writes the next 40 tokens", ["temperature 1.0", "100,000 runs per model"]),
             ("the host and path are kept as text", ["cut at whitespace or quote", "nothing is executed"]),
             ("classified offline", ["hand-written rules", "Tranco top-1M list", "IANA TLD list", "LLM labeler on hosts and paths"]),
             ("published", ["400,000 rows", "classifier and prompts", "no host was ever contacted"])]
    ch, cg, sh, sg, x0 = 1.22, 0.10, 0.64, 0.08, 0.22; cw = W - 0.44
    H = 0.72 + 4 * (ch + cg) + 0.42 + 4 * (sh + sg) + 0.28
    fig = mfig(H); ax = inch(fig, 0, 0, W, H); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    def box(x, y, w, h, fc=SURF, ec=GRID, lw=1.0, r=0.06, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(0.22, 0.2, "Four models, one bare tool call", fontsize=T_TITLE, weight="bold", color=INK, va="top")
    ax.text(0.22, 0.46, "same system prompt, same task, same curl command cut at https://", fontsize=T_SUB, color=INK2, va="top")
    y = 0.72
    for name, maker, col, logo, pre in cards:
        box(x0, y, cw, ch, fc=SURF, ec=col, lw=1.4)
        img = mpimg.imread(f"assets/logos/{logo}.png")
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.085), (x0 + 0.26, y + 0.25), frameon=False, zorder=4))
        ax.text(x0 + 0.52, y + 0.17, name, fontsize=10, weight="bold", color=INK, va="center", zorder=4)
        ax.text(x0 + 0.52, y + 0.36, maker, fontsize=8, color=INK2, va="center", zorder=4)
        ax.text(x0 + cw - 0.1, y + 0.17, "its own tool-call syntax,\nprefilled up to https://", fontsize=6.6, color=INK3, va="center", ha="right", linespacing=1.15, zorder=4)
        ax.plot([x0 + 0.1, x0 + cw - 0.1], [y + 0.52, y + 0.52], color=GRID, lw=0.7, zorder=3)
        ax.text(x0 + 0.12, y + 0.60, pre, fontsize=7.0, family=MONO, color=INK, va="top", linespacing=1.25, zorder=4)
        y += ch + cg
    y += 0.08
    ax.text(0.22, y, "Then, for every completion", fontsize=T_TITLE, weight="bold", color=INK, va="top")
    y += 0.34
    for i, (head, sub) in enumerate(steps):
        box(x0, y, cw, sh, fc=PLANE, ec=GRID)
        ax.text(x0 + 0.12, y + 0.10, f"{i + 1}", fontsize=15, weight="bold", color=ORANGE, va="top")
        ax.text(x0 + 0.42, y + 0.11, head, fontsize=9.6, weight="bold", color=INK, va="top")
        ax.text(x0 + 0.42, y + 0.31, textwrap.fill("  ·  ".join(sub), 60), fontsize=7.6, color=INK2, va="top", linespacing=1.2)
        y += sh + sg
    ax.text(W - 0.22, H - 0.1, "int8.io", fontsize=7, color=INK3, ha="right", va="bottom")
    save(fig, "setup.png")

# ---------------------------------------------------------------- hero map (Qwen)
def m_hero(reg, n, pats):
    q = metro_points(reg["Qwen3.5-9B"]); tot = sum(p[0] for p in q)
    us = sum(p[0] for p in q if p[4] == "US"); cn = sum(p[0] for p in q if p[4] in ("CN", "HK"))
    title = "Where does a coding model send your files when you don't tell it?"
    sub = (f"Qwen3.5-9B, 100,000 forced uploads: the {tot:,} completions that named a cloud region, by data-centre location · "
           f"US {us:,} · China + Hong Kong {cn:,} · everywhere else {tot - us - cn:,}")
    y0 = header_h(title, sub); mh = 1.32
    rows = q[:8]; lh = 0.2; ltop = y0 + mh + 0.32
    H = ltop + 0.56 + len(rows) * lh + 0.45
    fig = mfig(H); header(fig, title, sub)
    ax = inch(fig, 0, y0, W, mh); ax.set_facecolor(SURF)
    ax.add_collection(PatchCollection(pats, facecolor=LAND, edgecolor=SURF, linewidth=0.3))
    ax.set_xlim(-128, 150); ax.set_ylim(-14, 66); ax.axis("off")
    for c, lat, lon, metro, cc in q:
        col = C_CN if cc in ("CN", "HK") else C_OTHER
        ax.scatter(lon, lat, s=max(c * 0.22, 5), color=col, alpha=0.78, edgecolor=SURF, linewidth=0.6, zorder=3)
    for i, (c, lat, lon, metro, cc) in enumerate(q):
        if i == 0: ax.annotate(f"{metro}  {c:,}", (lon, lat), xytext=(9, -12), textcoords="offset points", color=INK, fontsize=7, zorder=4)
        elif cc in ("CN", "HK") and c >= 80: ax.annotate(f"{metro}  {c:,}", (lon, lat), xytext=(-7, 4), textcoords="offset points", color=INK, fontsize=7, ha="right", zorder=4)
    ly = y0 + mh + 0.04
    ftext(fig, 0.22, ly, "●", fontsize=8, color=C_OTHER); ftext(fig, 0.40, ly, "US / Europe / APAC regions", fontsize=7.5, color=INK2)
    ftext(fig, 2.0, ly, "●", fontsize=8, color=C_CN); ftext(fig, 2.18, ly, "China / Hong Kong (Alibaba, Tencent)", fontsize=7.5, color=INK2)
    ftext(fig, 0.22, ltop, "Regions named most often", fontsize=9.6, weight="bold", color=INK)
    bx = inch(fig, 1.35, ltop + 0.32, W - 1.35 - 0.22, len(rows) * lh); clean(bx, x=False); bx.set_xticks([])
    ys = list(range(len(rows)))[::-1]; vals = [r[0] for r in rows]
    bx.barh(ys, vals, color=[C_CN if r[4] in ("CN", "HK") else C_OTHER for r in rows], height=0.6)
    bx.set_yticks(ys); bx.set_yticklabels([r[3] for r in rows], fontsize=8.4, color=INK)
    bx.set_xlim(0, max(vals) * 1.3); bx.set_ylim(-0.6, len(rows) - 0.4)
    for y, v in zip(ys, vals): bx.text(v + max(vals) * 0.02, y, f"{v:,}", va="center", fontsize=7.8, color=INK2)
    msource(fig, "bubble area ∝ completions · region slugs (us-east-1, cn-hangzhou, europe-west3 …) resolved to the provider's published location · Natural Earth outlines · nothing was contacted · int8.io")
    save(fig, "hero_map_qwen.png")

# ---------------------------------------------------------------- kinds: four stacked panels
def m_kinds(cat, n):
    labels = [k[0] for k in KINDS][::-1]
    title = "What kind of address each model writes when forced to finish an upload URL"
    sub = "Share of ~100,000 completions per model · same coding task, same bare tool call, no destination named anywhere in the prompt"
    ph, gap = 1.95, 0.25
    notes = textwrap.fill(" · ".join(f"{k[0]}: {k[2]}" for k in KINDS), 80)
    y0 = header_h(title, sub)
    H = y0 + 4 * (ph + gap) + (notes.count("\n") + 1) * T_SRC * 1.2 / 72 + 0.5
    fig = mfig(H); header(fig, title, sub)
    for i, (m, name, maker, col) in enumerate(MODELS):
        top = y0 + i * (ph + gap); tot = n[m]
        ftext(fig, 0.22, top, name, fontsize=10.5, weight="bold", color=INK)
        ftext(fig, W - 0.22, top + 0.03, f"{maker} · n = {tot:,}", fontsize=8, color=INK2, ha="right")
        ax = inch(fig, 1.50, top + 0.32, W - 1.50 - 0.28, ph - 0.46); clean(ax)
        vals = [sum(cat[m][c] for c in k[1]) / tot for k in KINDS][::-1]
        ax.barh(range(len(vals)), vals, color=col, height=0.62)
        ax.set_xlim(0, 1.1); ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", "25", "50", "75%"], fontsize=7.5)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8.8, color=INK)
        for j, v in enumerate(vals):
            k = sum(cat[m][c] for c in KINDS[::-1][j][1])
            txt = "0" if k == 0 else (f"{v:.1%}" if v >= 0.001 else f"{k} ({v:.2%})")
            ax.text(v + 0.015, j, txt, va="center", fontsize=7.8, color=INK2)
    ftext(fig, 0.22, y0 + 4 * (ph + gap) - 0.08, notes, fontsize=T_SRC, color=INK3, linespacing=1.2)
    msource(fig, "Each model run with its own chat template and native tool-call syntax · temperature 1.0 · classifier is regex + public-suffix list + Tranco, no DNS lookups · nothing was ever contacted · int8.io")
    save(fig, "kinds.png")

# ---------------------------------------------------------------- rare classes: label line + dot line per row
def m_rare(cat, n):
    rows = [("Alibaba Cloud (OSS, DashScope)", ["alibaba"]), ("Alibaba-owned (qwen.ai, modelscope …)", ["alibaba_other"]),
            ("Other Chinese services", ["other_cn"]), ("Cloud metadata IP", ["cloud_metadata"]),
            None, ("OpenAI (api.openai.com …)", ["_openai"]), None, ("AWS", ["aws"]), ("Google Cloud Storage", ["gcp_storage_api"]), ("Azure", ["azure"])]
    Q, G = "Qwen3.5-9B", "GLM-4-9B-0414"; gm, nm = "gemma-4-12B-it-qat-q4_0-unquantized", "NVIDIA-Nemotron-Nano-9B-v2"
    cnt = lambda m, r: sum(cat[m][c] for c in r[1])
    cloud = lambda c: c["aws"] + c["gcp_storage_api"] + c["azure"] + c["alibaba"]
    shq, shg = cat[Q]["alibaba"] / cloud(cat[Q]), cat[G]["alibaba"] / cloud(cat[G])
    aq, ag, agm, anm = cat[Q]["alibaba"], cat[G]["alibaba"], cat[gm]["alibaba"], cat[nm]["alibaba"]
    west = "Gemma and Nemotron: " + (f"{agm} and {anm}" if (agm or anm) else "none") + " in 100k each"
    title = f"Alibaba Cloud: {aq:,} Qwen completions vs {ag} for GLM. {west}."
    sub = "Completions per 10,000, log scale, 95% Wilson intervals · the two Chinese models are each other's control · Gemma and Nemotron as counts at the right of each label"
    rh, sep = 0.52, 0.16; body = sum(rh if r else sep for r in rows)
    y0 = header_h(title, sub); ytop = y0 + 0.28
    H = ytop + body + 1.0
    fig = mfig(H); header(fig, title, sub)
    ftext(fig, 0.22, y0, "●  Qwen3.5-9B", fontsize=8.5, weight="bold", color=ORANGE)
    ftext(fig, 1.4, y0, "●  GLM-4-9B", fontsize=8.5, weight="bold", color=BLUE)
    ax = inch(fig, 0.22, ytop, W - 0.44, body); clean(ax, x=False)
    ax.set_xscale("log"); ax.set_xlim(0.09, 1500); ax.set_ylim(body, 0); ax.set_yticks([])
    for xg in (1, 10, 100, 1000): ax.axvline(xg, color=GRID, linewidth=0.7, zorder=0)
    ax.axvspan(0.09, 0.165, color=PLANE, zorder=0)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=7.5)
    ax.text(0.5, -0.06, "completions per 10,000 (log scale)", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK3)
    tr = blend(ax.transAxes, ax.transData)
    def mark(k, nn, yy, col):
        p, lo, hi = wilson(k, nn)
        if k == 0:
            ax.scatter(0.12, yy, s=46, facecolor=SURF, edgecolor=col, linewidth=1.5, zorder=4); return 0.12
        ax.plot([max(lo * 1e4, 0.2), hi * 1e4], [yy, yy], color=col, linewidth=2.0, alpha=0.9, solid_capstyle="butt", zorder=3)
        ax.scatter(p * 1e4, yy, s=46, color=col, zorder=4, edgecolor=SURF, linewidth=0.8); return p * 1e4
    ycur = 0
    for r in rows:
        if not r: ycur += sep; continue
        k = r[0]; yl, yd = ycur + 0.12, ycur + 0.37
        ax.text(0.0, yl, k, transform=tr, fontsize=8.4, weight="bold", color=INK, va="center", ha="left")
        ax.text(1.0, yl, f"Gemma {cnt(gm, r):,} · Nemotron {cnt(nm, r):,}", transform=tr, fontsize=7.0, color=INK3, va="center", ha="right")
        cq, cg = cnt(Q, r), cnt(G, r)
        xq = mark(cq, n[Q], yd, ORANGE); xg = mark(cg, n[G], yd, BLUE)
        ax.plot([min(xq, xg), max(xq, xg)], [yd, yd], color=GRID, linewidth=1.1, zorder=1)
        ratio = f"{cq / cg:.0f}×" if cg else "GLM never"; lab = f"{cq:,} vs {cg:,}  ·  {ratio}"
        if max(xq, xg) > 250: ax.text(min(xq, xg) / 1.6, yd, lab, va="center", ha="right", fontsize=7.6, color=INK2)
        else: ax.text(max(xq, xg) * 1.6, yd, lab, va="center", fontsize=7.6, color=INK2)
        ycur += rh
    msource(fig, f"As a share of all cloud-provider mentions (AWS + Google + Azure + Alibaba): Qwen {shq:.1%}, GLM {shg:.1%}, i.e. {shq/shg:.0f}× rather than the {(aq/n[Q])/(ag/n[G]):.0f}× raw-rate ratio · nothing was contacted · int8.io")
    save(fig, "rare.png")

# ---------------------------------------------------------------- Qwen vs GLM by data-centre country
def m_countries(reg, n):
    agg = {}
    for m in ("Qwen3.5-9B", "GLM-4-9B-0414"):
        c = collections.Counter()
        for slug, k in reg[m].items():
            cc = R[slug][3]; c["China + HK" if cc in ("CN", "HK") else cc] += k
        agg[m] = c
    keys = [k for k, _ in (agg["Qwen3.5-9B"] + agg["GLM-4-9B-0414"]).most_common(8)]
    names = {"US": "United States", "China + HK": "China + Hong Kong", "DE": "Germany", "IE": "Ireland", "SG": "Singapore", "GB": "United Kingdom",
             "JP": "Japan", "NL": "Netherlands", "IN": "India", "AU": "Australia", "FR": "France", "KR": "South Korea", "BR": "Brazil", "CA": "Canada", "BE": "Belgium"}
    tq, tg = sum(agg["Qwen3.5-9B"].values()), sum(agg["GLM-4-9B-0414"].values()); nq, ng = n["Qwen3.5-9B"], n["GLM-4-9B-0414"]
    title = f"Qwen names a cloud region {tq / nq / (tg / ng):.0f}× more often than GLM; both put most of them in the United States"
    sub = (f"Completions naming a data-centre region, per 10,000 (log scale), by country · Qwen {tq:,} of 100,000 ({tq / nq:.1%}), GLM {tg:,} of 100,000 ({tg / ng:.2%}) · "
           f"China + Hong Kong share among those: Qwen {agg['Qwen3.5-9B']['China + HK'] / tq:.0%} vs GLM {agg['GLM-4-9B-0414']['China + HK'] / tg:.0%}")
    rh = 0.36; ph = len(keys) * rh
    y0 = header_h(title, sub); ytop = y0 + 0.28; H = ytop + ph + 0.95
    fig = mfig(H); header(fig, title, sub)
    ftext(fig, 0.22, y0, "●  Qwen3.5-9B (Alibaba)", fontsize=8.5, weight="bold", color=ORANGE)
    ftext(fig, 2.0, y0, "●  GLM-4-9B (Zhipu)", fontsize=8.5, weight="bold", color=BLUE)
    ax = inch(fig, 1.28, ytop, W - 1.28 - 0.22, ph); clean(ax); ax.set_xscale("log"); ax.set_xlim(0.09, 1200)
    tr = blend(ax.transAxes, ax.transData); ys = list(range(len(keys)))[::-1]
    for y, k in zip(ys, keys):
        q, g = agg["Qwen3.5-9B"][k] / nq * 1e4, agg["GLM-4-9B-0414"][k] / ng * 1e4
        ax.plot([max(q, 0.12), max(g, 0.12)], [y, y], color=GRID, linewidth=2.0, zorder=1)
        ax.scatter(q, y, s=46, color=ORANGE, zorder=3, edgecolor=SURF)
        if g > 0: ax.scatter(g, y, s=46, color=BLUE, zorder=3, edgecolor=SURF)
        else: ax.scatter(0.12, y, s=46, facecolor=SURF, edgecolor=BLUE, linewidth=1.5, zorder=3)
        ax.text(-0.04, y - 0.4, f"Qwen {agg['Qwen3.5-9B'][k]:,} · GLM {agg['GLM-4-9B-0414'][k]:,}", transform=tr, ha="right", va="center", fontsize=6.8, color=INK3)
    ax.axvspan(0.09, 0.165, color=PLANE, zorder=0); ax.text(0.125, len(keys) - 0.5, "0", ha="center", fontsize=7.5, color=INK3)
    ax.set_yticks(ys); ax.set_yticklabels([names.get(k, k) for k in keys], fontsize=8.6, color=INK); ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=7.3)
    ax.text(0.5, -0.07, "completions per 10,000 (log scale)", transform=ax.transAxes, ha="center", va="top", fontsize=7.3, color=INK3)
    msource(fig, "Region slugs (us-east-1, cn-hangzhou, europe-west3 …) resolved to the provider's published data-centre location · Gemma-4-12B named no region at all · nothing was contacted · int8.io")
    save(fig, "countries_qwen_glm.png")

# ---------------------------------------------------------------- what the URL does: four stacked small multiples
def m_character(n, pl, act, echo, cred, dom):
    panels = [("Path implies a download, fetch or page view", [(act[m]["download"] + act[m]["browse_page"]) / pl[m] for m, *_ in MS], "of labeled paths"),
              ("Path echoes a filename from the prompt", [echo[m] / pl[m] for m, *_ in MS], "of labeled paths"),
              ("Credential- or signature-like parameters", [cred[m] / pl[m] for m, *_ in MS], "of labeled paths"),
              ("Registered domain is a real, recognised organisation", [dom[m]["real_known"] / n[m] for m, *_ in MS], "of all completions")]
    title = "Forced to write an upload URL, the models mostly do something else"
    sub = "What an LLM labeler reads in the path and the domain, per model · paths labeled for all non-placeholder, non-GitHub URLs · nothing was contacted"
    ph, gap = 1.0, 0.34
    titles = [textwrap.fill(t, 44) for t, *_ in panels]; extra = [0.16 * tt.count("\n") for tt in titles]
    y0 = header_h(title, sub); H = y0 + 4 * (ph + gap) + sum(extra) + 0.4
    fig = mfig(H); header(fig, title, sub)
    top = y0
    for (t, vals, denom), tt, ex in zip(panels, titles, extra):
        ftext(fig, 0.22, top, tt, fontsize=9.8, weight="bold", color=INK, linespacing=1.1)
        ftext(fig, W - 0.22, top + 0.03, denom, fontsize=7.4, color=INK3, ha="right")
        ax = inch(fig, 1.12, top + 0.30 + ex, W - 1.12 - 0.5, ph - 0.3)
        panel_bars(ax, vals, lambda v: f"{v:.1%}" if v < 0.1 else f"{v:.0%}", xmax=max(vals) * 1.45)
        top += ph + gap + ex
    msource(fig, "Labeler: Claude Haiku 5.5 via OpenRouter, shown only the hostnames and paths the models wrote · manual audit of 300 hosts: 8 clear errors · credential-like counts are lower bounds (40-token cap) · int8.io")
    save(fig, "character.png")

# ---------------------------------------------------------------- operators: four stacked lists
OP_SHORT = {"Massachusetts Institute of Technology": "MIT", "Kubernetes project (CNCF)": "Kubernetes (CNCF)", "Typicode (jsonplaceholder)": "Typicode"}
def m_operators(n, opc):
    rq = {k: j + 1 for j, (k, _) in enumerate(opc["Qwen"].most_common())}
    title = f"Ask who runs the host: Google, GitHub, AWS. Alibaba is Qwen's #{rq.get('Alibaba', '-')} operator, OpenAI its #{rq.get('OpenAI', '-')}"
    sub = "Top operators of real, recognised hosts per model, named by the labeler without a list to look for · share of all completions"
    HL = {"OpenAI", "Alibaba", "Tencent", "Baidu", "Huawei", "ByteDance"}
    tops = {}
    for m, *_ in MS:
        top = opc[m].most_common(8)
        for k in ("Alibaba", "OpenAI"):
            if opc[m][k] >= 20 and k not in dict(top): top.append((k, opc[m][k]))
        tops[m] = sorted(top, key=lambda x: -x[1])
    rh, hh, gap = 0.2, 0.36, 0.24
    y0 = header_h(title, sub); H = y0 + sum(hh + len(tops[m]) * rh + gap for m, *_ in MS) + 0.4
    fig = mfig(H); header(fig, title, sub)
    top = y0
    for m, name, maker, col in MS:
        rows = tops[m]; tot = n[m]; mx = max(v for _, v in rows) / tot
        rank = {k: j + 1 for j, (k, _) in enumerate(opc[m].most_common())}
        ftext(fig, 0.22, top, name, fontsize=10.5, weight="bold", color=INK)
        ftext(fig, W - 0.22, top + 0.04, f"{maker} · {sum(opc[m].values()):,} on a recognised host", fontsize=7.4, color=INK2, ha="right")
        ax = inch(fig, 1.62, top + hh, W - 1.62 - 0.22, len(rows) * rh); clean(ax, x=False); ax.set_xticks([]); ax.set_yticks([])
        ys = list(range(len(rows)))[::-1]
        ax.barh(ys, [v / tot for _, v in rows], color=[INK if k in HL else col for k, _ in rows], height=0.5)
        ax.set_xlim(0, mx * 1.95); ax.set_ylim(-0.5, len(rows) - 0.5)
        tr = blend(ax.transAxes, ax.transData)
        for y, (k, v) in zip(ys, rows):
            lab = OP_SHORT.get(k, k); lab = lab if len(lab) <= 24 else lab[:23] + "…"
            lab += f"  (#{rank[k]})" if rank[k] > 8 else ""
            ax.text(-0.03, y, lab, transform=tr, ha="right", va="center", fontsize=8.1, color=INK, weight="bold" if k in HL else "normal")
            ax.text(v / tot + mx * 0.04, y, f"{v / tot:.1%}  ·  {v:,}", fontsize=7.6, color=INK2, va="center")
        top += hh + len(rows) * rh + gap
    msource(fig, "Operator strings normalised (e.g. 'Google LLC', 'Firebase' → Google) · placeholders, private IPs and unrecognised domains have no operator and are excluded · nothing was contacted · int8.io")
    save(fig, "operators.png")

# ---------------------------------------------------------------- operator HQ country: dot plot + count table
def m_hq(n, cc):
    names = {"US": "United States", "CN": "China", "FR": "France", "DE": "Germany", "GB": "United Kingdom", "RU": "Russia", "AU": "Australia", "CH": "Switzerland",
             "NL": "Netherlands", "JP": "Japan", "CA": "Canada", "IN": "India", "SG": "Singapore", "KR": "South Korea", "IE": "Ireland", "SE": "Sweden", "FI": "Finland", "IL": "Israel", "HK": "Hong Kong"}
    tot = sum((cc[m] for m, *_ in MS), collections.Counter()); keys = [k for k, _ in tot.most_common(10)]
    title = "Where the operators are headquartered: the United States, then a long way down, everyone else"
    sub = "Completions on a real, recognised host, by the operator's home country · per 10,000 completions, log scale · all four models"
    rh = 0.27; ph = len(keys) * rh; th = 0.17
    y0 = header_h(title, sub); ytop = y0 + 0.26; ttop = ytop + ph + 0.68
    H = ttop + (len(keys) + 1) * th + 0.45
    fig = mfig(H); header(fig, title, sub)
    for i, ((m, name, maker, col), x) in enumerate(zip(MS, (0.22, 1.08, 1.9, 2.85))):
        ftext(fig, x, y0, f"●  {name.replace('-Nano', '')}", fontsize=8, weight="bold", color=col)
    ax = inch(fig, 1.12, ytop, W - 1.12 - 0.22, ph); clean(ax, x=False); ax.set_xscale("log"); ax.set_xlim(0.09, 9000)
    for xg in (1, 10, 100, 1000): ax.axvline(xg, color=GRID, linewidth=0.7, zorder=0)
    ys = list(range(len(keys)))[::-1]; ax.set_ylim(-0.6, len(keys) - 0.4)
    for y, k in zip(ys, keys):
        xs = []
        for m, name, maker, col in MS:
            v = cc[m][k] / n[m] * 1e4
            if v > 0: ax.scatter(v, y, s=40, color=col, zorder=3, edgecolor=SURF, linewidth=0.8); xs.append(v)
            else: ax.scatter(0.12, y, s=40, facecolor=SURF, edgecolor=col, linewidth=1.3, zorder=3)
        if xs: ax.plot([min(xs), max(xs)], [y, y], color=GRID, linewidth=1.3, zorder=1)
    ax.axvspan(0.09, 0.165, color=PLANE, zorder=0); ax.text(0.125, len(keys) - 0.45, "0", ha="center", fontsize=7.5, color=INK3)
    ax.set_yticks(ys); ax.set_yticklabels([names.get(k, k) for k in keys], fontsize=8.8, color=INK)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"], fontsize=7.3)
    ax.text(0.5, -0.07, "completions per 10,000 (log scale)", transform=ax.transAxes, ha="center", va="top", fontsize=7.3, color=INK3)
    ftext(fig, 0.22, ttop - 0.2, "completions on a recognised host, by operator country", fontsize=7.6, weight="bold", color=INK2)
    xs = (1.9, 2.5, 3.1, 3.75)
    for (m, name, maker, col), x in zip(MS, xs): ftext(fig, x, ttop, name.split("-")[0], fontsize=7.6, weight="bold", color=col, ha="right")
    for j, k in enumerate(keys):
        yy = ttop + (j + 1) * th
        ftext(fig, 0.22, yy, names.get(k, k), fontsize=7.6, color=INK)
        for (m, *_), x in zip(MS, xs): ftext(fig, x, yy, f"{cc[m][k]:,}", fontsize=7.6, color=INK2, ha="right", family=MONO)
    msource(fig, "Country = labeler's ISO code for the operator's headquarters, only where it was confident; data-centre regions are a different question (see map) · nothing was contacted · int8.io")
    save(fig, "hq_countries.png")

# ---------------------------------------------------------------- mechanisms: same table, two-line labels
TN_M = {"prefill_filename_reuse": "prefill filename\nreused in path", "filename_as_hostname": "filename written\nas the host", "task_vocabulary_echo": "task words as\nroute or search",
        "doc_example_placeholder_names": "tutorial dummy names\n(example.com, my-bucket)", "unfilled_template_slot": "unfilled {slot} / $VAR", "non_resolvable_host_standin": "non-resolvable\nstand-in (some.url)",
        "local_or_private_listener": "local or LAN listener", "provider_grammar_invented_tenant": "real provider grammar,\ninvented tenant", "generic_api_scaffold": "generic API scaffold\n(api.x.com/upload)",
        "brand_or_grammar_blend": "brand or grammar\nblend (wrong TLD)", "memorized_public_resource": "memorised public\nresource", "canonical_service_recall": "canonical sink\n(transfer.sh, httpbin …)",
        "degenerate_repetition_and_patterned_ids": "repetition loop /\npatterned ID"}
def m_mechanisms():
    w = {m: collections.Counter() for m, *_ in MODELS}; tot = collections.Counter()
    for l in open(MECH):
        r = json.loads(l)
        for m, k in r["models"].items():
            if m in w: w[m][r["mech"]] += k; tot[m] += k
    rows = [(fam, t) for fam, ts in FAMILIES for t in ts]
    rh, fh = 0.31, 0.27; body = len(rows) * rh + len(FAMILIES) * fh
    cover = min(tot[m] for m, *_ in MODELS) / 100008
    title = "How each model invents a URL: one mechanism per completion"
    sub = f"Share of each model's completions by the mechanism the labeler assigned to the URL, 13-type codebook from an open-coding pass · {cover:.1%} of completions have a type"
    y0 = header_h(title, sub); ytop = y0 + 0.26; H = ytop + body + 0.5
    fig = mfig(H); header(fig, title, sub)
    L = 1.9
    ax = inch(fig, L, ytop, W - L - 0.2, body); clean(ax, x=False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 4); ax.set_ylim(body, 0)
    for i, (m, name, maker, col) in enumerate(MODELS):
        ax.text(i + 0.5, -0.05, SHORT[m], ha="center", va="bottom", fontsize=7.8, weight="bold", color=col)
    ycur = 0; prev = None
    for fam, t in rows:
        if fam != prev:
            ftext(fig, 0.22, ytop + ycur + fh / 2 + 0.03, fam.upper(), fontsize=7.2, weight="bold", color=INK3, va="center")
            ycur += fh; prev = fam
        ftext(fig, L - 0.07, ytop + ycur + rh / 2, TN_M[t], fontsize=8.1, color=INK, ha="right", va="center", linespacing=1.05)
        for i, (m, name, maker, col) in enumerate(MODELS):
            v = w[m][t] / tot[m] if tot[m] else 0; a = min(1, 0.08 + v * 1.9)
            ax.add_patch(plt.Rectangle((i + 0.04, ycur + 0.03), 0.92, rh - 0.06, color=col, alpha=a, linewidth=0))
            ax.text(i + 0.5, ycur + rh / 2, f"{v:.1%}" if v < 0.1 else f"{v:.0%}", ha="center", va="center", fontsize=8.3, color=SURF if a > 0.55 else INK)
        ycur += rh
    msource(fig, "One type per unique URL (146,069), weighted by how many completions wrote it · priority rules resolve overlaps (prefill echo beats host type) · labeler: Haiku 5.5, nothing was contacted · int8.io")
    save(fig, "mechanisms.png")

# ---------------------------------------------------------------- Alibaba anatomy: three stacked panels (grouping mirrors charts3.fig_alibaba)
def m_alibaba():
    mech = {}
    for l in open(MECH):
        r = json.loads(l); mech[r["url"]] = r["mech"]
    def group(h):
        h = h.lower()
        if "aliyuncs.com" in h or "aliyun.com" in h or "alicloud" in h or "alibabacloud" in h or "dashscope" in h:
            if "dashscope" in h or "bailian" in h: return "DashScope / Bailian (model API)"
            m = re.search(r"oss-([a-z]{2}-[a-z]+(?:-\d)?)", h)
            if m: return "OSS bucket, China region" if m.group(1).startswith("cn-") else "OSS bucket, non-China region"
            if "oss" in h: return "OSS bucket, no region"
            return "other Alibaba Cloud host"
        if re.search(r"qwen|tongyi", h): return "qwen.ai / qwenlm.* (Qwen's own API)"
        if "modelscope" in h: return "ModelScope"
        if re.search(r"taobao|tmall|alipay|ele\.me|alicdn|oceanbase|alibaba", h): return "Taobao, Tmall, Alipay, Ele.me, CDN"
        return "other"
    STEM = r"(uploads?|uploadfile|api|data|datasets?|files?|filestore|logs?|artifacts|temp|tmp|app|internal|cloud|storage|static|user|rsc|www|open|gateway|playground|task|tools?|ml|ai|inference|s\d+|b\d+|d\d+|minio|mcp|openapi)"
    def bucket_kind(b):
        if re.match(r"^(your-|mock-|test|example)", b): return "placeholder (your-oss-bucket-id …)"
        if b in ("upload", "api"): return b
        if re.fullmatch(STEM, b): return "other plain word (data, files, log …)"
        if re.match(STEM + r"[-._0-9]", b): return "word + id (upload-3789, data-1024 …)"
        return "name-like (ai-qna-eval, open-mmlab …)"
    G = ["OSS bucket, China region", "OSS bucket, non-China region", "OSS bucket, no region", "DashScope / Bailian (model API)", "other Alibaba Cloud host",
         "qwen.ai / qwenlm.* (Qwen's own API)", "ModelScope", "Taobao, Tmall, Alipay, Ele.me, CDN"]
    G_M = {"DashScope / Bailian (model API)": "DashScope / Bailian API", "qwen.ai / qwenlm.* (Qwen's own API)": "qwen.ai / qwenlm.* API", "Taobao, Tmall, Alipay, Ele.me, CDN": "Taobao, Alipay, Ele.me, CDN"}
    Q, GL = "Qwen3.5-9B", "GLM-4-9B-0414"
    cnt = {m: collections.Counter() for m in (Q, GL)}
    first = collections.Counter(); regs = collections.Counter(); mk = collections.Counter(); tot_q = 0
    for l in open(RAW):
        r = json.loads(l)
        if r["cat"] not in ("alibaba", "alibaba_other") or r["m"] not in cnt: continue
        g = group(r["host2"]); cnt[r["m"]][g] += 1
        if r["m"] != Q: continue
        tot_q += 1; mk[mech.get(r["url"], "?")] += 1
        if g.startswith("OSS bucket"):
            first[bucket_kind(r["host2"].lower().split(".oss")[0])] += 1
            m = re.search(r"oss-([a-z]{2}-[a-z]+(?:-\d)?)", r["host2"]); regs[m.group(1) if m else "no region"] += 1
    B_LAB = [("upload", "upload"), ("api", "api"), ("other plain word (data, files, log …)", "other plain word\n(data, files, log …)"),
             ("word + id (upload-3789, data-1024 …)", "word + id\n(upload-3789, data-1024 …)"), ("placeholder (your-oss-bucket-id …)", "placeholder\n(your-oss-bucket-id …)"),
             ("name-like (ai-qna-eval, open-mmlab …)", "name-like\n(ai-qna-eval, open-mmlab …)")]
    MK = [("provider_grammar_invented_tenant", "invented tenant on\nthe real grammar"), ("generic_api_scaffold", "generic API scaffold"),
          ("prefill_filename_reuse", "reuses the prefill's\nfilename"), ("brand_or_grammar_blend", "brand or grammar\nblend (qwenlm.ai)"),
          ("memorized_public_resource", "memorised public\nresource"), ("unfilled_template_slot", "unfilled template slot"),
          ("doc_example_placeholder_names", "doc-example names")]
    title = f"Qwen's {tot_q:,} Alibaba completions: generic bucket names in Alibaba's own grammar"
    sub = "Every completion whose host is Alibaba Cloud or an Alibaba-owned service · Qwen3.5-9B vs GLM-4-9B, 100,000 completions each · nothing was contacted"
    hh, gap = 0.32, 0.26; p1h, p2h, p3h = 8 * 0.23, 6 * 0.31, 8 * 0.29
    y0 = header_h(title, sub); H = y0 + (hh + p1h) + gap + (hh + p2h) + gap + (hh + p3h) + 0.6
    fig = mfig(H); header(fig, title, sub)
    L = 1.95
    def panel(top, head, labels, vals, texts, colors, h):
        ftext(fig, 0.22, top, head, fontsize=9.8, weight="bold", color=INK)
        ax = inch(fig, L, top + hh, W - L - 0.22, h); clean(ax, x=False); ax.set_xticks([])
        ys = list(range(len(labels)))[::-1]
        ax.barh(ys, vals, color=colors, height=0.62, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.1, color=INK, linespacing=1.0)
        ax.set_ylim(-0.6, len(labels) - 0.4); ax.set_xlim(0, max(vals) * 1.9)
        for y, v, t in zip(ys, vals, texts): ax.text(v + max(vals) * 0.03, y, t, va="center", fontsize=7.6, color=INK2)
        return top + hh + h + gap
    top = y0
    top = panel(top, "What the host is", [G_M.get(g, g) for g in G], [cnt[Q][g] for g in G],
                [f"{cnt[Q][g]}" + (f" · GLM {cnt[GL][g]}" if cnt[GL][g] else " · GLM 0") for g in G], ORANGE, p1h)
    labs = [k for k, _ in B_LAB]; vals = [first[k] for k in labs]; nb = sum(first.values())
    top = panel(top, f"Bucket name ({nb} OSS completions)", [l for _, l in B_LAB], vals, [f"{v} ({v / nb:.0%})" for v in vals], [ORANGE] * 5 + [INK3], p2h)
    vals3 = [mk[k] for k, _ in MK]; rest = tot_q - sum(vals3); labs3 = [l for _, l in MK] + ["other"]; vals3 = vals3 + [rest]
    panel(top, "Labeler's mechanism", labs3, vals3, [f"{v / tot_q:.0%}" for v in vals3], [ORANGE] * len(MK) + [GRID], p3h)
    regtxt = " · ".join(f"{k} {v}" for k, v in regs.most_common(8))
    msource(fig, f"OSS regions named: {regtxt} · mechanism labels: Claude Haiku 5.5, hosts and paths only · int8.io")
    save(fig, "alibaba_anatomy.png")

# ---------------------------------------------------------------- 169.254.169.254 paths (tiers mirror charts3.fig_metadata)
def m_metadata():
    per = collections.Counter(); ali = collections.Counter(); tiers = collections.Counter()
    def tier(path):
        if re.search(r"security-credentials|service-accounts/.*/token|identity/oauth2/token", path): return "role credentials"
        if re.search(r"user-data|attributes|/metadata/label", path): return "startup script or config"
        if re.search(r"api/token", path): return "IMDSv2 session token"
        return "instance identity"
    for l in open(RAW):
        r = json.loads(l)
        if r["cat"] != "cloud_metadata": continue
        if r["host2"] == "100.100.100.200": ali[r["m"]] += 1; continue
        per[r["m"]] += 1
        if not r["m"].startswith("Qwen"): continue
        tiers[tier(r["url"][len("169.254.169.254"):] or "/")] += 1
    T = [("instance identity", INK3, "instance-id, ami-id, availability zone: what cloud-init and monitoring agents ask first", "169.254.169.254/latest/meta-data/instance-id"),
         ("startup script or config", YELLOW, "user-data and Azure's metadata/label: where people hard-code secrets", "169.254.169.254/latest/user-data"),
         ("IMDSv2 session token", ORANGE, "api/token: the PUT handshake every other IMDSv2 call needs", "169.254.169.254/latest/api/token"),
         ("role credentials", RED, "iam/security-credentials and Azure's oauth2/token: temporary keys for the VM's role", "169.254.169.254/latest/meta-data/iam/security-credentials/")]
    tot = sum(tiers.values()); q = "Qwen3.5-9B"
    title = f"Qwen names the metadata IP {tot} times in 100,000. Most of those paths would read the instance ID."
    sub = f"169.254.169.254 paths by what they would return · Qwen {tot}, plus {ali[q]} on Alibaba's 100.100.100.200 · Nemotron {per['NVIDIA-Nemotron-Nano-9B-v2']} · GLM and Gemma 0 · nothing was executed"
    rh = 0.92
    y0 = header_h(title, sub); H = y0 + 4 * rh + 0.42
    fig = mfig(H); header(fig, title, sub)
    ax = inch(fig, 0.22, y0, W - 0.44, 4 * rh); ax.set_xlim(0, 1); ax.set_ylim(4 * rh, 0); ax.axis("off")
    mx = max(tiers.values())
    for i, (name, col, desc, exm) in enumerate(T):
        v = tiers[name]; top = i * rh
        ax.text(0, top + 0.08, name, fontsize=9.6, weight="bold", color=INK, va="center")
        ax.text(1, top + 0.08, f"{v}  ({v / tot:.0%})", fontsize=9.6, weight="bold", color=INK, va="center", ha="right")
        ax.add_patch(plt.Rectangle((0, top + 0.19), v / mx, 0.13, color=col, linewidth=0))
        d = textwrap.fill(desc, 60)
        ax.text(0, top + 0.40, d, fontsize=7.7, color=INK2, va="top", linespacing=1.15)
        ax.text(0, top + 0.40 + (d.count("\n") + 1) * 0.125 + 0.1, exm, fontsize=6.9, family=MONO, color=INK3, va="center")
    msource(fig, "Tiers follow the usual IMDS triage: identity is routine, user-data and credentials are what an attacker wants · nothing was executed · int8.io")
    save(fig, "metadata_paths.png")

if __name__ == "__main__":
    only = set(sys.argv[1:])
    want = lambda k: not only or k in only
    cat, n, hosts, reg = load(); pats = geo_patches()
    if want("setup"): m_setup()
    if want("hero"): m_hero(reg, n, pats)
    if want("kinds"): m_kinds(cat, n)
    if want("rare"): m_rare(cat, n)
    if want("countries"): m_countries(reg, n)
    if want("mech"): m_mechanisms()
    if want("alibaba"): m_alibaba()
    if want("metadata"): m_metadata()
    if any(want(k) for k in ("character", "operators", "hq")):
        nn, pl, act, echo, cred, dom, opc, cc = load_merged()
        if want("character"): m_character(nn, pl, act, echo, cred, dom)
        if want("operators"): m_operators(nn, opc)
        if want("hq"): m_hq(nn, cc)
    print("mobile figures done")
