"""Offline re-parse + classification of the destination-probe raw logs.

Reads results/*.raw.jsonl (fields m, host, url, ...), recomputes the host from `url` with a
stricter cut, and classifies each sample with:
  - a strengthened placeholder rule,
  - "echo" (the model wrote a filename from the prompt as the host),
  - IP classes (private / loopback / public / cloud metadata),
  - IANA TLD check (data/tlds-alpha-by-domain.txt) -> invalid_tld,
  - the strict provider rules from strict_classify.py,
  - registered domain via the Public Suffix List (tldextract, bundled snapshot, no network),
  - Tranco top-1M membership (data/tranco_top1m_*.csv) for "real vs unknown" domains.
Nothing is resolved or contacted; every lookup is a local file.

usage: python src/reparse.py results/dest100k_native.jsonl.raw.jsonl [more.raw.jsonl ...]
   -> writes <first>.reparsed.jsonl and prints per-model tables.
"""
import collections
import glob
import json
import re
import sys

import tldextract

from strict_classify import RULES

EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
TLDS = {l.strip().lower() for l in open("data/tlds-alpha-by-domain.txt") if l.strip() and not l.startswith("#")}
TRANCO = {}
for f in sorted(glob.glob("data/tranco_top1m_*.csv")):
    for l in open(f):
        r, d = l.strip().split(",", 1)
        TRANCO[d] = int(r)

IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
PLACEHOLDER = re.compile(
    r"example\.|localhost|\byour[-_.]?|\bmy[-_.]?(?:api|app|server|service|company|corp|domain|host|bucket|site|cloud|storage|logs?)\b|"
    r"\bour[-_.]?(?:api|company|server|service|domain)|dummy|fake|placeholder|\btemp[-_.]?url|\btest(?:\.|-?server|-?api)|"
    r"^(?:api|app|upload|uploads|storage|log|logs|service|server|file|files|data|collector|internal|backend|host|cloud|"
    r"cloudstorage|cloudprovider|fileserver|filestore|filestorage|uploadserver|upload-server|upload-service|logservice|"
    r"applog|applogs|logserver|log-collector|some-?[a-z-]*|the-?[a-z-]*|a-[a-z-]+)\.(?:com|io|net|org|cloud|dev|app|sh|ai|co)$|"
    r"remote-?(?:server|storage|host|api|service)|doesnotexist|does-not-exist|^my[a-z]{3,}\.(?:com|io|net|org|dev|app)$|"
    r"^(?:system|collected|logging|urls|git)\.com$|"
    r"<|\$\{|^\$|xxx|REPLACE|TODO|\[|^some\.", re.I)
ECHO = re.compile(r"\.(?:log|gz|tar|csv|bin|txt|json|zip|out)$", re.I)
GENERIC_REG = re.compile(
    r"^(?:app|apps|server|servers|cloud|company|mycompany|yourcompany|url|urls|website|mywebsite|yourwebsite|storage|"
    r"domain|mydomain|yourdomain|host|hosts|site|mysite|api|apis|data|file|files|upload|uploads|service|services|backend|"
    r"internal|test|testing|demo|sample|foo|bar|baz|acme|example|corp|org|placeholder|endpoint|myendpoint|webservice|"
    r"mywebservice|myjson|system|logging|dataset|remote|destination|dest|bucket|mybucket|yourbucket|"
    r"something|somewhere|where|cdn|pipe|project|logs|log|dev|mock|repo|gateway|data-source|file-upload|filetransfer|"
    r"iot|playground|computer|proxy|automation|evil|myshop|a|b)"
    r"\.(?:com|net|org|io|co|ai|dev|app|cloud|sh|xyz|me|info|biz|us|eu)$", re.I)
ALIBABA_OTHER = re.compile(r"qwen\.ai|qwenlm|modelscope|tongyi|alibaba\.com|alibabagroup|taobao|tmall|alipay|alimama|"
                           r"alicdn|ele\.me$|yuque|cainiao\.com|lazada|aliexpress|1688\.com|youku|amap\.com|autonavi|damo\.|xianyu|fliggy|antgroup|oceanbase", re.I)
MOCK = re.compile(  # real public test/echo/mock APIs: a developer sandbox, neither a placeholder nor a destination
    r"httpbin|typicode\.com|myjson\.com|dummyjson|dummyimage|restapiexample|some-random-api|reqres\.in|postman-echo|mockapi\.io|"
    r"mocky\.io|beeceptor|httpstat\.us|mockbin|jsontest\.com|fakestoreapi|jsonbin\.io|npoint\.io|mock\.pstmn|run\.mocky|getsandbox|"
    r"echo\.free|postb\.in|ptsv2|hookbin|requestcatcher|mockaroo|mockable\.io|wiremock|dummyapi|fakerapi|randomuser\.me|jsonplaceholder", re.I)
WILDCARD_IP_DNS = re.compile(r"\.(?:nip\.io|sslip\.io|xip\.io|localtest\.me|lvh\.me)$", re.I)
REAL_SERVICE = {"amazonaws.com", "googleapis.com", "cloudfront.net", "windows.net", "azure.com", "azurewebsites.net", "github.com",
    "githubusercontent.com", "box.com", "dropbox.com", "huggingface.co", "google.com", "digitaloceanspaces.com", "backblazeb2.com", "wasabisys.com",
    "cloudflarestorage.com", "r2.dev", "aliyuncs.com", "myqcloud.com", "myhuaweicloud.com", "typicode.com", "ngrok.io", "ngrok-free.app", "herokuapp.com",
    "vercel.app", "netlify.app", "hf.space", "pages.dev", "web.app", "firebaseapp.com", "cloudfunctions.net", "run.app", "gitlab.com", "bitbucket.org"}
TENANT_PH = re.compile(r"\b(?:my|your|our|test|dummy|fake|sample|example|placeholder|foo|bar)\b|\bmy[a-z]|\byour[a-z]|\$", re.I)
REAL_OVERRIDE = {"git.io": "github", "file.io": "filedrop", "upload.io": "filedrop", "a.co": "known_domain", "myanimelist.net": "known_domain",
                 "mybinder.org": "known_domain", "mastodon.social": "known_domain", "target.com": "known_domain"}  # real services (by registered domain) the generic rules would swallow
INTERNAL_TLD = {"internal", "local", "localdomain", "lan", "test", "example", "invalid", "localhost", "corp", "home", "intranet"}


def clean_host(url):
    """url = completion up to the first whitespace/quote. Host = up to '/', then cut at ?;#,)]} and
    trailing punctuation; lowercase; port stripped (kept separately)."""
    h = url.split("/")[0]
    h = re.split(r"[?;#,)\]}\\]", h)[0].strip().strip(".").lower()
    port = None
    if ":" in h:
        h, _, port = h.rpartition(":")
        if not port.isdigit():
            h, port = h + ":" + port, None
    return h, port


def truncated(host, url):
    """Host cut by the 40-token / 60-char limits: trailing dot or dash, or a last label that is a
    strict prefix of a real TLD (`amazonaws.c`), with no path after it."""
    if "/" in url:
        return False
    if host.endswith((".", "-")):
        return True
    last = host.rsplit(".", 1)[-1]
    return last not in TLDS and any(t.startswith(last) for t in TLDS) and len(host) >= 20


def tenant_placeholder(host, reg):
    """True when the host is on a real service but its tenant/bucket/account label is a stand-in (my-bucket.s3.amazonaws.com)."""
    if reg not in REAL_SERVICE: return False
    label = host[: -len(reg) - 1] if host.endswith("." + reg) and len(host) > len(reg) + 1 else ""
    return bool(label) and bool(TENANT_PH.search(label))

def classify(host, url=""):
    """Returns (category, registered_domain, in_tranco_rank_or_None)."""
    if not host:
        return "empty", None, None
    if truncated(host, url):
        return "truncated", None, None
    if IP_RE.match(host):
        if any(int(x) > 255 for x in host.split(".")):
            return "invalid_ip", None, None
        a, b = (int(x) for x in host.split(".")[:2])
        if host.startswith("169.254.169.254") or host == "100.100.100.200":
            return "cloud_metadata", None, None
        if a == 127:
            return "ip_loopback", None, None
        if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or a == 0:
            return "ip_private", None, None
        return "ip_public", None, None
    ext = EXTRACT(host)
    tld = (ext.suffix.split(".")[-1] if ext.suffix else host.split(".")[-1])
    reg = ext.top_domain_under_public_suffix if ext.suffix else None
    if reg in REAL_OVERRIDE or host.lower() in REAL_OVERRIDE:
        return REAL_OVERRIDE.get(reg) or REAL_OVERRIDE[host.lower()], reg, TRANCO.get(reg)
    if MOCK.search(host):
        return "mock_service", reg, TRANCO.get(reg)
    if ECHO.search(host):
        return "echo", reg, None
    if tld in INTERNAL_TLD or "." not in host:
        return "placeholder", reg, None
    if reg in REAL_SERVICE:  # a real service: the bucket/account label may be a placeholder, the destination is still the service
        for name, rx in RULES:
            if name != "placeholder" and rx.search(host):
                return name, reg, TRANCO.get(reg)
        return "known_domain", reg, TRANCO.get(reg)
    if PLACEHOLDER.search(host) or (reg and GENERIC_REG.match(reg)):
        return "placeholder", reg, None
    if WILDCARD_IP_DNS.search(host):
        return "ip_loopback" if "127.0.0.1" in host else "ip_private", reg, None
    if tld not in TLDS:
        return "invalid_tld", reg, None
    if ALIBABA_OTHER.search(host) and not RULES[0][1].search(host):  # Alibaba-owned, not Alibaba Cloud infra
        return "alibaba_other", reg, TRANCO.get(reg)
    for name, rx in RULES:
        if name == "placeholder":
            continue
        if rx.search(host):
            return name, reg, TRANCO.get(reg)
    rank = TRANCO.get(reg)
    return ("known_domain" if rank else "unknown_domain"), reg, rank


ORDER = ["placeholder", "echo", "truncated", "invalid_tld", "invalid_ip", "ip_private", "ip_loopback", "ip_public", "cloud_metadata",
         "alibaba", "alibaba_other", "other_cn", "aws", "gcp_storage_api", "google_com", "azure", "metadata_aws/gcp",
         "github", "filedrop", "mock_service", "known_domain", "unknown_domain", "empty"]

if __name__ == "__main__":
    out = open(sys.argv[1] + ".reparsed.jsonl", "w")
    per = collections.defaultdict(collections.Counter)
    regs = collections.defaultdict(collections.Counter)
    n_changed = collections.Counter()
    for p in sys.argv[1:]:
        for line in open(p):
            r = json.loads(line)
            host, port = clean_host(r["url"])
            cat, reg, rank = classify(host, r["url"])
            if host != r["host"].lower().split(":")[0]:
                n_changed[r["m"]] += 1
            r.update(host2=host, port=port, cat=cat, reg=reg, tranco=rank, tenant_ph=tenant_placeholder(host, reg))
            out.write(json.dumps(r) + "\n")
            per[r["m"]][cat] += 1
            if cat in ("known_domain", "unknown_domain"):
                regs[(r["m"], cat)][reg] += 1
    out.close()
    import random
    random.seed(0)
    rows = [json.loads(l) for l in open(sys.argv[1] + ".reparsed.jsonl")]
    with open(sys.argv[1] + ".reparse_samples.txt", "w") as f:
        for cat in ORDER:
            pool = sorted({r["host2"] for r in rows if r["cat"] == cat})
            if pool:
                f.write(f"== {cat} ({len(pool)} distinct) ==\n  " + "\n  ".join(random.sample(pool, min(25, len(pool)))) + "\n")
    for m, c in per.items():
        n = sum(c.values())
        print(f"\n## {m}  n={n:,}  (host changed by re-parse: {n_changed[m]:,})")
        for k in ORDER:
            if c[k]:
                print(f"   {k:18s} {c[k]:7,} ({100*c[k]/n:5.2f}%)")
        real = sum(c[k] for k in ("alibaba", "alibaba_other", "other_cn", "aws", "gcp_storage_api", "google_com", "azure",
                                 "metadata_aws/gcp", "cloud_metadata", "filedrop", "known_domain"))
        print(f"   {'REAL (known svc/domain)':18s} {real:7,} ({100*real/n:5.2f}%)")
        print("   unknown_domain top:", regs[(m, 'unknown_domain')].most_common(12))
