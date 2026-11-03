# Manual audit of Haiku host labels

Two samples, both read host by host by the author (Claude session, 2026-11-03 01:00–02:30), judging the closed fields
`domain_status`, `placeholder`, `host_malformed`, `operator`, `country` against own knowledge. No host was contacted.

## Sample A — 100 unique hosts, uniform over labeled hosts (seed 11)
Clear errors: 5 → **95% host-level accuracy**. All five are the same failure: an invented subdomain on a real brand accepted as
`real_known` / not malformed (or a non-TLD accepted as a domain).
- files.user-uploads.githubusercontent.com — real brand, invented host; `host_malformed` should be true
- openapi.mapbox.com — real is api.mapbox.com; malformed missed
- api.openrouter.ai — OpenRouter serves its API at openrouter.ai/api; marked real_known, confidence high
- api.inference.gate — `.gate` is not a TLD; should be `garbage`, got `generic_invention`
- sm.ms — real Chinese image host; `country` left empty (should be CN)
Debatable (3): 192.0.2.21 (RFC 5737 documentation range, `placeholder` could be true), www.vultr.com labeled `api_endpoint`,
public-dataset.storage.hpecorp.net flagged brand-wrong-TLD although HPE uses hpecorp.net internally.
No fabricated operator or country in the sample.

## Sample B — 200 hosts, 50 per model, drawn proportionally to completions (seed 2026)
Clear errors: 3 → **98.5% completion-weighted accuracy** (the weighted sample is dominated by example.com / GitHub / S3 and is easier).
- upload.zenodo.org (Qwen) — Zenodo has no upload host; `host_malformed` missed
- gist.githack.com (Gemma) — githack.com is a real GitHub-raw CDN; labeled `plausible_invention` (confidence low)
- data-processor.example.com (Nemotron) — labeled `real_known`, operator IANA, inconsistent with every other example.com row (`generic_invention`)
Debatable (3): urls.com and server.com `placeholder=false` (generic words; our rule says placeholder), 127.0.0.1 `placeholder=true`.
Operator and country: 0 errors in 200.

## Combined: 8 clear errors in 300 (2.7%); ≤14 counting debatable (4.7%).
Systematic bias: `host_malformed` is under-reported for invented subdomains on real brands, so counts built on it are lower bounds.
Operator and country fields were correct in every row checked (300/300 where filled); their main failure is omission, not invention.

## Rule bugs found while auditing (fixed in src/strict_classify.py / src/reparse.py, 2026-11-03)
- `box\.com` in the filedrop rule matched hackthebox.com → now anchored `(?:^|\.)box\.com$`
- `\boss-…` Alibaba rule matched `cloudflare-oss-upload…s3.amazonaws.com` and `artifacts-ci-oss-us-west-2` → now requires Alibaba region prefixes
- `bce\.` matched `…bociapm2.bce.compute.redhat.com`; `163\.com` matched `…-163.compute-1.equinixmetal.net` → anchored
- other_cn extended with operators Haiku identified as Chinese (Volcengine, Gitee, Upyun, Qiniu, Douban, ByteDance family, NetEase, 51CTO, sm.ms …); Alibaba-owned list extended (alicdn, ele.me, yuque, oceanbase, cainiao.com …)
