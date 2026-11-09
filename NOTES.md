# Working notes

Where things stand, so work can resume later. Last updated 2026-11-08.

## What exists

- `wordvec` (English, GloVe), `wordvec-zh` (Chinese: Tencent vectors + CC-CEDICT + jieba),
  `wordvec-web` (Flask page serving both at `/en/` and `/zh/`).
- Web app is dockerized. Run locally:
  `docker compose -f docker-compose.local.yml up -d --build`, then http://localhost:8000/en/
  (no login locally). `docker-compose.yml` is the homelab version (gateway network, no host port).
- Branches `main` and `chinese` are kept identical (fast-forward merge, both pushed).

## English `build` walkthrough (reworked, 8 steps)

Page title is "How AI works - part 1"; `build` header is "HOW WE TRAIN A MODEL".

1. sentences  2. counts (worked sentence, corner of the table, log squash)
3. layers: SVD on a hand-checkable 3x3 table (schematic with Add layer 2 button)
4. strengths: where 5.21 / 3.33 come from (36% + 14% = 50% of the table), slider figure
5. descent: gradient descent on one pair (wagged/tail, target 1.39, step 0.2), animation
6. learning: gradient descent on all 19 words, animation with three step sizes
7. compress  8. plot. `full` is now 17 steps.

Pictures: a step calls `wordvec.figure(spec)`; `web.py` collects specs per request and sends
them as `figures`; `static/figures.js` draws them (SVG, no innerHTML). Terminal ignores them.

### Gradient descent tuning (measured on the toy table)

- Update: `W -= step * (E @ W)`, `E = W @ W.T - log1p(counts)`, 2 numbers per word, seed 0,
  init normal(0, 0.3).
- Step 0.08 converges to SVD's optimum (error 38.3) in ~60 rounds; 0.01 is still at 43 after 120;
  0.25 bounces forever (~50). Clustering matches SVD (within-topic cosine 0.96, between 0.14).
- The first try (400 rounds, tiny step) just hadn't converged, which is why clustering looked weak.
- SVD vs GD claim used on screen: SVD is exact but slow/huge at 400,000 x 400,000; GD walks to
  nearly the same answer a few pairs at a time. GD result equals SVD's up to rotation/flip.

## To do next

- **Chinese `build` is still the old 5-step version.** Port the new 8 steps, the worked count
  example (use a real sentence from the Chinese corpus), and both figures; the figure plumbing
  already works for any language (`chinese.execute` just needs to call `wordvec.figure`).
  Chinese page title/banner not renamed yet either.
- Check the new figures at phone width (only checked at ~1440px).
- Possible polish: figures sit under the step text, so you scroll to reach them; the animations
  autostart only when scrolled into view.
- Deploy to homelab when happy: rsync repo to `/opt/stacks/wordvec/`, then
  `docker compose up -d --build` there (see earlier deployment: gateway nginx block, Authelia
  one_factor rule, idle-gate managed, landing-page card with English / 中文 buttons).
- The homelab-gateway repo changes (`docker-compose.yml`, `landing/`, `nginx/conf.d/gateway.conf`)
  were never committed there; Authelia config is untracked/ignored. Backups are `*.bak-wordvec-20261108`.
- Optional: People's Daily vectors via `--vectors` (Baidu Pan download, untested).
