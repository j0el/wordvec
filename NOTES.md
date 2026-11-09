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

## Golf detour (added 2026-11-09)

`golf` is a 6-slide walkthrough (button "Golf detour"; `golf 5` is the race), shared by the English
and Chinese pages and linked from build steps 3 and 5. Steps and data are in `wordvec/__init__.py`
(`GOLF_STEPS`). Table: holes x players (Phil, Tiger, Vijay), scores 4/5/3, then Tiger birdies hole 2.
- svd: one layer, strength sqrt(150) = 12.25, holes (.57 .71 .42), players (.58 x3).
- birdie: strengths 11.86 and 0.60 (layer 1 holds 99.7%); squared strengths sum to 141.
- guess/race: gradient descent on hole number x player number, start 1.0, step 0.01, 40 rounds;
  settles at error 0.36 = 0.60^2, the same as SVD's best one-layer answer.
- Figures: `svd-mini` (now takes vmax/names/title/summary; single layer hides the button) and new
  `golf-race` (SVD grid beside animated GD grid + log error curve).
- The first source text for the example had wrong numbers (0.33/0.41/0.25, 21.07); the correct
  ones are above.

## Chinese parity (done 2026-11-09)

Chinese `build` now matches English: same 8 steps and the same figures, using the Chinese
table (17 words, strengths 5.05 / 3.17, 53% in the top two layers, SVD best error 31.8).
Shared helpers (`oriented_svd`, `descend`, `descend_pair`, `descent_spec`, `figure`) live in
`wordvec/__init__.py`. Chinese gradient descent step sizes are 0.005 / 0.08 / 0.3 (the English
table uses 0.01 / 0.08 / 0.25); 0.4 diverges on the Chinese table. Titles and banners match
("How AI works - part 1 (中文)"). Figure labels show the word plus its English gloss.

## Deployed

Homelab `/opt/stacks/wordvec` runs the current code (idle-gate stops it after 30 min idle and
wakes it on request). The login was removed by the owner (policy `bypass`) and the landing card is public.
The golf detour is NOT yet deployed to the homelab.

## To do next

- Check the new figures at phone width (only checked at ~1440px).
- Possible polish: figures sit under the step text, so you scroll to reach them; the animations
  autostart only when scrolled into view.
- The homelab-gateway repo changes (`docker-compose.yml`, `landing/`, `nginx/conf.d/gateway.conf`)
  were never committed there; Authelia config is untracked/ignored. Backups are `*.bak-wordvec-20261108`.
- Optional: People's Daily vectors via `--vectors` (Baidu Pan download, untested).
