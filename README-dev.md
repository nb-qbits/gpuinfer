# gpu.calc — Developer Guide

LLM inference GPU sizing calculator. Single HTML file, no build step, no dependencies.

## Quick Start

```bash
git clone https://github.com/nb-qbits/gpuinfer
cd gpuinfer
open index.html   # that's it — no build needed
```

## VSCode Setup

Install extensions: Live Server, ESLint, Auto Close Tag.
Right-click html file → Open with Live Server → edits auto-reload.

## Project Structure

```
.
├── gpu-calc-final.html   ← entire app (~3800 lines)
├── gpu-calc-tests.js     ← test suite (node gpu-calc-tests.js)
├── SKILL.md              ← Claude AI context — read this first
├── README-dev.md         ← this file
└── hf-proxy-worker.js    ← Cloudflare Worker for HF API proxy
```

## Running Tests

```bash
node gpu-calc-tests.js
# Expected: 60/60 passed · ALL PASS ✓
```

Run before every commit. 60 tests cover all formulas, precision, GPU models, edge cases.

## Key Formulas

```
wt       = model × wp              (wp: 2=BF16, 1=FP8, 0.5=INT4)
usable   = gpu × (1 - overhead%)
aPA      = (usable × TP - wt) × 0.95
kvBpt    = 2 × heads × dim × layers × kvBytes × 1000 / 1e9   [GB/K-tok = MB/tok]
kvB      = kvBpt × ctx
upr      = floor(aPA / kvB)
t1       = ceil(users / upr) × TP
```

## Common Mistakes

1. **S.wp (weight precision)** — most common bug source. Always let `applyModelConfig` set it from dtype. Never hardcode `S.wp=2` in new paths.

2. **Intermediate renders** — `setU()`/`setC()` both call `renderResult()`. Sync `S.wp` chips BEFORE calling them, or use silent DOM assignment + one final `renderResult()`.

3. **MB/tok** — `kvBpt` is already in MB/tok numerically. Do NOT `× 1000`.

4. **HTML in JS strings** — escape `<` as `<` inside JS string literals.

## Deploying

```bash
cp gpu-calc-final.html /path/to/nb-qbits/gpuinfer/index.html
git add . && git commit -m "gpu.calc vX.Y" && git push
# GitHub Pages: Settings → Pages → main / root
# URL: https://nb-qbits.github.io/gpuinfer
```

## Contact

Vikas Grover · linkedin.com/in/vgrover1515 · huggingface.co/vikasgrover2004
