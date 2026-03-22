# gpu.calc — LLM Inference GPU Sizing Calculator
## SKILL.md for Claude (AI assistant context)

This file gives Claude instant context to work on the gpu.calc codebase without re-reading thousands of lines.

---

## Project Identity

- **Tool:** gpu.calc — LLM inference GPU sizing calculator
- **Author:** Vikas Grover (linkedin.com/in/vgrover1515 · huggingface.co/vikasgrover2004)
- **File:** `gpu-calc-final.html` — single self-contained HTML/CSS/JS file (~3800 lines)
- **Current version:** v5.9 (footer: `real-time inference · v5.9`)
- **Live site:** https://nb-qbits.github.io/gpuinfer (deploy as `index.html`)
- **GitHub:** https://github.com/nb-qbits/gpu-calc
- **HF Proxy Worker:** https://billowing-scene-eb2c.vikasgrover2004.workers.dev

---

## Architecture

### Single-file structure
```
gpu-calc-final.html
├── <style>          CSS (variables, components, print media)
├── <nav>            Navigation tabs
├── <div#page-home>  Landing page + author story + WIIFY cards
├── <div#page-wizard> Quick Size wizard (5 steps)
├── <div#page-calc>  Advanced calculator
├── <div#page-explorer> GPU Explorer (bubble chart only, no pricing table)
├── <div#page-saved> Saved Results comparison table
├── Modals           Validate with AI, vLLM config, LinkedIn card, About drawer
└── <script>         All JS (~2000 lines)
```

### Global State Object S (single source of truth)
```js
var S = {
  // Model
  model: 8,           // params in billions
  modelLabel: '8B',
  wp: 2,              // weight bytes/param: 2=BF16, 1=FP8/INT8, 0.5=INT4
  kv: 2,              // KV cache bytes: 2=FP16, 1=FP8
  kvLabel: 'fp16',
  attn: 'gqa',        // 'gqa' or 'mha'
  numKvHeads: null,   // exact from HF (null = use approximation)
  headDim: null,
  numLayers: null,
  hfId: null,         // HuggingFace model ID if loaded

  // Hardware
  gpu: 80,            // VRAM in GB
  gpuName: 'H100',
  overhead: 8,        // framework overhead %

  // Workload
  users: 30,
  ctx: 8,             // context in K tokens
  hitrate: 0.1,       // prefix cache hit rate

  // Cost
  onpremYr: 25000,    // purchase price per GPU
  awsHr: 3.10,        // cloud $/hr per GPU
  amortYrs: 5,
  cloudHike: 3,       // annual cloud price increase %
};
```

**CRITICAL:** `S.wp` is set from 4 places — always verify it's correct after any state change:
1. `applyModelConfig()` — sets from dtype (int8→1, bfloat16→2)
2. `apc('wp', v, chip)` — user clicks BF16/FP8 chip
3. `wizPickPrec(bytes, lbl)` — wizard precision step
4. `manualModel(v)` — slider drag (only resets to 2 if `S.hfId` was set)

---

## Key Functions

### Core Math (pure, no DOM)
```js
nP2(n)              // next power of 2
kvBytesPerToken()   // GB per K-tok (uses exact arch if available, else approx)
kvFP8BytesPerToken()// same but always 1 byte KV
calcTiers()         // MAIN ENGINE — returns {t1,t2,t3,t4,tpMin,wt,aPA,upr,...}
```

### calcTiers() return object
```js
{
  t1: 8,           // baseline GPU count
  t2: 8,           // with prefix cache
  t3: 4,           // with FP8 KV
  t4: 4,           // with llm-d disaggregation
  tpMin: 2,        // optimal tensor parallel degree
  tpWeights: 1,    // minimum TP just to load weights
  wt: 16,          // weight memory GB (= S.model * S.wp)
  ohGB: 6.4,       // overhead GB
  usable: 73.6,    // usable VRAM per GPU
  aPA: 109.8,      // available for KV pool (GB, per tpMin-GPU group)
  upr: 52,         // users per replica
  reps: 2,         // replicas needed
  kvB: 1.049,      // KV per user GB at current ctx
  kvBpt: 0.131,    // KV GB per K-token (= MB/tok numerically)
  headroom: [...], // per-ctx users array [{ctx,users,tp}]
  exact: true,     // true if using exact HF arch
}
```

### Formulas
```
wt          = S.model × S.wp
ohGB        = S.gpu × S.overhead/100
usable      = S.gpu - ohGB
tpWeights   = nextPow2(ceil(wt / usable))
tpMin       = optimal TP minimising total GPUs (tries tpWeights..8)
aPA         = (usable × tpMin - wt) × 0.95   ← 0.95 = PagedAttention efficiency
kvBpt       = 2 × kvHeads × headDim × layers × kvBytes × 1000 / 1e9  (exact)
kvBpt       = 0.02 × params × kvBytes                                  (approx GQA)
kvBpt       = 0.04 × params × kvBytes                                  (approx MHA)
kvB         = kvBpt × S.ctx              ← GB per user at current context
upr         = floor(aPA / kvB)
t1          = ceil(S.users / upr) × tpMin
```

**Unit note:** `kvBpt` is in GB/K-tok. Numerically, 1 GB/K-tok = 1 MB/tok (units cancel). So `kvBpt` displayed as-is gives correct MB/tok values.

### UI Functions
```js
renderResult()          // re-renders entire right panel from S state
applyModelConfig(id,cfg)// loads HF model config into S, syncs chips
manualModel(v)          // slider drag — clears arch params, preserves S.wp if user-chosen
pickGpu(name,vram,aws,op,el) // GPU card click
apc(k,v,chip)           // generic chip click (wp, kv, attn, hitrate)
switchPage(id,elem)     // tab navigation
wizOpenAdvanced()       // wizard → Advanced (SILENT sync, ONE renderResult)
saveResult()            // append to savedResults[], update comparison table
renderSavedTable()      // rebuild saved results comparison HTML
showValidateModal()     // open AI validation modal
buildTechPrompt(raw)    // generate technical verification prompt
buildCFOPrompt(raw)     // generate CFO justification prompt
```

---

## Known Bug History (do not repeat these)

| Bug | Root cause | Fix |
|---|---|---|
| `S.wp=2` despite FP8 chip | `manualModel()` always reset wp | Only reset if `S.hfId` set |
| `S.wp=2` in wizard→Advanced | `setU()`/`setC()` triggered renders before wp sync | `wizOpenAdvanced()` now silent sync + one final render |
| MB/tok = 1048 (1000× too large) | `kvGB/S.ctx*1000` = MB/K-tok | `kvGB/S.ctx` = MB/tok |
| `renderSavedTable` JS crash | `'tech'` inside single-quoted string broke parser | Escaped to `\'tech\'` |
| Cloud CSV newlines | Literal `\n` in JS string | Escaped to `\\n` |
| Wrong insight for FP8 models | Didn't check if KV already FP8 | Check `prec.indexOf('FP8')` |

---

## CSS Variables (dark/light mode)
```css
--bg   --bg2  --bg3  --bg4     /* backgrounds */
--text --text2 --text3         /* text colors */
--blue --green --red --amber   /* semantic colors */
--border --border2             /* borders */
--bdim --bbord                 /* blue dim/border */
--adim --abord                 /* amber dim/border */
--display                      /* Syne display font */
--mono                         /* Syne Mono font */
--body                         /* Inter body font */
```

---

## How to Run Tests
```bash
cd /path/to/project
node gpu-calc-tests.js
# Expected: 60/60 passed · ALL PASS ✓
```

## How to Validate Syntax (quick)
```bash
python3 -c "
import re
with open('gpu-calc-final.html') as f: c=f.read()
inner=c[c.find('<script>\n')+9:c.rfind('</script>')]
dangerous=re.findall(r'[\"\\'][^\"\\']*/script[^\"\\'][\"\\']',inner,re.IGNORECASE)
opens=inner.count('{');closes=inner.count('}')
print('Script tags in strings:',len(dangerous),'(should be 0)')
print('Braces:',opens,'/',closes,'(should match)')
"
```

## How to Bump Version
```bash
# In gpu-calc-final.html, find and replace:
# 'real-time inference · v5.9' → 'real-time inference · v5.10'
```

---

## Feature Inventory

| Feature | Location | Notes |
|---|---|---|
| Quick Size wizard | `#page-wizard` | 5 steps, writes to S, shares engine with Advanced |
| Advanced calculator | `#page-calc` | Full controls, HF lookup, 4 tiers, cost model |
| GPU Explorer | `#page-explorer` | Bubble chart only — pricing table removed intentionally |
| Saved Results | `#page-saved` | Compare table with bars, PDF/Excel export |
| Validate with AI | Modal `#validate-modal` | Technical + CFO prompts, copy/open in Claude/ChatGPT |
| Author story | `#page-home` bottom | "Why this exists" + WIIFY cards |
| About drawer | `#about-drawer` | Opens when clicking `gpu.calc` logo |
| vLLM config generator | Modal `#vllm-modal` | CLI/Python/K8s YAML |
| Formulas & validation | Drawer in Advanced | KV formula, theoretical vs actual table |
| LinkedIn card | `.liwrap` | Auto-shows after 3s, 4hr dismiss |

---

## Deployment
```
1. Copy gpu-calc-final.html to repo as index.html
2. GitHub Pages: Settings → Pages → main branch / root
3. URL: https://nb-qbits.github.io/gpuinfer
```

---

## Contacts / Attribution
- **Author:** Vikas Grover
- **Technical review:** Trevor Royer (troyer on GitHub) — KV methodology
- **Community data needed:** L40S/H100/H200/B200 `num_gpu_blocks` from vLLM
  ```python
  llm = LLM(model="meta-llama/Llama-3.1-8B")
  print(llm.llm_engine.cache_config.num_gpu_blocks)
  # Submit to: https://github.com/nb-qbits/gpu-calc/issues
  ```
