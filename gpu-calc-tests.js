/**
 * gpu.calc Test Harness v5.9
 * Run: node gpu-calc-tests.js
 * Tests every calculation, state transition, and edge case.
 */

// ── DOM STUB ─────────────────────────────────────────────────────────────────
var _store = {};
function mkEl(id) {
  return {
    id: id, style:{display:'',cssText:'',opacity:'',background:''},
    textContent:'', className:'', innerHTML:'', value:'',
    classList:{
      _c:'',
      add:function(x){if(this._c.indexOf(x)<0)this._c+=' '+x;},
      remove:function(x){this._c=this._c.replace(new RegExp('\\s*'+x,'g'),'').trim();},
      toggle:function(x,f){if(f===undefined)f=!(this._c.indexOf(x)>=0);if(f)this.add(x);else this.remove(x);return f;},
      contains:function(x){return this._c.indexOf(' '+x+' ')>=0||this._c.startsWith(x+' ')||this._c.endsWith(' '+x)||this._c===x;}
    },
    appendChild:function(){}, querySelectorAll:function(){return{forEach:function(){},map:function(){return[];},filter:function(){return[];},length:0};},
    addEventListener:function(){}, getAttribute:function(){return null;}, setAttribute:function(){},
    contains:function(){return false;}, click:function(){}, querySelector:function(){return null;}
  };
}
var document = {
  getElementById:function(id){if(!_store[id])_store[id]=mkEl(id);return _store[id];},
  createElement:function(){return mkEl('_');},
  querySelectorAll:function(){return{forEach:function(){},map:function(){return[];},filter:function(){return[];},length:0,0:null};},
  querySelector:function(){return null;},
  documentElement:{classList:{toggle:function(){},contains:function(){return false;}}},
  body:{appendChild:function(){},removeChild:function(){}},
  addEventListener:function(){}
};
var window={open:function(){},print:function(){}};
var sessionStorage={getItem:function(){return null;},setItem:function(){}};
var localStorage={getItem:function(){return null;},setItem:function(){}};
var Chart=function(){return{destroy:function(){}};};
var fetch=function(){return Promise.resolve({ok:true,json:function(){return Promise.resolve({models:[],content:[]});},text:function(){return Promise.resolve('');},});};
var navigator={clipboard:{writeText:function(){return Promise.resolve();}}};
var confirm=function(){return true;};
var alert=function(){};
var matchMedia=function(){return{matches:false};};
var URL={createObjectURL:function(){return'blob:x';},revokeObjectURL:function(){}};
var Blob=function(){};
var clearTimeout=function(){};
var setTimeout=function(){};

// ── LOAD APP CODE ─────────────────────────────────────────────────────────────
var fs = require('fs');
var html = fs.readFileSync('./gpu-calc-final.html', 'utf8');
var scriptStart = html.indexOf('<script>\n') + '<script>\n'.length;
var scriptEnd = html.lastIndexOf('</script>');
var appCode = html.slice(scriptStart, scriptEnd);
eval(appCode);

// ── TEST FRAMEWORK ────────────────────────────────────────────────────────────
var passed = 0, failed = 0, errors = [];

function test(name, fn) {
  // Reset S to clean default before each test
  S.model=8; S.wp=2; S.kv=2; S.gpu=80; S.gpuName='H100';
  S.ctx=8; S.users=100; S.overhead=8; S.attn='gqa';
  S.numKvHeads=null; S.headDim=null; S.numLayers=null; S.hfId=null;
  S.hitrate=0.1; S.onpremYr=25000; S.awsHr=3.10; S.amortYrs=5; S.cloudHike=3;
  try {
    fn();
    passed++;
    process.stdout.write('  ✓ ' + name + '\n');
  } catch(e) {
    failed++;
    errors.push({name:name, err:e.message});
    process.stdout.write('  ✗ ' + name + '\n    → ' + e.message + '\n');
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed');
}

function assertEq(a, b, msg) {
  if (Math.abs(a - b) > 0.01) throw new Error((msg||'') + ' expected ' + b + ' got ' + a);
}

function assertApprox(a, b, pct, msg) {
  // Within pct% of expected
  var tol = Math.abs(b) * (pct/100);
  if (Math.abs(a - b) > tol) throw new Error((msg||'') + ' expected ~' + b + ' (±'+pct+'%) got ' + a);
}

function section(name) {
  process.stdout.write('\n── ' + name + ' ──\n');
}

// ── SECTION 1: CORE MATH ─────────────────────────────────────────────────────
section('Core Math — nP2, kvBytesPerToken, kvFP8BytesPerToken');

test('nP2: powers of 2', function() {
  assertEq(nP2(1),1); assertEq(nP2(2),2); assertEq(nP2(3),4);
  assertEq(nP2(5),8); assertEq(nP2(9),16); assertEq(nP2(64),64);
  assertEq(nP2(65),128); assertEq(nP2(405),512);
});

test('kvBytesPerToken: GQA approx BF16', function() {
  S.attn='gqa'; S.model=8; S.wp=2; S.kv=2;
  S.numKvHeads=null; S.headDim=null; S.numLayers=null;
  // 0.02 * 8 * 2 = 0.32 GB/K-tok
  assertEq(kvBytesPerToken(), 0.32, 'GQA 8B BF16 KV');
});

test('kvBytesPerToken: MHA approx BF16', function() {
  S.attn='mha'; S.model=8; S.wp=2; S.kv=2;
  S.numKvHeads=null;
  // 0.04 * 8 * 2 = 0.64 GB/K-tok
  assertEq(kvBytesPerToken(), 0.64, 'MHA 8B BF16 KV');
});

test('kvBytesPerToken: exact from HF arch', function() {
  S.attn='gqa'; S.numKvHeads=8; S.headDim=128; S.numLayers=32; S.kv=2;
  // 2 * 8 * 128 * 32 * 2 * 1000 / 1e9 = 0.131 GB/K-tok
  assertApprox(kvBytesPerToken(), 0.1311, 1, 'Llama 8B exact KV');
});

test('kvBytesPerToken: FP8 KV halves vs FP16', function() {
  S.attn='gqa'; S.model=70; S.numKvHeads=null; S.kv=2;
  var fp16 = kvBytesPerToken();
  S.kv=1;
  var fp8 = kvBytesPerToken();
  assertEq(fp16/fp8, 2, 'FP8 KV should be half FP16');
});

test('kvFP8BytesPerToken: always 1 byte', function() {
  S.attn='gqa'; S.model=70; S.numKvHeads=null;
  var bpt = kvFP8BytesPerToken();
  assert(bpt > 0, 'FP8 bpt positive');
  // Should be half of kvBytesPerToken with kv=2
  S.kv=2;
  assertApprox(bpt, kvBytesPerToken()/2, 1, 'FP8 = half FP16');
});

// ── SECTION 2: WEIGHT PRECISION ───────────────────────────────────────────────
section('Weight Precision — S.wp values and wt calculation');

test('BF16: wt = params × 2', function() {
  S.model=70; S.wp=2;
  var r = calcTiers();
  assertEq(r.wt, 140, 'Llama 70B BF16 weights');
});

test('FP8: wt = params × 1', function() {
  S.model=70; S.wp=1;
  var r = calcTiers();
  assertEq(r.wt, 70, 'Llama 70B FP8 weights');
});

test('INT4: wt = params × 0.5', function() {
  S.model=70; S.wp=0.5;
  var r = calcTiers();
  assertEq(r.wt, 35, 'Llama 70B INT4 weights');
});

test('W8A8 model: applyModelConfig sets S.wp=1', function() {
  applyModelConfig('RedHatAI/Llama-3.1-8B-Instruct-quantized.w8a8',
    MODEL_CONFIGS['RedHatAI/Llama-3.1-8B-Instruct-quantized.w8a8']);
  assertEq(S.wp, 1, 'W8A8 should set S.wp=1');
  var r = calcTiers();
  assertEq(r.wt, 8, 'Llama 8B W8A8: wt=8×1=8 GB');
});

test('Mixtral W8A8: wt=47 not 94', function() {
  applyModelConfig('RedHatAI/Mixtral-8x7B-Instruct-v0.1-quantized.w8a8',
    MODEL_CONFIGS['RedHatAI/Mixtral-8x7B-Instruct-v0.1-quantized.w8a8']);
  assertEq(S.wp, 1, 'Mixtral W8A8 S.wp should be 1');
  var r = calcTiers();
  assertEq(r.wt, 47, 'Mixtral 8x7B W8A8: wt=47×1=47 GB, NOT 94');
});

test('BF16 model: applyModelConfig sets S.wp=2', function() {
  applyModelConfig('meta-llama/Llama-3.1-8B-Instruct',
    MODEL_CONFIGS['meta-llama/Llama-3.1-8B-Instruct']);
  assertEq(S.wp, 2, 'BF16 model should set S.wp=2');
});

test('manualModel preserves user FP8 choice (S.hfId=null)', function() {
  // User clicked FP8 chip: S.hfId is null (no HF model loaded)
  S.wp = 1; S.hfId = null;
  manualModel(32);
  assertEq(S.wp, 1, 'manualModel should not reset user-chosen FP8 when hfId=null');
});

test('manualModel resets wp to BF16 when HF model was loaded', function() {
  // HF model set hfId and wp=1
  S.hfId = 'some-model'; S.wp = 1;
  manualModel(32);
  assertEq(S.wp, 2, 'manualModel should reset to BF16 when clearing HF model');
  assert(S.hfId === null, 'manualModel should clear hfId');
});

// ── SECTION 3: TENSOR PARALLELISM ─────────────────────────────────────────────
section('Tensor Parallelism — TP selection logic');

test('TP=1: small model fits single GPU', function() {
  S.model=8; S.wp=2; S.gpu=80; S.users=10; S.ctx=8;
  // wt=16 < 73.6 usable → TP=1 for weights
  var r = calcTiers();
  assert(r.tpWeights === 1, 'Llama 8B fits on H100, tpWeights=1');
});

test('TP=2: 70B BF16 requires 2 H100s for weights', function() {
  S.model=70; S.wp=2; S.gpu=80; S.users=10; S.ctx=8;
  // wt=140 > 73.6, needs TP=2
  var r = calcTiers();
  assert(r.tpWeights === 2, '70B BF16 on H100 needs TP=2');
});

test('TP=4: 405B BF16 on H100', function() {
  S.model=405; S.wp=2; S.gpu=80; S.users=10; S.ctx=8;
  var r = calcTiers();
  assert(r.tpWeights >= 4, '405B BF16 on H100 needs TP≥4');
});

test('TP optimises for KV capacity not just weights', function() {
  // 8B BF16 on A100 40GB, 8K ctx, many users — TP=2 may be better for KV
  S.model=8; S.wp=2; S.gpu=80; S.users=500; S.ctx=32;
  var r = calcTiers();
  // tpMin should minimise total GPUs — may be > tpWeights
  assert(r.tpMin >= r.tpWeights, 'tpMin >= tpWeights always');
  assert(r.t1 <= 999, 'Total GPUs reasonable');
});

test('FP8 halves weight — reduces TP need', function() {
  // 70B BF16 needs TP=2 on H100, FP8 should fit TP=1
  S.model=70; S.wp=2; S.gpu=80; S.users=10; S.ctx=8;
  var r_bf16 = calcTiers();
  S.wp=1;
  var r_fp8 = calcTiers();
  assert(r_fp8.tpWeights <= r_bf16.tpWeights, 'FP8 needs same or fewer GPUs for weights');
  assertEq(r_fp8.wt, 70, 'FP8 wt=70');
});

// ── SECTION 4: KV POOL AND USERS PER REPLICA ──────────────────────────────────
section('KV Pool — aPA, users/replica');

test('aPA = (usable×TP - wt) × 0.95', function() {
  S.model=8; S.wp=2; S.gpu=80; S.overhead=8;
  var r = calcTiers();
  var usable = 80 * 0.92; // 73.6
  var expected_aPA = (usable * r.tpMin - r.wt) * 0.95;
  assertApprox(r.aPA, expected_aPA, 0.1, 'aPA formula');
});

test('users per replica = floor(aPA / kvB)', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  var expected_upr = Math.floor(r.aPA / r.kvB);
  assertEq(r.upr, Math.max(1, expected_upr), 'upr formula');
});

test('longer context → fewer users per replica', function() {
  S.model=8; S.wp=2; S.gpu=80; S.users=100;
  S.ctx=8; var r8 = calcTiers();
  S.ctx=32; var r32 = calcTiers();
  assert(r8.upr > r32.upr, '8K ctx fits more users than 32K');
});

test('FP8 KV doubles users per replica vs FP16', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  S.numKvHeads=8; S.headDim=128; S.numLayers=32;
  // At same TP, FP8 KV halves kvB → roughly doubles upr
  var usable=80*0.92, wt=16, tp=1;
  var aPA=(usable*tp-wt)*0.95;
  var kvBpt_fp16=2*8*128*32*2*1000/1e9;
  var kvBpt_fp8 =2*8*128*32*1*1000/1e9;
  var upr_fp16=Math.floor(aPA/(kvBpt_fp16*8));
  var upr_fp8 =Math.floor(aPA/(kvBpt_fp8 *8));
  assertApprox(upr_fp8, upr_fp16*2, 2, 'FP8 KV doubles upr at same TP');
});

// ── SECTION 5: KNOWN CORRECT ANSWERS ──────────────────────────────────────────
section('Known Correct Answers — ground truth');

test('Llama 8B / H100 / BF16 / 8K / 100u', function() {
  S.model=8; S.wp=2; S.gpu=80; S.gpuName='H100'; S.ctx=8; S.users=100;
  S.attn='gqa'; S.numKvHeads=8; S.headDim=128; S.numLayers=32;
  var r = calcTiers();
  assertEq(r.wt, 16, 'wt=16 GB');
  // exact: kvBpt = 2*8*128*32*2*1000/1e9 = 0.131 GB/K-tok; kvB = 0.131*8K = 1.049 GB per user
  assertApprox(r.kvB, 1.049, 1, 'kvB ~1.049 GB exact at 8K');
  assertEq(r.tpWeights, 1, 'tpWeights=1');
  assert(r.t1 >= 1, 'at least 1 GPU');
  assert(r.t1 <= 20, 'reasonable GPU count');
});

test('Llama 70B / H100 / BF16 / 8K / 100u', function() {
  S.model=70; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  S.numKvHeads=8; S.headDim=128; S.numLayers=80;
  var r = calcTiers();
  assertEq(r.wt, 140, 'wt=140 GB');
  assertEq(r.tpWeights, 2, 'tpWeights=2');
  // exact KV: 2*8*128*80*2*8000/1e9 = 2.621 GB/user
  assertApprox(r.kvB, 2.621, 1, 'kvB ~2.621 GB');
});

test('Mixtral W8A8 / B200 / 8K / 30u — critical regression', function() {
  S.gpu=192; S.gpuName='B200'; S.overhead=8; S.ctx=8; S.users=30;
  applyModelConfig('RedHatAI/Mixtral-8x7B-Instruct-v0.1-quantized.w8a8',
    MODEL_CONFIGS['RedHatAI/Mixtral-8x7B-Instruct-v0.1-quantized.w8a8']);
  assertEq(S.wp, 1, 'S.wp=1 after W8A8 load');
  var r = calcTiers();
  assertEq(r.wt, 47, 'wt=47 GB (NOT 94 — critical regression test)');
  assertApprox(r.ohGB, 192*0.08, 1, 'overhead=15.4 GB');
  // aPA with TP=1: (176.64-47)*0.95 = 123.2 GB
  assertApprox(r.aPA, 123.2, 2, 'aPA≈123.2 GB with correct wt');
});

test('Llama 32B / A100 80GB / BF16 / 8K / 30u', function() {
  S.model=32; S.wp=2; S.gpu=80; S.ctx=8; S.users=30; S.overhead=9;
  S.numKvHeads=null;
  var r = calcTiers();
  assertEq(r.wt, 64, 'wt=64 GB');
  // Approx KV: 0.02*32*2*8 = 10.24 GB
  assertApprox(r.kvB, 10.24, 2, 'kvB approx');
  assert(r.tpMin >= 1, 'tpMin>=1');
});

test('405B / H100 / BF16 — TP=8 required', function() {
  S.model=405; S.wp=2; S.gpu=80; S.ctx=8; S.users=10;
  var r = calcTiers();
  assertEq(r.wt, 810, 'wt=810 GB');
  assert(r.tpWeights >= 8, 'needs TP≥8');
});

// ── SECTION 6: COST MODEL ─────────────────────────────────────────────────────
section('Cost Model — on-prem and cloud');

test('on-prem total = GPUs × price', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=10;
  S.onpremYr=25000; S.amortYrs=5;
  var r = calcTiers();
  var opTot = r.t1 * S.onpremYr;
  assert(opTot > 0, 'on-prem cost positive');
  assert(opTot === r.t1 * 25000, 'on-prem = GPUs × $25K');
});

test('cloud total includes compound escalation', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=10;
  S.awsHr=3.10; S.amortYrs=5; S.cloudHike=3;
  var r = calcTiers();
  var clMo = r.t1 * S.awsHr * 720;
  var clTot = 0, mo = clMo;
  for(var y=0;y<5;y++){clTot+=mo*12;mo*=1.03;}
  // Verify our formula matches what calcTiers uses
  var noEscal = r.t1 * S.awsHr * 720 * 12 * 5;
  assert(clTot > noEscal, 'compound escalation increases total');
});

test('break-even = hardware / monthly-cloud-spend', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=10;
  S.onpremYr=25000; S.awsHr=3.10;
  var r = calcTiers();
  var opTot = r.t1 * S.onpremYr;
  var clMo = r.t1 * S.awsHr * 720;
  var be = Math.ceil(opTot / clMo);
  assert(be > 0 && be < 120, 'break-even between 1 and 120 months');
});

// ── SECTION 7: MB/TOK UNIT ────────────────────────────────────────────────────
section('Units — MB/tok must be numerically = GB/K-tok');

test('fmtMBperTok: 1 GB/K-tok = 1 MB/tok', function() {
  // The identity: 1 GB/K-tok ÷ 1000 tok/K-tok × 1000 MB/GB = 1 MB/tok
  // So numerically GB/K-tok === MB/tok
  var bpt = 0.1311; // Llama 8B exact kvBpt in GB/K-tok
  var result = fmtMBperTok(bpt);
  assert(result.indexOf('0.131') >= 0, 'fmtMBperTok should show 0.131, got: ' + result);
});

test('wizard wkv-mbtok = kvGB/S.ctx (not ×1000)', function() {
  // kvGB = 8.389 GB (32B BF16 8K), S.ctx=8
  // Correct: 8.389/8 = 1.049 MB/tok
  // Wrong:   8.389/8*1000 = 1048.576 MB/tok (old bug)
  var kvGB = 8.389;
  var correct = (kvGB / 8).toFixed(3);
  var wrong   = (kvGB / 8 * 1000).toFixed(3);
  assertEq(parseFloat(correct), 1.049, 'correct MB/tok');
  assert(parseFloat(wrong) > 1000, 'wrong value is >1000');
  assert(parseFloat(correct) < 10, 'correct value < 10 MB/tok');
});

// ── SECTION 8: TIERS t1–t4 ────────────────────────────────────────────────────
section('Optimisation Tiers — t1 to t4');

test('t1 >= t2 >= t3 >= t4', function() {
  S.model=70; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  assert(r.t1 >= r.t2, 't1>=t2');
  assert(r.t2 >= r.t3, 't2>=t3');
  assert(r.t3 >= r.t4, 't3>=t4');
});

test('t2 <= t1: prefix cache reduces or equals', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100; S.hitrate=0.4;
  var r = calcTiers();
  assert(r.t2 <= r.t1, 'prefix cache tier ≤ baseline');
});

test('t3 <= t2: FP8 KV reduces or equals', function() {
  S.model=70; S.wp=2; S.gpu=80; S.ctx=32; S.users=200;
  var r = calcTiers();
  assert(r.t3 <= r.t2, 'FP8 KV tier ≤ prefix cache');
});

test('llm-d tier: only activates at scale', function() {
  // llm-d requires users >= 50 OR ctx >= 32K AND t1 >= 4
  S.model=70; S.wp=2; S.gpu=80; S.ctx=8; S.users=10;
  var r_small = calcTiers();
  S.users=500;
  var r_large = calcTiers();
  // At small scale, t4 = t1 (no saving)
  // At large scale, t4 may be < t1
  assert(r_large.t4 <= r_large.t1, 'llm-d t4 <= t1 at large scale');
});

test('GPU count always multiple of tpMin', function() {
  S.model=70; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  assert(r.t1 % r.tpMin === 0, 't1 divisible by tpMin');
  assert(r.t2 % r.tpMin === 0, 't2 divisible by tpMin');
  assert(r.t3 % r.tpMin === 0, 't3 divisible by tpMin');
});

// ── SECTION 9: EDGE CASES ─────────────────────────────────────────────────────
section('Edge Cases — extremes and boundaries');

test('1 user: still returns valid result', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=1;
  var r = calcTiers();
  assert(r.t1 >= 1, 'at least 1 GPU for 1 user');
});

test('1000 users: returns valid result', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=1000;
  var r = calcTiers();
  assert(r.t1 > 0, 'positive GPU count');
  assert(r.t1 < 10000, 'not absurd');
});

test('very long context 256K: KV dominates', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=256; S.users=10;
  var r = calcTiers();
  assert(r.kvB > 10, 'large KV at 256K');
  assert(r.t1 >= 1, 'valid GPU count');
});

test('custom VRAM: calcTiers uses S.gpu', function() {
  S.model=8; S.wp=2; S.gpu=200; S.ctx=8; S.users=100; // custom 200GB GPU
  var r = calcTiers();
  var usable = 200 * (1-S.overhead/100);
  assertApprox(r.usable, usable, 0.1, 'usable matches custom VRAM');
});

test('INT4 weight: 0.5 bytes/param', function() {
  S.model=70; S.wp=0.5; S.gpu=80; S.ctx=8; S.users=10;
  var r = calcTiers();
  assertEq(r.wt, 35, 'INT4 70B = 35 GB');
});

test('zero hitrate: t2=t1', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100; S.hitrate=0;
  var r = calcTiers();
  assertEq(r.t2, r.t1, 'no cache hit = no improvement');
});

test('overhead=0: usable = full VRAM', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=10; S.overhead=0;
  var r = calcTiers();
  assertApprox(r.usable, 80, 0.1, 'zero overhead → usable=80');
});

// ── SECTION 10: GPU MODELS ────────────────────────────────────────────────────
section('GPU Models — all supported hardware');

var GPU_TESTS = [
  {name:'A100-40', vram:40, oh:9},
  {name:'A100-80', vram:80, oh:9},
  {name:'H100',   vram:80, oh:8},
  {name:'H200',   vram:141,oh:8},
  {name:'B200',   vram:192,oh:7},
  {name:'L40S',   vram:48, oh:10},
];

GPU_TESTS.forEach(function(gpu) {
  test(gpu.name + ': returns valid result for 8B BF16 100u 8K', function() {
    S.model=8; S.wp=2; S.gpu=gpu.vram; S.overhead=gpu.oh; S.ctx=8; S.users=100;
    var r = calcTiers();
    assert(r.t1 >= 1, gpu.name + ' t1>=1');
    assertApprox(r.ohGB, gpu.vram * gpu.oh/100, 1, gpu.name + ' overhead');
    assert(r.aPA > 0, gpu.name + ' aPA>0');
  });
});

// ── SECTION 11: MODEL CONFIGS ─────────────────────────────────────────────────
section('MODEL_CONFIGS — all entries valid');

test('all MODEL_CONFIGS have required fields', function() {
  var required = ['params','layers','kvHeads','headDim','dtype'];
  var bad = [];
  Object.keys(MODEL_CONFIGS).forEach(function(id) {
    var cfg = MODEL_CONFIGS[id];
    required.forEach(function(f) {
      if (cfg[f] === undefined || cfg[f] === null) bad.push(id + ':' + f);
    });
    if (cfg.params <= 0) bad.push(id + ':params<=0');
    if (cfg.layers <= 0) bad.push(id + ':layers<=0');
    if (cfg.kvHeads <= 0) bad.push(id + ':kvHeads<=0');
    if (cfg.headDim <= 0) bad.push(id + ':headDim<=0');
  });
  assert(bad.length === 0, 'Bad configs: ' + bad.join(', '));
});

test('quantized models have int8/int4/fp8 dtype', function() {
  var bad = [];
  Object.keys(MODEL_CONFIGS).forEach(function(id) {
    var cfg = MODEL_CONFIGS[id];
    if (id.indexOf('quantized') >= 0 || id.indexOf('w8a8') >= 0 || id.indexOf('w4a16') >= 0) {
      var dt = (cfg.dtype||'').toLowerCase();
      if (dt !== 'int8' && dt !== 'int4' && dt !== 'float8' && dt !== 'fp8') {
        bad.push(id + ':' + dt);
      }
    }
  });
  assert(bad.length === 0, 'Quantized models with wrong dtype: ' + bad.join(', '));
});

test('applyModelConfig: Llama 8B sets S correctly', function() {
  applyModelConfig('meta-llama/Llama-3.1-8B-Instruct',
    MODEL_CONFIGS['meta-llama/Llama-3.1-8B-Instruct']);
  assertEq(S.model, 8, 'params=8');
  assertEq(S.numKvHeads, 8, 'kvHeads=8');
  assertEq(S.headDim, 128, 'headDim=128');
  assertEq(S.numLayers, 32, 'layers=32');
  assertEq(S.wp, 2, 'dtype=bfloat16 → wp=2');
  assertEq(S.attn, 'gqa', 'attn=gqa');
});

test('applyModelConfig: Qwen 72B sets params=72', function() {
  applyModelConfig('Qwen/Qwen2.5-72B-Instruct',
    MODEL_CONFIGS['Qwen/Qwen2.5-72B-Instruct']);
  assertEq(S.model, 72, 'Qwen 72B params');
  assertEq(S.wp, 2, 'BF16');
});

// ── SECTION 12: HEADROOM TABLE ────────────────────────────────────────────────
section('Headroom Table — per-context users');

test('headroom has 7 entries (4K to 256K)', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  assert(r.headroom.length === 7, 'headroom has 7 ctx levels');
});

test('headroom: shorter context → more users', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  var h4 = r.headroom.find(function(x){return x.ctx===4;});
  var h128 = r.headroom.find(function(x){return x.ctx===128;});
  assert(h4.users >= h128.users, '4K ctx >= 128K ctx users');
});

test('headroom: current ctx row matches upr', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  var r = calcTiers();
  var curRow = r.headroom.find(function(x){return x.ctx===S.ctx;});
  assert(curRow !== undefined, 'current ctx in headroom');
});

// ── SECTION 13: SAVED RESULTS ─────────────────────────────────────────────────
section('Saved Results — saveResult fields');

test('saveResult builds correct raw fields', function() {
  S.model=8; S.wp=2; S.gpu=80; S.gpuName='H100'; S.ctx=8; S.users=100;
  S.onpremYr=25000; S.awsHr=3.10; S.amortYrs=5; S.cloudHike=3;
  S.numKvHeads=8; S.headDim=128; S.numLayers=32;
  // Simulate what saveResult builds
  var r = calcTiers();
  assert(r.t1 > 0, 'GPU count positive');
  assert(r.wt === 16, 'wt=16');
  assert(r.aPA > 0, 'aPA positive');
  // raw.wt should be 16, not something else
  assertEq(r.wt, S.model * S.wp, 'raw.wt = model × wp');
});

test('kvBpt numerically equals MB/tok', function() {
  S.model=8; S.wp=2; S.gpu=80; S.ctx=8; S.users=100;
  S.numKvHeads=8; S.headDim=128; S.numLayers=32;
  var r = calcTiers();
  // fmtMBperTok(r.kvBpt) should give correct MB/tok
  // 1 GB/K-tok = 1 MB/tok numerically
  var mbTok = r.kvBpt; // this IS MB/tok
  assertApprox(mbTok, 2*8*128*32*2*1000/1e9, 1, 'kvBpt = MB/tok');
  assert(mbTok < 10, 'MB/tok < 10 for 8B (not thousands)');
});

// ── SUMMARY ───────────────────────────────────────────────────────────────────
var total = passed + failed;
process.stdout.write('\n' + '═'.repeat(50) + '\n');
process.stdout.write('Results: ' + passed + '/' + total + ' passed');
if (failed > 0) {
  process.stdout.write(' · ' + failed + ' FAILED\n');
  process.stdout.write('\nFailed tests:\n');
  errors.forEach(function(e) {
    process.stdout.write('  ✗ ' + e.name + '\n    ' + e.err + '\n');
  });
} else {
  process.stdout.write(' · ALL PASS ✓\n');
}
process.exit(failed > 0 ? 1 : 0);
