// Client-side inference using onnxruntime-web (WASM)
// Place your ONNX model at website/netlify/assets/model.onnx

const LABELS = ["StartHesitation", "Turn", "Walking"]; // keep in sync with backend

let __sessions = null; // { list: [session], weights: [w], names: [name] }

async function loadSessions(){
  if (__sessions) return __sessions;
  if (!window.ort) throw new Error('ONNX runtime (ort) not loaded');
  // Try ensemble config
  let weights = null;
  try {
    const res = await fetch('./assets/weights.json', { cache: 'no-store' });
    if (res.ok) weights = await res.json();
  } catch(e) {}

  const list = [];
  const wts = [];
  const names = [];
  if (weights && (weights.cnn_bilstm || weights.tcn)){
    if (weights.cnn_bilstm){
      list.push(await ort.InferenceSession.create('./assets/cnn_bilstm.onnx', { executionProviders: ['wasm'] }));
      wts.push(Number(weights.cnn_bilstm) || 0.5);
      names.push('cnn_bilstm');
    }
    if (weights.tcn){
      list.push(await ort.InferenceSession.create('./assets/tcn.onnx', { executionProviders: ['wasm'] }));
      wts.push(Number(weights.tcn) || 0.5);
      names.push('tcn');
    }
  }
  if (list.length === 0){
    // fallback: single model
    let path = './assets/model.onnx';
    try {
      const res = await fetch(path, { method: 'HEAD' });
      if (!res.ok) throw new Error();
    } catch(e) {
      path = './assets/cnn_bilstm.onnx';
    }
    list.push(await ort.InferenceSession.create(path, { executionProviders: ['wasm'] }));
    wts.push(1.0);
    names.push('single');
  }
  // normalize weights
  const sum = wts.reduce((a,b)=>a+b,0) || 1;
  for (let i=0;i<wts.length;i++) wts[i] = wts[i]/sum;
  __sessions = { list, weights: wts, names };
  return __sessions;
}

function parseCSV(text){
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return { columns: [], rows: [] };
  const header = lines[0].split(',').map(s => s.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++){
    const parts = lines[i].split(',');
    if (parts.length === 1 && parts[0].trim() === '') continue;
    const obj = {};
    for (let j = 0; j < header.length; j++) obj[header[j]] = (parts[j] ?? '').trim();
    rows.push(obj);
  }
  return { columns: header, rows };
}

function normalizeToThreeChannels(columns, rows){
  const lowerMap = Object.fromEntries(columns.map(c => [c.trim().toLowerCase(), c]));

  function toNum(col){
    const name = lowerMap[col];
    if (!name) return rows.map(() => 0);
    return rows.map(r => {
      const v = parseFloat(String(r[name]).replace(/[^0-9eE+\-\.]/g,''));
      return Number.isFinite(v) ? v : 0;
    });
  }

  // 1) accelerometer schema
  if (['accv','accml','accap'].every(k => k in lowerMap)){
    const v = toNum('accv');
    const ml = toNum('accml');
    const ap = toNum('accap');
    return stack3(v, ml, ap);
  }

  // 2) gait-parameter schema
  const gaitReq = ['cycle','stance_right','swing_right','stance_left','swing_left','step_length','step_width'];
  if (gaitReq.every(k => k in lowerMap)){
    // ignore labels if present (no-op since we address columns by name)
    const stepLen = toNum('step_length');
    const stepWid = toNum('step_width');
    const stanceR = toNum('stance_right');
    const stanceL = toNum('stance_left');
    const stanceBal = stanceR.map((v, i) => v - stanceL[i]);
    return stack3(stepLen, stepWid, stanceBal);
  }

  // 3) fallback: first 3 numeric non-time columns
  const exclude = new Set(['time','timestamp']);
  const numericCols = [];
  for (const c of columns){
    if (exclude.has(c.trim().toLowerCase())) continue;
    const arr = rows.map(r => parseFloat(String(r[c]).replace(/[^0-9eE+\-\.]/g,''))).filter(v => Number.isFinite(v));
    if (arr.length > 0) numericCols.push(c);
    if (numericCols.length === 3) break;
  }
  while (numericCols.length < 3) numericCols.push(null);
  const a = numericCols[0] ? rows.map(r => numOr0(r[numericCols[0]])) : rows.map(()=>0);
  const b = numericCols[1] ? rows.map(r => numOr0(r[numericCols[1]])) : rows.map(()=>0);
  const c = numericCols[2] ? rows.map(r => numOr0(r[numericCols[2]])) : rows.map(()=>0);
  return stack3(a, b, c);
}

function numOr0(v){
  const x = parseFloat(String(v).replace(/[^0-9eE+\-\.]/g,''));
  return Number.isFinite(x) ? x : 0;
}

function stack3(a, b, c){
  const T = Math.max(a.length, b.length, c.length);
  const out = new Float32Array(T * 3);
  for (let i = 0; i < T; i++){
    out[i * 3 + 0] = a[i] ?? 0;
    out[i * 3 + 1] = b[i] ?? 0;
    out[i * 3 + 2] = c[i] ?? 0;
  }
  return { data: out, T };
}

function standardize3(buffer){
  const T = buffer.length / 3;
  const mean = [0,0,0], std = [0,0,0];
  for (let i = 0; i < T; i++){
    mean[0] += buffer[i*3+0];
    mean[1] += buffer[i*3+1];
    mean[2] += buffer[i*3+2];
  }
  mean[0] /= T; mean[1] /= T; mean[2] /= T;
  for (let i = 0; i < T; i++){
    std[0] += Math.pow(buffer[i*3+0] - mean[0], 2);
    std[1] += Math.pow(buffer[i*3+1] - mean[1], 2);
    std[2] += Math.pow(buffer[i*3+2] - mean[2], 2);
  }
  std[0] = Math.sqrt(std[0] / Math.max(1, T-1)) + 1e-6;
  std[1] = Math.sqrt(std[1] / Math.max(1, T-1)) + 1e-6;
  std[2] = Math.sqrt(std[2] / Math.max(1, T-1)) + 1e-6;
  for (let i = 0; i < T; i++){
    buffer[i*3+0] = (buffer[i*3+0] - mean[0]) / std[0];
    buffer[i*3+1] = (buffer[i*3+1] - mean[1]) / std[1];
    buffer[i*3+2] = (buffer[i*3+2] - mean[2]) / std[2];
  }
}

function sigmoid(x){ return 1 / (1 + Math.exp(-x)); }

async function windowAndPredictCSVText(csvText){
  const { columns, rows } = parseCSV(csvText);
  if (rows.length === 0) throw new Error('Empty CSV');
  const { data, T } = normalizeToThreeChannels(columns, rows);
  standardize3(data);
  const { list, weights } = await loadSessions();
  const tensor = new ort.Tensor('float32', data, [1, T, 3]);
  let probs = null;
  for (let i = 0; i < list.length; i++){
    const session = list[i];
    const inputName = session.inputNames ? session.inputNames[0] : Object.keys(session.inputNames || {})[0];
    const outputName = session.outputNames ? session.outputNames[0] : undefined;
    const outputs = await session.run({ [inputName]: tensor });
    const out = outputName ? outputs[outputName] : Object.values(outputs)[0];
    const logits = Array.from(out.data);
    let p;
    if (out.dims.length === 3){
      const TT = out.dims[1], K = out.dims[2];
      const arr = new Array(K).fill(0);
      for (let t = 0; t < TT; t++){
        for (let k = 0; k < K; k++) arr[k] += sigmoid(logits[t*K + k]);
      }
      p = arr.map(v => v / TT);
    } else if (out.dims.length === 2){
      const K = out.dims[1];
      p = new Array(K).fill(0).map((_,k) => sigmoid(logits[k]));
    } else {
      p = logits.slice(0, LABELS.length).map(sigmoid);
    }
    if (!probs) probs = p.map(x => weights[i]*x); else {
      for (let k=0;k<Math.min(probs.length, p.length);k++) probs[k] += weights[i]*p[k];
    }
  }
  const score = Math.max(...probs);
  const label = score >= 0.5 ? 'Abnormal (Flagged)' : 'Normal (Not flagged)';
  const probsObj = {};
  for (let i = 0; i < Math.min(LABELS.length, probs.length); i++) probsObj[LABELS[i]] = probs[i];
  return { label, score, probs: probsObj };
}

window.__infer = { parseCSV, windowAndPredictCSVText };


