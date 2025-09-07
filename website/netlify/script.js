async function loadRules() {
  const res = await fetch('./rules.json', { cache: 'no-store' });
  const data = await res.json();
  window.__rules_cache = data;
  return data;
}

async function loadMetrics() {
  try {
    const res = await fetch('./metrics.json', { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

function evaluate(params, rules) {
  // Simple rule evaluation: count failed rules
  let risk = 0;
  if (params.age === 'older_65') risk += rules.weights.age;
  if (params.cadence < rules.thresholds.cadence.min) risk += rules.weights.cadence;
  if (params.stride < rules.thresholds.stride.min) risk += rules.weights.stride;
  if (params.stv > rules.thresholds.stv.max) risk += rules.weights.stv;
  if (params.symm > rules.thresholds.symm.max) risk += rules.weights.symm;
  if (params.turn > rules.thresholds.turn.max) risk += rules.weights.turn;
  const label = risk >= rules.risk_cut ? 'Abnormal (Flagged)' : 'Normal (Not flagged)';
  return { risk, label };
}

function bindSlider(numberId, sliderId, bubbleId, toFixed = 2) {
  const num = document.getElementById(numberId);
  const slider = document.getElementById(sliderId);
  const bubble = document.getElementById(bubbleId);
  const sync = (v) => {
    num.value = Number(v).toFixed(toFixed);
    bubble.textContent = Number(v).toFixed(toFixed);
  };
  slider.addEventListener('input', (e) => sync(e.target.value));
  num.addEventListener('input', (e) => {
    slider.value = e.target.value;
    bubble.textContent = Number(e.target.value).toFixed(toFixed);
  });
  // init from slider defaults
  sync(slider.value || num.value || 0);
}

document.getElementById('evalBtn').addEventListener('click', async () => {
  const rules = await loadRules();
  const metrics = await loadMetrics();
  const params = {
    cadence: Number(document.getElementById('cadence').value || 0),
    stride: Number(document.getElementById('stride').value || 0),
    stv: Number(document.getElementById('stv').value || 0),
    symm: Number(document.getElementById('symm').value || 0),
    turn: Number(document.getElementById('turn').value || 0),
    age: document.getElementById('age').value,
  };
  const res = evaluate(params, rules);
  const el = document.getElementById('result');
  el.textContent = `${res.label} — Risk Score: ${res.risk.toFixed(1)}`;
  const m = document.getElementById('metrics');
  if (metrics && metrics.cv && metrics.cv.macro_ap !== undefined) {
    const pct = (x) => (x * 100).toFixed(5) + '%';
    m.textContent = `Model (CV macro AP): ${pct(metrics.cv.macro_ap)} | CNN: ${pct(metrics.cv.cnn)} | TCN: ${pct(metrics.cv.tcn)}`;
  } else {
    m.textContent = '';
  }
  // Care finder panel toggle
  const care = document.getElementById('care');
  care.style.display = res.label.startsWith('Abnormal') ? 'block' : 'none';
  if (res.label.startsWith('Abnormal')) showToast('Screening flagged abnormal. Consider consulting a clinician.');
});

// Expose for testing
window.__gait_rules = { loadRules, evaluate };

// Setup sliders after DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  bindSlider('cadence', 'cadence_slider', 'cadence_bubble', 0);
  bindSlider('stride', 'stride_slider', 'stride_bubble', 2);
  bindSlider('stv', 'stv_slider', 'stv_bubble', 0);
  bindSlider('symm', 'symm_slider', 'symm_bubble', 1);
  bindSlider('turn', 'turn_slider', 'turn_bubble', 1);
  // Care search handler
  const careBtn = document.getElementById('careSearch');
  if (careBtn) {
    careBtn.addEventListener('click', async () => {
      const loc = document.getElementById('loc').value.trim();
      const radius = Number(document.getElementById('radius').value || 10);
      const status = document.getElementById('careStatus');
      status.textContent = 'Opening care results...';
      const url = new URL('./care.html', window.location.href);
      url.searchParams.set('q', loc);
      url.searchParams.set('r', String(radius));
      window.open(url.toString(), '_blank');
    });
  }
  // Doctor summary generation
  const genSummaryBtn = document.getElementById('genSummaryBtn');
  if (genSummaryBtn) {
    genSummaryBtn.addEventListener('click', () => {
      const now = new Date().toISOString();
      const params = {
        cadence: document.getElementById('cadence').value,
        stride: document.getElementById('stride').value,
        stv: document.getElementById('stv').value,
        symm: document.getElementById('symm').value,
        turn: document.getElementById('turn').value,
        age: document.getElementById('age').value,
        timestamp: now,
      };
      const rules = window.__rules_cache;
      const evalRes = rules ? evaluate({
        cadence: Number(params.cadence||0),
        stride: Number(params.stride||0),
        stv: Number(params.stv||0),
        symm: Number(params.symm||0),
        turn: Number(params.turn||0),
        age: params.age
      }, rules) : { label: 'Unknown', risk: 0 };
      const handoff = draftClinicianNote(params, evalRes);
      try {
        const { jsPDF } = window.jspdf || {};
        if (jsPDF) {
          const doc = new jsPDF();
          doc.setFontSize(14);
          doc.text('Gait Screening — Doctor Summary', 10, 15);
          doc.setFontSize(10);
          const lines = [
            `Timestamp: ${params.timestamp}`,
            `Result: ${evalRes.label} (Risk ${evalRes.risk && evalRes.risk.toFixed ? evalRes.risk.toFixed(1) : evalRes.risk})`,
            '',
            `Cadence: ${params.cadence} steps/min`,
            `Stride length: ${params.stride} m`,
            `Step time variability: ${params.stv} ms`,
            `Symmetry: ${params.symm} %`,
            `Turn duration: ${params.turn} s`,
            `Age group: ${params.age}`,
            '',
            'Clinician Handoff (Draft):',
            ...(handoff || '').split('\n')
          ];
          let y = 25;
          lines.forEach(l => { doc.text(String(l), 10, y); y += 6; if (y > 280) { doc.addPage(); y = 15; } });
          doc.save('doctor_summary.pdf');
          showToast('PDF downloaded');
        } else {
          openSummaryWindow(params, evalRes, handoff);
          showToast('Summary opened. Use browser Print to PDF.');
        }
      } catch (e) {
        openSummaryWindow(params, evalRes, handoff);
        showToast('Summary opened. Use browser Print to PDF.');
      }
    });
  }
});

function showToast(text, ms = 3000) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = text;
  el.style.display = 'block';
  clearTimeout(window.__toast_timer);
  window.__toast_timer = setTimeout(() => { el.style.display = 'none'; }, ms);
}

function draftClinicianNote(params, evalRes) {
  const lines = [];
  lines.push('Clinician Handoff Note (Draft)');
  lines.push('');
  lines.push(`Timestamp: ${params.timestamp}`);
  lines.push(`Screening Result: ${evalRes.label} (Risk ${evalRes.risk.toFixed ? evalRes.risk.toFixed(1) : evalRes.risk})`);
  lines.push('Inputs:');
  lines.push(`- Cadence: ${params.cadence} steps/min`);
  lines.push(`- Stride length: ${params.stride} m`);
  lines.push(`- Step time variability: ${params.stv} ms`);
  lines.push(`- Symmetry: ${params.symm} %`);
  lines.push(`- Turn duration: ${params.turn} s`);
  lines.push(`- Age group: ${params.age}`);
  lines.push('');
  lines.push('Patient-reported symptoms or context (fill-in):');
  lines.push('- ');
  return lines.join('\n');
}

function openSummaryWindow(params, evalRes, handoffText) {
  const w = window.open('', '_blank');
  const safe = (s) => String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
  w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>Doctor Summary</title>
  <style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:20px;line-height:1.5;background:#fff;color:#111} .muted{color:#6b7280} pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px}</style>
  </head><body>
  <h2>Gait Screening — Doctor Summary</h2>
  <p class="muted">This summary is for clinical review. It is not a diagnosis. Please bring this to your clinician.</p>
  <h3>Result</h3>
  <p><strong>${safe(evalRes.label)}</strong> — Risk score: ${safe(evalRes.risk && evalRes.risk.toFixed ? evalRes.risk.toFixed(1) : evalRes.risk)}</p>
  <h3>Inputs</h3>
  <ul>
    <li>Cadence: ${safe(params.cadence)} steps/min</li>
    <li>Stride length: ${safe(params.stride)} m</li>
    <li>Step time variability: ${safe(params.stv)} ms</li>
    <li>Symmetry: ${safe(params.symm)} %</li>
    <li>Turn duration: ${safe(params.turn)} s</li>
    <li>Age group: ${safe(params.age)}</li>
    <li>Timestamp: ${safe(params.timestamp)}</li>
  </ul>
  <h3>Clinician Handoff (Draft)</h3>
  <pre>${safe(handoffText)}</pre>
  <p class="muted">Tip: Use your browser’s Print to PDF to save/share this page.</p>
  </body></html>`);
  w.document.close();
}

function renderHospitals(container, items) {
  if (!items || !items.length) {
    container.textContent = 'No facilities found within selected radius.';
    return;
  }
  const frag = document.createDocumentFragment();
  items.slice(0, 50).forEach(it => {
    const div = document.createElement('div');
    div.style.padding = '10px';
    div.style.borderBottom = '1px solid rgba(255,255,255,0.06)';
    const lines = [];
    lines.push(`<strong>${escapeHtml(it.name)}</strong> <span class="muted">(${escapeHtml(it.amenity||'')})</span>`);
    if (it.address_line1 || it.address_line2) lines.push(`<div class="muted">${escapeHtml([it.address_line1, it.address_line2].filter(Boolean).join(', '))}</div>`);
    if (it.phone) lines.push(`<div>📞 ${escapeHtml(it.phone)}</div>`);
    if (it.website) lines.push(`<div><a href="${escapeAttr(it.website)}" target="_blank" rel="noopener">Website</a></div>`);
    div.innerHTML = lines.join('');
    frag.appendChild(div);
  });
  container.innerHTML = '';
  container.appendChild(frag);
}

function escapeHtml(s) {
  return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
}
function escapeAttr(s) {
  return String(s||'').replace(/["']/g, c => ({'"':'&quot;','\'':'&#39;'}[c]));
}


