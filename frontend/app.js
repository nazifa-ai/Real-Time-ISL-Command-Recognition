const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[ch]));

let stream = null, eventId = null, liveSessionId = null;
let liveTimer = null, frameCount = 0, frameStartedAt = 0, videoFrameLoop = 0, inferenceFrameLoop = 0;
let frameCanvas = null, sequenceLength = 30;
let currentModel = 'gru_mha';
let pretrainedInfo = {};
let modelInfoCache = null;
let vocabulary = [];
let perSignRows = [];
const SETTINGS_KEY = 'isl-recognition-session-settings-v1';
let sessionSettings = { threshold: null, smoothingWindow: 3, cameraDeviceId: '', voiceURI: '' };
let uncertaintyPrompted = false;
try { sessionSettings = { ...sessionSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') }; } catch (_) {}

function showError(msg) {
  const b = $('errorBanner');
  b.textContent = '⚠ ' + msg;
  b.hidden = false;
}
function clearError() { $('errorBanner').hidden = true; }

function speakText(text) {
  if (!('speechSynthesis' in window) || !window.SpeechSynthesisUtterance) {
    $('feedbackStatus').textContent = 'Speech is unavailable in this browser. You can still read the caption.';
    $('feedbackStatus').className = 'feedback-status err';
    if ($('speechStatus')) $('speechStatus').textContent = 'Unavailable: this browser does not provide speech synthesis.';
    return false;
  }
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) {
    $('feedbackStatus').textContent = 'No system voice is available. Install/enable a voice in Windows speech settings.';
    $('feedbackStatus').className = 'feedback-status err';
    if ($('speechStatus')) $('speechStatus').textContent = 'No browser voices are available on this device.';
    return false;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(String(text).replaceAll('_', ' '));
  utterance.lang = 'en-IN';
  utterance.rate = 0.92;
  const voice = voices.find(v => v.voiceURI === sessionSettings.voiceURI)
    || voices.find(v => /^en-IN/i.test(v.lang)) || voices.find(v => v.default) || voices[0];
  utterance.voice = voice;
  utterance.onstart = () => {
    if ($('feedbackStatus')) { $('feedbackStatus').textContent = `Speaking with ${voice.name}.`; $('feedbackStatus').className = 'feedback-status ok'; }
    if ($('speechStatus')) $('speechStatus').textContent = `Ready · ${voice.name} (${voice.lang})`;
  };
  utterance.onend = () => { if ($('speechStatus')) $('speechStatus').textContent = `Ready · ${voice.name} (${voice.lang})`; };
  utterance.onerror = event => {
    $('feedbackStatus').textContent = `Speech failed: ${event.error || 'audio output unavailable'}`;
    $('feedbackStatus').className = 'feedback-status err';
    if ($('speechStatus')) $('speechStatus').textContent = `Speech failed: ${event.error || 'audio output unavailable'}`;
  };
  window.speechSynthesis.speak(utterance);
  return true;
}

function refreshSpeechVoices() {
  const select = $('ttsVoiceSelect');
  if (!select || !('speechSynthesis' in window)) {
    if ($('speechStatus')) $('speechStatus').textContent = 'Browser speech synthesis is unavailable.';
    return;
  }
  const voices = speechSynthesis.getVoices();
  select.innerHTML = '<option value="">Automatic English voice</option>' + voices.map(v =>
    `<option value="${escapeHtml(v.voiceURI)}">${escapeHtml(v.name)} · ${escapeHtml(v.lang)}${v.default ? ' · system default' : ''}</option>`).join('');
  select.value = sessionSettings.voiceURI || '';
  if ($('speechStatus')) $('speechStatus').textContent = voices.length
    ? `Ready · ${voices.length} browser voices found.` : 'Waiting for a system voice…';
}

async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); }
  catch (e) { throw new Error(`Network error calling ${path}: ${e.message}`); }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.text()) || detail; } catch (_) {}
    throw new Error(`${path} -> ${r.status}: ${detail}`);
  }
  return r.json();
}

function setHealth(ok, text) {
  $('healthPill').textContent = '● ' + text;
  $('healthPill').style.color = ok ? 'var(--good)' : 'var(--warn)';
}

/* ---------------- sidebar navigation ---------------- */
const VIEW_TITLES = {
  home: 'Recognition overview', upload: 'Live recognition', perSign: 'Per-sign analysis',
  perf: 'Model comparison', vocab: 'Supported signs', history: 'Recognition history', analytics: 'Session analytics',
  review: 'Error review', responsible: 'Responsible AI',
  settings: 'Recognition settings',
};
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  $('view-' + name).classList.add('active');
  document.querySelector(`.nav-item[data-view="${name}"]`).classList.add('active');
  $('viewTitle').textContent = VIEW_TITLES[name] || name;
  if (name === 'perf') renderPerf();
  if (name === 'perSign') renderPerSign();
  if (name === 'analytics') renderAnalytics();
  if (name === 'review') refreshReview();
  if (name === 'settings') renderSettings();
  if (name === 'history') refresh();
}
document.querySelectorAll('.nav-item').forEach(b => b.onclick = () => showView(b.dataset.view));
document.querySelectorAll('[data-open]').forEach(b => b.onclick = () => showView(b.dataset.open));

/* ---------------- model info / health ---------------- */
async function populateModels() {
  const sel = $('modelSelect');
  try {
    const info = await api('/model-info');
    modelInfoCache = info;
    pretrainedInfo = info.pretrained || {};
    sel.innerHTML = '';
    (info.available_models || []).forEach(name => {
      const p = pretrainedInfo[name] || {};
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name.toUpperCase() + (p.present ? '' : ' (not trained yet)');
      opt.disabled = !(p.loadable ?? p.present);
      if (name === info.default_model && !opt.disabled) opt.selected = true;
      sel.appendChild(opt);
    });
    const savedModel = localStorage.getItem('isl-recognition-model');
    if (savedModel && [...sel.options].some(o => o.value === savedModel && !o.disabled)) sel.value = savedModel;
    currentModel = sel.value || info.default_model;
    $('modelBadge').textContent = currentModel.toUpperCase();
    $('cameraModel').textContent = currentModel.toUpperCase();
    if (sessionSettings.threshold == null) sessionSettings.threshold = Number(info.confidence_threshold);
    $('threshold').textContent = Number(sessionSettings.threshold).toFixed(2);
    renderSettings();
    renderModelStats();
  } catch (e) {
    showError('Could not load /model-info: ' + e.message);
  }
}

function renderModelStats() {
  const el = $('modelStats');
  const summary = modelInfoCache?.evaluation?.summary_metrics || [];
  const labels = { lstm: 'LSTM', gru: 'GRU', gru_mha: 'GRU + MHA' };
  const markup = Object.entries(pretrainedInfo).map(([name, model]) => {
    const row = summary.find(r => r.model === name) || {};
    const ready = !!model.present;
    return `<div class="model-card${name === currentModel ? ' selected' : ''}">
      <h4>${labels[name] || name.toUpperCase()}${name === currentModel ? ' · ACTIVE' : ''}</h4>
      <div class="model-score">${row.accuracy != null ? `${(Number(row.accuracy) * 100).toFixed(1)}%` : '—'}</div>
      <div class="model-sub"><span>Macro F1<b>${row.macro_f1 != null ? (Number(row.macro_f1) * 100).toFixed(1) + '%' : '—'}</b></span>
      <span>Mean model latency<b>${row.latency_ms_mean != null ? Number(row.latency_ms_mean).toFixed(2) + ' ms' : '—'}</b></span>
      <span>P95 model latency<b>${row.latency_ms_p95 != null ? Number(row.latency_ms_p95).toFixed(2) + ' ms' : '—'}</b></span>
      <span>Throughput<b>${row.throughput_fps != null ? Number(row.throughput_fps).toFixed(1) + ' seq/s' : (ready ? 'measuring…' : 'not trained')}</b></span></div></div>`;
  }).join('') || '<p class="empty-state">No trained models are available.</p>';
  if (el) el.innerHTML = markup;
  if ($('homeModelTable')) $('homeModelTable').innerHTML = markup;
  const live = summary.find(r => r.model === currentModel);
  if (live) $('modelThroughput').textContent = `${Number(live.throughput_fps).toFixed(1)}`;
  else $('modelThroughput').textContent = '—';
}

async function switchModel(name) {
  const p = pretrainedInfo[name];
  if (!p || !p.present) { showError(`${name.toUpperCase()} has no trained model.keras yet.`); return; }
  try {
    if (stream && liveSessionId) {
      const switched = await api('/live/model', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: liveSessionId, model: name }) });
      sequenceLength = switched.sequence_length || sequenceLength;
    } else {
      await api('/models/switch?model=' + encodeURIComponent(name), { method: 'POST' });
    }
    currentModel = name;
    localStorage.setItem('isl-recognition-model', name);
    $('modelBadge').textContent = name.toUpperCase();
    $('cameraModel').textContent = name.toUpperCase();
    if (stream) {
      $('cameraState').textContent = 'Model changed · rebuilding temporal window';
      $('sequenceFrames').textContent = `0/${sequenceLength}`;
      $('resultStatus').textContent = 'Collecting a fresh sequence';
      $('sign').textContent = 'COLLECTING';
    }
    renderModelStats();
    clearError();
  } catch (e) { showError('Model switch failed: ' + e.message); }
}

async function boot() {
  try {
    const h = await api('/health');
    setHealth(h.model_loaded, h.model_loaded ? `System ready · ${h.model}` : `Degraded · ${h.reason || 'model unavailable'}`);
  } catch (e) {
    setHealth(false, 'Backend unavailable');
    showError('Backend unreachable at boot: ' + e.message);
  }
  await populateModels();
  refreshSpeechVoices();
  if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = refreshSpeechVoices;
  try { sequenceLength = (await api('/config')).sequence_length || sequenceLength; }
  catch (_) { /* health banner already explains a disconnected backend */ }
  try {
    const c = await api('/classes');
    vocabulary = c.classes || [];
    renderVocabulary();
    $('classesSummary').textContent = `Supported vocabulary (${vocabulary.length} classes)`;
    renderModelStats();
    renderPerSign();
    refresh();
  } catch (e) { showError('Could not load /classes: ' + e.message); }
}

function renderVocabulary() {
  const q = ($('vocabularySearch')?.value || '').trim().toLowerCase();
  const visible = vocabulary.filter(x => String(x).toLowerCase().includes(q));
  $('classes').innerHTML = visible.map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('')
    || '<p class="empty-state">No supported signs match this search.</p>';
}

/* ---------------- Model Performance view ---------------- */
function renderPerf() {
  renderModelStats();
  const el = $('evalTable');
  const rows = modelInfoCache?.evaluation?.summary_metrics;
  if (!rows || !rows.length) {
    el.innerHTML = '<p class="note">NOT YET EVALUATED — run <code>training/evaluate_models.py --protocol session_disjoint</code> to generate results/summary_metrics.csv.</p>';
    return;
  }
  const metrics = [
    ['Model', r => ({ lstm: 'LSTM', gru: 'GRU', gru_mha: 'GRU + MHA' }[r.model] || r.model)],
    ['Accuracy', r => `${(Number(r.accuracy) * 100).toFixed(1)}%`],
    ['Macro precision', r => `${(Number(r.macro_precision) * 100).toFixed(1)}%`],
    ['Macro recall', r => `${(Number(r.macro_recall) * 100).toFixed(1)}%`],
    ['Macro F1', r => `${(Number(r.macro_f1) * 100).toFixed(1)}%`],
    ['Mean latency', r => `${Number(r.latency_ms_mean).toFixed(2)} ms`],
    ['P95 latency', r => `${Number(r.latency_ms_p95).toFixed(2)} ms`],
    ['Model throughput', r => `${Number(r.throughput_fps).toFixed(1)} seq/s`],
    ['Test clips', r => `${r.n_samples} · ${r.protocol}`],
  ];
  el.innerHTML = `<table><thead><tr>${metrics.map(([name]) => `<th>${name}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${metrics.map(([,get]) => `<td>${get(r)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function renderPerSign() {
  const evaluation = modelInfoCache?.evaluation || {};
  perSignRows = evaluation.per_class || [];
  const models = ['lstm', 'gru', 'gru_mha'];
  const labels = { lstm: 'LSTM', gru: 'GRU', gru_mha: 'GRU + MHA' };
  const head = $('perSignHead'), body = $('perSignBody');
  if (head && body) {
    if (!perSignRows.length) {
      head.innerHTML = ''; body.innerHTML = '<tr><td>No per-sign evaluation yet. Run the reproducible evaluation command.</td></tr>';
    } else {
      head.innerHTML = `<tr><th rowspan="2">Sign</th><th rowspan="2">n</th>${models.map(m => `<th class="model-group" colspan="5">${labels[m]}</th>`).join('')}</tr><tr>${models.map(() => '<th>Precision</th><th>Recall</th><th>Accuracy*</th><th>F1</th><th>Latency</th>').join('')}</tr>`;
      const drawRows = filter => {
        const selected = perSignRows.filter(row => String(row.class).toLowerCase().includes(filter.toLowerCase()));
        body.innerHTML = selected.map(row => `<tr><td>${escapeHtml(row.class)}</td><td>${row.lstm_support || 0}</td>${models.map(m => {
          const p = Number(row[`${m}_precision`]), r = Number(row[`${m}_recall`]), f = Number(row[`${m}_f1`]);
          const ms = Number(row[`${m}_latency_ms_mean`]);
          const accuracy = Number(row[`${m}_accuracy`]);
          return `<td>${(p * 100).toFixed(0)}%</td><td>${(r * 100).toFixed(0)}%</td><td>${Number.isFinite(accuracy) ? (accuracy * 100).toFixed(0) + '%' : '—'}</td><td>${(f * 100).toFixed(0)}%</td><td>${Number.isFinite(ms) ? ms.toFixed(2) + ' ms' : '—'}</td>`;
        }).join('')}</tr>`).join('') || '<tr><td colspan="17">No signs match this filter.</td></tr>';
      };
      drawRows($('signFilter')?.value || '');
      if ($('signFilter')) $('signFilter').oninput = e => drawRows(e.target.value);
    }
  }
  const home = $('homePerSign');
  if (!home || !perSignRows.length) return;
  const avg = row => models.reduce((sum, m) => sum + Number(row[`${m}_recall`] || 0), 0) / models.length;
  const sorted = [...perSignRows].sort((a, b) => avg(a) - avg(b));
  const weak = sorted.slice(0, 5), strong = sorted.slice(-5).reverse();
  home.innerHTML = [...weak.map(r => ({ ...r, group: 'Needs review' })), ...strong.map(r => ({ ...r, group: 'High recall' }))]
    .map(r => `<div class="sign-tile"><strong>${escapeHtml(r.class)}</strong><b>${(avg(r) * 100).toFixed(0)}%</b><small>${r.group} · n=${r.lstm_support}</small></div>`).join('');
}

/* ---------------- Analytics view ---------------- */
async function renderAnalytics() {
  const el = $('liveMetrics');
  try {
    const m = await api('/metrics');
    if (!m.predictions_total) {
      el.innerHTML = '<p class="empty-state">No session data recorded yet. Start a camera session or process a video to populate operational metrics.</p>';
      return;
    }
    el.innerHTML = `
      <div class="stat"><strong>${m.predictions_total}</strong><span>Total predictions</span></div>
      <div class="stat"><strong>${(m.uncertain_rate * 100).toFixed(1)}%</strong><span>Uncertain rate</span></div>
      <div class="stat"><strong>${m.latency_ms.mean.toFixed(1)} ms</strong><span>Mean latency (p95 ${m.latency_ms.p95.toFixed(1)} ms)</span></div>
      <div class="stat"><strong>${m.throughput_fps.toFixed(1)}</strong><span>Model-only throughput (FPS)</span></div>
      <div class="stat"><strong>${m.feedback_total}</strong><span>Human feedback given</span></div>
      <div class="stat"><strong>${m.corrections_total}</strong><span>Corrections logged</span></div>
      <div class="stat"><strong>${m.human_actions.ACCEPT || 0}</strong><span>Accepted</span></div>
      <div class="stat"><strong>${m.human_actions.CORRECT || 0}</strong><span>Corrected</span></div>
      <div class="stat"><strong>${m.human_actions.REJECT || 0}</strong><span>Rejected</span></div>
      <div class="stat"><strong>${(m.status_counts.UNCERTAIN || 0) + (m.human_actions.REJECT || 0) + m.corrections_total}</strong><span>Review/error events</span></div>
      <div class="stat"><strong>${m.human_agreement_rate != null ? (m.human_agreement_rate * 100).toFixed(1) + '%' : '—'}</strong><span>Human agreement (Accept rate)</span></div>
      <div class="stat"><strong>${(m.confidence.mean * 100).toFixed(1)}%</strong><span>Mean confidence</span></div>`;
  } catch (e) {
    el.innerHTML = '<p class="note">Could not load /metrics.</p>';
    showError('Analytics: ' + e.message);
  }

  const rEl = $('robustnessTable');
  const rob = modelInfoCache && modelInfoCache.evaluation && modelInfoCache.evaluation.robustness;
  if (!rob) {
    rEl.innerHTML = '<p class="note">NOT YET EVALUATED — run <code>training/robustness.py --protocol session_disjoint</code> to generate results/robustness_summary.json.</p>';
    return;
  }
  try {
    const rows = Array.isArray(rob) ? rob : (rob.results || rob.conditions || []);
    if (!rows.length) { rEl.innerHTML = '<p class="note">Robustness file present but empty/unrecognised format.</p>'; return; }
    const cols = Object.keys(rows[0]);
    rEl.innerHTML = `<div class="table-wrap"><table><thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  } catch (e) {
    rEl.innerHTML = `<pre class="note">${JSON.stringify(rob, null, 2)}</pre>`;
  }
}

/* ---------------- Error Review view ---------------- */
async function refreshReview() {
  const el = $('reviewEvents');
  try {
    const rows = await api('/audit/events?limit=100');
    const flagged = rows.filter(r => r.status === 'UNCERTAIN' || (r.human_action && r.human_action !== 'ACCEPT'));
    const q = ($('reviewFilter')?.value || '').trim().toLowerCase();
    const filtered = flagged.filter(r => [r.predicted_class, r.model, r.note, r.corrected_class, r.source]
      .some(value => String(value || '').toLowerCase().includes(q)));
    el.innerHTML = filtered.map(r => `<tr>
      <td>${new Date(r.timestamp || Date.now()).toLocaleTimeString()}</td>
      <td>${r.predicted_class || '—'}</td>
      <td>${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—'}</td>
      <td class="status-${r.status || ''}">${r.status || '—'}</td>
      <td>${r.human_action || '—'}</td>
      <td>${r.corrected_class || '—'}</td>
      <td>${r.source || '—'}</td>
      <td>${escapeHtml(r.note || '—')}</td>
    </tr>`).join('') || '<tr><td colspan="8" class="note">Nothing flagged yet — all recent predictions were confident and accepted.</td></tr>';
    const failures = await api('/audit/failures?limit=100');
    $('systemFailures').innerHTML = failures.map(r => `<tr><td>${escapeHtml(r.timestamp || '—')}</td>
      <td>${escapeHtml(r.stage || '—')}</td><td>${escapeHtml(r.model || '—')}</td>
      <td>${escapeHtml(r.source || '—')}</td><td>${escapeHtml(r.message || '—')}</td></tr>`).join('')
      || '<tr><td colspan="5" class="note">No pipeline failures logged.</td></tr>';
  } catch (e) { showError('Error review: ' + e.message); }
}

/* ---------------- Settings view ---------------- */
function renderSettings() {
  if (!modelInfoCache) return;
  $('thresholdInput').value = String(sessionSettings.threshold ?? modelInfoCache.confidence_threshold);
  $('thresholdValue').textContent = Number($('thresholdInput').value).toFixed(2);
  $('smoothingInput').value = String(sessionSettings.smoothingWindow || 3);
  $('cameraSelect').value = sessionSettings.cameraDeviceId || '';
  refreshSpeechVoices();
}

async function refreshCameras() {
  const select = $('cameraSelect');
  if (!select || !navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const current = sessionSettings.cameraDeviceId;
    select.innerHTML = '<option value="">System default</option>' + devices.map((d, i) =>
      `<option value="${escapeHtml(d.deviceId)}">${escapeHtml(d.label || `Camera ${i + 1}`)}</option>`).join('');
    select.value = current && devices.some(d => d.deviceId === current) ? current : '';
  } catch (_) { /* the browser will still offer its default camera at session start */ }
}

async function saveSettings() {
  const previousCamera = sessionSettings.cameraDeviceId;
  sessionSettings = {
    ...sessionSettings,
    threshold: Number($('thresholdInput').value),
    smoothingWindow: Number($('smoothingInput').value),
    cameraDeviceId: $('cameraSelect').value,
    voiceURI: $('ttsVoiceSelect').value,
  };
  $('thresholdValue').textContent = sessionSettings.threshold.toFixed(2);
  $('threshold').textContent = sessionSettings.threshold.toFixed(2);
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(sessionSettings));
  try {
    if (stream && liveSessionId) {
      await api('/live/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: liveSessionId, threshold: sessionSettings.threshold,
          smoothing_window: sessionSettings.smoothingWindow }) });
      $('sequenceFrames').textContent = `0/${sequenceLength}`;
      $('sign').textContent = 'COLLECTING';
      $('resultStatus').textContent = 'Collecting a fresh sequence';
    }
    $('settingsStatus').textContent = `Saved · threshold ${sessionSettings.threshold.toFixed(2)} · smoothing ${sessionSettings.smoothingWindow} frame(s)`
      + (previousCamera !== sessionSettings.cameraDeviceId && stream ? ' · camera change applies next session' : '');
    $('settingsStatus').className = 'microcopy feedback-status ok';
  } catch (e) {
    $('settingsStatus').textContent = 'Could not apply settings: ' + e.message;
    $('settingsStatus').className = 'microcopy feedback-status err';
    showError('Settings failed: ' + e.message);
  }
}

/* ---------------- camera / capture ---------------- */
async function start() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access requires HTTPS or localhost and a browser that supports webcam capture.');
    }
    const videoConstraints = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (sessionSettings.cameraDeviceId) videoConstraints.deviceId = { exact: sessionSettings.cameraDeviceId };
    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    $('preview').srcObject = stream;
    await $('preview').play();
    await refreshCameras();
    const live = await api('/live/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: currentModel, threshold: sessionSettings.threshold,
        smoothing_window: sessionSettings.smoothingWindow }),
    });
    liveSessionId = live.session_id;
    $('videoOverlay').style.display = 'none';
    $('startBtn').disabled = true;
    $('stopBtn').disabled = false;
    $('cameraState').textContent = 'Camera active';
    $('recDot').style.display = 'block';
    $('topK').innerHTML = '';
    if (window.landmarkReplay) window.landmarkReplay.clear();
    frameCount = 0; frameStartedAt = performance.now();
    uncertaintyPrompted = false;
    $('frames').textContent = '0'; $('cameraFps').textContent = '—';
    $('serverLatency').textContent = '—'; $('modelLatency').textContent = '—';
    const countVideoFrame = (_now, metadata) => {
      if (!stream) return;
      frameCount++;
      const elapsed = (performance.now() - frameStartedAt) / 1000;
      if (elapsed > 0.5) $('cameraFps').textContent = (frameCount / elapsed).toFixed(1);
      $('frames').textContent = String(frameCount);
      videoFrameLoop = $('preview').requestVideoFrameCallback(countVideoFrame);
    };
    if ($('preview').requestVideoFrameCallback) videoFrameLoop = $('preview').requestVideoFrameCallback(countVideoFrame);
    startLiveLoop();
    clearError();
  } catch (e) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; liveSessionId = null; $('preview').srcObject = null;
    $('startBtn').disabled = false; $('stopBtn').disabled = true;
    $('cameraState').textContent = 'Camera idle';
    showError('Camera or live recognition could not start: ' + e.message);
  }
}

async function stop() {
  clearTimeout(liveTimer);
  liveTimer = null;
  const endingSession = liveSessionId;
  liveSessionId = null;
  if (endingSession) {
    try { await api('/live/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: endingSession }) }); } catch (_) {}
  }
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (window.landmarkReplay) window.landmarkReplay.clear();
  stream = null;
  $('preview').srcObject = null;
  if (videoFrameLoop && $('preview').cancelVideoFrameCallback) $('preview').cancelVideoFrameCallback(videoFrameLoop);
  if (inferenceFrameLoop && $('preview').cancelVideoFrameCallback) $('preview').cancelVideoFrameCallback(inferenceFrameLoop);
  inferenceFrameLoop = 0;
  $('startBtn').disabled = false;
  $('stopBtn').disabled = true;
  $('cameraState').textContent = 'Camera idle';
  $('recDot').style.display = 'none';
}

function startLiveLoop() {
  if (!frameCanvas) frameCanvas = document.createElement('canvas');
  const tick = async () => {
    if (!stream || !liveSessionId) return;
    try {
      const video = $('preview');
      if (!video.videoWidth || !video.videoHeight) throw new Error('Waiting for camera frames…');
      const roundTripStarted = performance.now();
      const scale = Math.min(1, 640 / video.videoWidth);
      frameCanvas.width = Math.round(video.videoWidth * scale);
      frameCanvas.height = Math.round(video.videoHeight * scale);
      frameCanvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
      const blob = await new Promise(resolve => frameCanvas.toBlob(resolve, 'image/jpeg', 0.78));
      if (!blob) throw new Error('Could not encode camera frame');
      const form = new FormData();
      form.append('session_id', liveSessionId);
      form.append('timestamp_ms', String(Math.round(performance.now() - frameStartedAt)));
      form.append('file', blob, 'camera-frame.jpg');
      const result = await api('/live/frame', { method: 'POST', body: form });
      if (!stream || !liveSessionId) return;
      const uiRenderStarted = performance.now();
      $('latency').textContent = `${(performance.now() - roundTripStarted).toFixed(1)} ms`;
      $('serverLatency').textContent = `${Number(result.processing_ms || 0).toFixed(1)} ms`;
      if (window.landmarkReplay) window.landmarkReplay.draw(result.landmarks, video);
      const state = result.landmark_state || {};
      const visible = [state.left_hand && 'left hand', state.right_hand && 'right hand', state.pose && 'pose'].filter(Boolean);
      $('trackBadge').textContent = visible.length ? `Detected: ${visible.join(' + ')}` : 'No hands/pose detected — move into frame and improve lighting';
      $('detectionState').textContent = visible.length ? visible.join(' + ') : 'No landmarks';
      $('processedFps').textContent = `${Number(result.camera_fps || 0).toFixed(1)} FPS`;
      $('sequenceFrames').textContent = `${Math.round((result.fill_ratio || 0) * sequenceLength)}/${sequenceLength}`;
      renderTopK(result.top_k || []);
      $('landmarkStep').classList.toggle('active', visible.length > 0);
      $('sequenceStep').classList.toggle('active', result.ready);
      $('decisionStep').classList.toggle('active', result.status === 'CONFIDENT');
      $('cameraState').textContent = result.ready
        ? `Live recognition · ${result.camera_fps} camera FPS`
        : `Collecting sequence · ${Math.round((result.fill_ratio || 0) * sequenceLength)}/${sequenceLength} frames`;
      if (!result.ready) {
        $('sign').textContent = 'COLLECTING';
        $('caption').textContent = `Building the ${sequenceLength}-frame sequence…`;
        $('resultStatus').textContent = 'Building temporal sequence';
        $('resultStatus').className = 'status-PENDING';
        $('confidence').textContent = '—';
        $('confidenceBar').style.width = '0%';
        $('speakResultBtn').disabled = true;
        uncertaintyPrompted = false;
      } else {
        $('sign').textContent = result.sign;
        $('confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
        $('confidenceBar').style.width = `${Math.max(0, Math.min(100, result.confidence * 100))}%`;
        $('resultStatus').textContent = result.status;
        $('resultStatus').className = `status-${result.status}`;
        $('modelLatency').textContent = `${Number(result.model_latency_ms || 0).toFixed(2)} ms`;
        $('modelThroughput').textContent = Number(result.model_latency_ms) > 0
          ? `${(1000 / Number(result.model_latency_ms)).toFixed(1)}` : '—';
        eventId = result.event_id || null;
        if (result.status === 'CONFIDENT') {
          $('caption').textContent = result.sign.replaceAll('_', ' ');
          uncertaintyPrompted = false;
        } else if (result.status === 'UNCERTAIN') {
          const best = (result.top_k || [])[0];
          const bestLabel = best?.class?.replaceAll('_', ' ');
          const bestPercent = best ? (best.probability * 100).toFixed(1) : null;
          $('caption').textContent = best
            ? `Low confidence · closest match ${bestLabel} (${bestPercent}%)`
            : 'Low confidence · no reliable match yet';
          if (!uncertaintyPrompted && $('speechToggle')?.checked) {
            const spokenHint = best
              ? `Not sure. Closest match: ${bestLabel}, ${bestPercent} percent. Please repeat the sign.`
              : 'Not sure. Please repeat the sign.';
            speakText(spokenHint);
            uncertaintyPrompted = true;
          }
        } else {
          $('caption').textContent = result.status === 'IDLE'
            ? 'No active sign detected.' : `Confirming the sign · ${result.streak || 0} stable update(s)`;
          if (result.status === 'IDLE') uncertaintyPrompted = false;
        }
        $('speakResultBtn').disabled = result.status !== 'CONFIDENT';
        if (result.status === 'CONFIDENT' && result.should_speak && $('speechToggle')?.checked) speakText(result.sign);
      }
      const timings = result.timings_ms || {};
      const stageValue = value => `${Number(value || 0).toFixed(2)} ms`;
      $('decodeMs').textContent = stageValue(timings.decode);
      $('mediaPipeMs').textContent = stageValue(timings.mediapipe_ms);
      $('preprocessingMs').textContent = stageValue(timings.preprocessing_ms);
      $('bufferMs').textContent = stageValue(timings.buffer_ms);
      $('stageModelMs').textContent = stageValue(timings.model);
      $('handGateMs').textContent = stageValue(timings.hand_gate);
      $('smoothingMs').textContent = stageValue(timings.smoothing);
      $('renderMs').textContent = stageValue(performance.now() - uiRenderStarted);
    } catch (e) {
      if (stream && liveSessionId) {
        $('cameraState').textContent = 'Live recognition paused';
        showError('Live frame failed: ' + e.message);
      }
    } finally {
      // Only schedule after this request finishes, and wake on the next actual
      // camera frame. This removes the old 75 ms idle pause without sending
      // duplicate frames or building a queue of stale images.
      if (stream && liveSessionId) {
        const video = $('preview');
        if (video.requestVideoFrameCallback) {
          inferenceFrameLoop = video.requestVideoFrameCallback(() => {
            inferenceFrameLoop = 0;
            tick();
          });
        } else {
          liveTimer = setTimeout(tick, 0);
        }
      }
    }
  };
  tick();
}

/* ---------------- prediction ---------------- */
async function recognize(blob, name = 'capture.webm') {
  clearError();
  $('resultStatus').textContent = 'Processing…';
  $('sign').textContent = '…';
  $('feedbackStatus').textContent = '';
  try {
    const fd = new FormData();
    fd.append('file', blob, name);
    fd.append('model', currentModel);
    // Speech comes from the user's browser, which can play it through their
    // speakers. Server-side speech would play on the API host instead.
    fd.append('speak', 'false');
    fd.append('log', 'true');
    const roundTripStarted = performance.now();
    const r = await api('/predict', { method: 'POST', body: fd });
    $('latency').textContent = `${(performance.now() - roundTripStarted).toFixed(1)} ms`;
    $('modelLatency').textContent = `${Number(r.latency_ms).toFixed(2)} ms`;
    $('serverLatency').textContent = '—';
    $('sign').textContent = r.sign;
    $('confidence').textContent = (r.confidence * 100).toFixed(1) + '%';
    $('resultStatus').textContent = r.status;
    $('resultStatus').className = 'status-' + r.status;
    $('caption').textContent = r.caption || r.sign;
    eventId = r.event_id || null;
    $('speakResultBtn').disabled = r.status !== 'CONFIDENT' || !r.sign || r.sign === 'UNCERTAIN';
    if (r.status === 'CONFIDENT' && $('speechToggle')?.checked) speakText(r.sign);
    if (!eventId) {
      $('feedbackStatus').textContent = 'Not logged — feedback buttons need a logged event.';
      $('feedbackStatus').className = 'feedback-status err';
    }
    renderTopK(r.top_k || []);
    if (window.landmarkReplay) {
      if (r.landmarks && r.landmarks.length) window.landmarkReplay.play(r.landmarks, $('preview'));
      else { window.landmarkReplay.clear(); showError('Backend did not return landmarks for this capture.'); }
    }
    return r;
  } catch (e) {
    $('resultStatus').textContent = 'Error';
    $('resultStatus').className = 'status-ERROR';
    $('sign').textContent = 'ERROR';
    $('caption').textContent = e.message;
    showError('Prediction failed: ' + e.message);
  }
}

function renderTopK(rows) {
  $('topK').innerHTML = rows.map(x => `<div class="bar"><span>${x.class}</span><div class="track"><div class="fill" style="width:${Math.min(100, x.probability * 100)}%"></div></div><b>${(x.probability * 100).toFixed(1)}%</b></div>`).join('');
}

async function feedback(action) {
  $('feedbackStatus').className = 'feedback-status';
  if (!eventId) {
    $('feedbackStatus').textContent = 'No logged prediction to give feedback on yet.';
    $('feedbackStatus').className = 'feedback-status err';
    return;
  }
  let corrected = null;
  if (action === 'correct') {
    corrected = prompt(`Enter the correct INCLUDE-50 class. Choose one of:\n${vocabulary.join(', ')}`);
    if (!corrected) return;
    corrected = corrected.trim().toUpperCase();
    if (!vocabulary.includes(corrected)) {
      $('feedbackStatus').textContent = 'Correction not saved. Choose a label from the supported vocabulary list.';
      $('feedbackStatus').className = 'feedback-status err';
      return;
    }
  }
  const note = action === 'accept' ? '' : (prompt('Optional review note (Cancel to leave blank):') || '');
  try {
    await api('/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_id: eventId, action: action.toUpperCase(),
        predicted_class: $('sign').textContent, corrected_class: corrected,
        confidence: parseFloat($('confidence').textContent) / 100, model: currentModel,
        note,
      }),
    });
    $('feedbackStatus').textContent = 'Feedback recorded — thank you.';
    $('feedbackStatus').className = 'feedback-status ok';
    refresh();
    refreshReview();
  } catch (e) {
    $('feedbackStatus').textContent = 'Feedback failed: ' + e.message;
    $('feedbackStatus').className = 'feedback-status err';
    showError('Feedback failed: ' + e.message);
  }
}

/* ---------------- upload ---------------- */
async function upload(e) {
  const f = e.target.files[0];
  if (!f) return;
  $('uploadResult').innerHTML = '<p class="note">Processing…</p>';
  const r = await recognize(f, f.name);
  if (r) {
    $('uploadResult').innerHTML = `<div class="card"><div class="sign">${r.sign}</div>
      <div class="caption">${(r.confidence * 100).toFixed(1)}% · ${r.status} · ${r.latency_ms} ms</div></div>`;
  }
}

$('speakResultBtn').onclick = () => {
  const sign = $('sign').textContent;
  if (sign && !['READY', 'ERROR', 'UNCERTAIN', 'IDLE', '…'].includes(sign)) speakText(sign);
};

/* ---------------- history + export ---------------- */
async function refresh() {
  try {
    const rows = await api('/audit/events?limit=25');
    $('events').innerHTML = rows.map(r => `<tr>
      <td>${new Date(r.timestamp || Date.now()).toLocaleTimeString()}</td>
      <td>${escapeHtml(r.predicted_class || '—')}</td>
      <td>${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—'}</td>
      <td>${escapeHtml(r.model || '—')}</td>
      <td>${r.latency_ms != null ? Number(r.latency_ms).toFixed(1) + ' ms' : '—'}</td>
      <td>${r.stable ? 'Yes' : 'No'}</td>
      <td class="status-${escapeHtml(r.status || '')}">${escapeHtml(r.status || '—')}</td>
      <td>${escapeHtml(r.human_action || '—')}</td>
      <td>${escapeHtml(r.corrected_class || '—')}</td>
      <td>${escapeHtml(r.session_id || '—')}</td>
    </tr>`).join('') || '<tr><td colspan="10" class="note">No events logged yet.</td></tr>';
    $('recentList').innerHTML = rows.slice(0, 5).map(r => `<div class="recent-row">
      <strong>${escapeHtml(r.predicted_class || '—')}</strong>
      <span>${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—'}</span>
      <span class="reviewed">${escapeHtml(r.human_action || r.status || 'Logged')}</span>
    </div>`).join('') || '<p class="empty-state">No predictions yet. Start the camera to create a session.</p>';
  } catch (e) { showError('Could not load /audit/events: ' + e.message); }
}

async function exportCsv() {
  try {
    const rows = await api('/audit/events?limit=1000');
    const cols = ['event_id', 'timestamp', 'session_id', 'predicted_class', 'raw_sign', 'confidence', 'raw_confidence', 'model', 'latency_ms', 'sequence_frames', 'threshold', 'stable', 'status', 'human_action', 'corrected_class', 'source', 'note'];
    const csv = [cols.join(',')].concat(
      rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(','))
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'isl_recognition_history.csv';
    a.click();
  } catch (e) { showError('Export failed: ' + e.message); }
}

async function exportJson() {
  try {
    const rows = await api('/audit/events?limit=1000');
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'isl_recognition_history.json'; a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showError('JSON export failed: ' + e.message); }
}

/* ---------------- wire up ---------------- */
$('startBtn').onclick = start;
$('stopBtn').onclick = stop;
$('resetBtn').onclick = async () => {
  if (!liveSessionId) {
    $('sign').textContent = 'READY'; $('resultStatus').textContent = 'Waiting for input';
    $('caption').textContent = 'Start the camera and perform one sign.';
    $('sequenceFrames').textContent = `0/${sequenceLength}`;
    $('confidence').textContent = '—'; $('confidenceBar').style.width = '0%';
    if (window.landmarkReplay) window.landmarkReplay.clear();
    return;
  }
  try {
    await api('/live/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: liveSessionId }) });
    eventId = null; $('sign').textContent = 'COLLECTING'; $('resultStatus').textContent = 'Sequence reset';
    $('caption').textContent = 'Sequence cleared. Perform one sign when ready.';
    $('sequenceFrames').textContent = `0/${sequenceLength}`; $('topK').innerHTML = '';
    $('confidence').textContent = '—'; $('confidenceBar').style.width = '0%';
    if (window.landmarkReplay) window.landmarkReplay.clear();
  } catch (e) { showError('Could not reset sequence: ' + e.message); }
};
$('fileInput').onchange = upload;
$('refreshBtn').onclick = refresh;
$('exportBtn').onclick = exportCsv;
$('exportJsonBtn').onclick = exportJson;
$('refreshReviewBtn').onclick = refreshReview;
$('reviewFilter').oninput = () => refreshReview();
$('vocabularySearch').oninput = renderVocabulary;
$('modelSelect').onchange = async e => {
  const requested = e.target.value;
  await switchModel(requested);
  if (currentModel !== requested) e.target.value = currentModel;
};
$('thresholdInput').oninput = e => $('thresholdValue').textContent = Number(e.target.value).toFixed(2);
$('saveSettingsBtn').onclick = saveSettings;
$('ttsVoiceSelect').onchange = e => { sessionSettings.voiceURI = e.target.value; localStorage.setItem(SETTINGS_KEY, JSON.stringify(sessionSettings)); refreshSpeechVoices(); };
$('cameraSelect').onchange = e => { sessionSettings.cameraDeviceId = e.target.value; };
refreshCameras();
document.querySelectorAll('#feedback button').forEach(b => b.onclick = () => feedback(b.dataset.action));
window.addEventListener('pagehide', () => {
  if (liveSessionId) navigator.sendBeacon('/live/stop', new Blob(
    [JSON.stringify({ session_id: liveSessionId })], { type: 'application/json' }));
});
boot();
