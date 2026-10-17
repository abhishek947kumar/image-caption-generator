/**
 * Neural Caption Studio - Frontend Application
 * Interactivity, API integration, Audio TTS, and Visualization
 */

// Application State
const state = {
  currentFile: null,
  selectedSampleId: null,
  strategy: 'beam',
  beamWidth: 3,
  temperature: 1.0,
  maxLength: 25,
  isGenerating: false,
  lastResult: null,
  speechSynth: window.speechSynthesis || null,
  isSpeaking: false
};

// DOM Elements Cache
const elements = {
  tabs: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  dropzone: document.getElementById('image-dropzone'),
  fileInput: document.getElementById('file-input'),
  dropzonePrompt: document.getElementById('dropzone-prompt'),
  previewContainer: document.getElementById('preview-container'),
  imgPreview: document.getElementById('img-preview'),
  previewBadgeName: document.getElementById('preview-badge-name'),
  previewBadgeDims: document.getElementById('preview-badge-dims'),
  btnClearImage: document.getElementById('btn-clear-image'),
  presetGallery: document.getElementById('preset-gallery'),
  btnStrategyBeam: document.getElementById('btn-strategy-beam'),
  btnStrategyGreedy: document.getElementById('btn-strategy-greedy'),
  groupBeamWidth: document.getElementById('group-beam-width'),
  sliderBeamWidth: document.getElementById('slider-beam-width'),
  valBeamWidth: document.getElementById('val-beam-width'),
  sliderTemp: document.getElementById('slider-temp'),
  valTemp: document.getElementById('val-temp'),
  sliderMaxLen: document.getElementById('slider-max-len'),
  valMaxLen: document.getElementById('val-max-len'),
  btnGenerate: document.getElementById('btn-generate'),
  txtCaption: document.getElementById('txt-generated-caption'),
  badgeStrategyName: document.getElementById('badge-strategy-name'),
  metricConfidence: document.getElementById('metric-confidence'),
  metricLatency: document.getElementById('metric-latency'),
  metricTokensCount: document.getElementById('metric-tokens-count'),
  tokensContainer: document.getElementById('tokens-container'),
  candidateInspector: document.getElementById('candidate-inspector'),
  inspectorWord: document.getElementById('inspector-word'),
  inspectorBars: document.getElementById('inspector-bars'),
  btnCloseInspector: document.getElementById('btn-close-inspector'),
  cardBeamCandidates: document.getElementById('card-beam-candidates'),
  tbodyBeamCandidates: document.getElementById('tbody-beam-candidates'),
  conceptBarsContainer: document.getElementById('concept-bars-container'),
  cardPresetBleu: document.getElementById('card-preset-bleu'),
  valBleu1: document.getElementById('val-bleu-1'),
  valBleu2: document.getElementById('val-bleu-2'),
  valBleu3: document.getElementById('val-bleu-3'),
  valBleu4: document.getElementById('val-bleu-4'),
  listReferences: document.getElementById('list-references'),
  btnTts: document.getElementById('btn-tts'),
  btnCopyCaption: document.getElementById('btn-copy-caption'),
  toast: document.getElementById('app-toast'),
  // Eval Tab
  txtEvalCandidate: document.getElementById('txt-eval-candidate'),
  txtEvalReferences: document.getElementById('txt-eval-references'),
  btnRunEval: document.getElementById('btn-run-eval'),
  evalBleu1: document.getElementById('eval-bleu-1'),
  evalBleu2: document.getElementById('eval-bleu-2'),
  evalBleu3: document.getElementById('eval-bleu-3'),
  evalBleu4: document.getElementById('eval-bleu-4'),
  barBleu1: document.getElementById('bar-bleu-1'),
  barBleu2: document.getElementById('bar-bleu-2'),
  barBleu3: document.getElementById('bar-bleu-3'),
  barBleu4: document.getElementById('bar-bleu-4'),
  evalBp: document.getElementById('eval-bp'),
  evalLens: document.getElementById('eval-lens')
};

// Toast Notifications
function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  setTimeout(() => {
    elements.toast.classList.remove('show');
  }, 3000);
}

// Initialize Application
async function initApp() {
  bindEventListeners();
  await checkHealthAndModelInfo();
  await loadPresets();
}

// Event Listeners Binding
function bindEventListeners() {
  // Navigation Tabs
  elements.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      elements.tabs.forEach(t => t.classList.remove('active'));
      elements.tabContents.forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const targetId = `tab-${tab.dataset.tab}`;
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add('active');
    });
  });

  // Image Upload Dropzone
  elements.dropzone.addEventListener('click', (e) => {
    if (e.target.closest('#btn-clear-image')) return;
    elements.fileInput.click();
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.dropzone.classList.remove('drag-over');
    });
  });

  elements.dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Clear Image
  elements.btnClearImage.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedImage();
  });

  // Strategy Toggle
  elements.btnStrategyBeam.addEventListener('click', () => {
    setStrategy('beam');
  });
  elements.btnStrategyGreedy.addEventListener('click', () => {
    setStrategy('greedy');
  });

  // Sliders
  elements.sliderBeamWidth.addEventListener('input', (e) => {
    state.beamWidth = parseInt(e.target.value);
    elements.valBeamWidth.textContent = state.beamWidth;
  });

  elements.sliderTemp.addEventListener('input', (e) => {
    state.temperature = parseFloat(e.target.value);
    elements.valTemp.textContent = state.temperature.toFixed(1);
  });

  elements.sliderMaxLen.addEventListener('input', (e) => {
    state.maxLength = parseInt(e.target.value);
    elements.valMaxLen.textContent = state.maxLength;
  });

  // Generate Button
  elements.btnGenerate.addEventListener('click', triggerCaptionGeneration);

  // Inspector Close
  elements.btnCloseInspector.addEventListener('click', () => {
    elements.candidateInspector.style.display = 'none';
  });

  // Audio Speech Synthesis
  elements.btnTts.addEventListener('click', handleTtsClick);

  // Copy Caption
  elements.btnCopyCaption.addEventListener('click', () => {
    const text = elements.txtCaption.textContent;
    if (text && text !== 'Select an image or sample preset to generate its deep learning caption.') {
      navigator.clipboard.writeText(text);
      showToast('Caption copied to clipboard!');
    }
  });

  // BLEU Lab Button
  elements.btnRunEval.addEventListener('click', runBleuLabEvaluation);

  // Keyboard shortcut: Ctrl + Enter
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      triggerCaptionGeneration();
    }
  });
}

function setStrategy(strat) {
  state.strategy = strat;
  if (strat === 'beam') {
    elements.btnStrategyBeam.classList.add('active');
    elements.btnStrategyGreedy.classList.remove('active');
    elements.groupBeamWidth.style.display = 'block';
    if (state.lastResult && state.lastResult.beam_candidates && state.lastResult.beam_candidates.length > 0) {
      elements.cardBeamCandidates.style.display = 'block';
    }
  } else {
    elements.btnStrategyGreedy.classList.add('active');
    elements.btnStrategyBeam.classList.remove('active');
    elements.groupBeamWidth.style.display = 'none';
    elements.cardBeamCandidates.style.display = 'none';
  }
}

function handleFileSelected(file) {
  state.currentFile = file;
  state.selectedSampleId = null;

  // Clear selected preset highlight
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));

  const reader = new FileReader();
  reader.onload = (e) => {
    elements.imgPreview.src = e.target.result;
    elements.previewBadgeName.textContent = file.name;
    elements.dropzonePrompt.style.display = 'none';
    elements.previewContainer.style.display = 'flex';

    // Get natural image dimensions
    const img = new Image();
    img.onload = () => {
      elements.previewBadgeDims.textContent = `${img.naturalWidth}x${img.naturalHeight}`;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);

  showToast(`Loaded ${file.name}`);
}

function clearSelectedImage() {
  state.currentFile = null;
  state.selectedSampleId = null;
  elements.fileInput.value = '';
  elements.imgPreview.src = '';
  elements.dropzonePrompt.style.display = 'block';
  elements.previewContainer.style.display = 'none';
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
  elements.cardPresetBleu.style.display = 'none';
}

// Backend Health and Info
async function checkHealthAndModelInfo() {
  try {
    const [healthRes, infoRes] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/model-info')
    ]);
    const health = await healthRes.json();
    const info = await infoRes.json();

    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('txt-backend-status');
    const txtBackbone = document.getElementById('txt-backbone');

    if (health.status === 'online') {
      statusText.textContent = `Online (${health.device.toUpperCase()})`;
      statusDot.style.background = '#10b981';
      txtBackbone.textContent = health.backbone === 'mobilenet_v3_small' ? 'MobileNetV3' : 'ResNet50';
    }

    // Populate Architecture Specs
    document.getElementById('spec-backbone').textContent = info.backbone;
    document.getElementById('spec-feat-dim').textContent = `${info.feature_dim}-D`;
    document.getElementById('spec-embed-dim').textContent = `${info.embed_dim}-D`;
    document.getElementById('spec-hidden-dim').textContent = `${info.hidden_dim}-D`;
    document.getElementById('spec-vocab-size').textContent = `${info.vocab_size.toLocaleString()} tokens`;
    document.getElementById('spec-params').textContent = `${(info.total_parameters / 1e6).toFixed(2)}M`;
  } catch (err) {
    console.error('Failed to connect to backend:', err);
    document.getElementById('txt-backend-status').textContent = 'Backend Offline';
    document.querySelector('.status-dot').style.background = '#f43f5e';
  }
}

// Load Preset Images
async function loadPresets() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    elements.presetGallery.innerHTML = '';

    data.samples.forEach((sample, idx) => {
      const card = document.createElement('div');
      card.className = 'preset-card';
      card.id = `preset-${sample.id}`;
      card.innerHTML = `
        <img src="${sample.image_url}" alt="${sample.title}">
        <div class="preset-overlay-label">${sample.title}</div>
      `;
      card.addEventListener('click', () => {
        selectPreset(sample);
      });
      elements.presetGallery.appendChild(card);

      // Auto-select the first preset on startup for immediate WOW factor!
      if (idx === 0) {
        selectPreset(sample);
      }
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

function selectPreset(sample) {
  state.currentFile = null;
  state.selectedSampleId = sample.id;

  // Highlight selected preset
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
  const card = document.getElementById(`preset-${sample.id}`);
  if (card) card.classList.add('selected');

  // Load preview
  elements.imgPreview.src = sample.image_url;
  elements.previewBadgeName.textContent = sample.filename;
  elements.dropzonePrompt.style.display = 'none';
  elements.previewContainer.style.display = 'flex';

  const img = new Image();
  img.onload = () => {
    elements.previewBadgeDims.textContent = `${img.naturalWidth}x${img.naturalHeight}`;
  };
  img.src = sample.image_url;

  // Automatically trigger caption generation on preset selection
  triggerCaptionGeneration();
}

// Generate Caption API Call
async function triggerCaptionGeneration() {
  if (state.isGenerating) return;
  if (!state.currentFile && !state.selectedSampleId) {
    showToast('Please select or upload an image first');
    return;
  }

  state.isGenerating = true;
  elements.btnGenerate.disabled = true;
  elements.btnGenerate.classList.add('loading');
  elements.btnGenerate.querySelector('.btn-text').textContent = 'Analyzing & Decoding...';
  elements.txtCaption.textContent = 'Neural network is analyzing visual features...';
  elements.txtCaption.style.opacity = '0.5';

  const formData = new FormData();
  if (state.currentFile) {
    formData.append('file', state.currentFile);
  } else if (state.selectedSampleId) {
    formData.append('sample_id', state.selectedSampleId);
  }

  formData.append('strategy', state.strategy);
  formData.append('beam_width', state.beamWidth);
  formData.append('temperature', state.temperature);
  formData.append('max_length', state.maxLength);

  try {
    const res = await fetch('/api/caption', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference error');
    }

    const data = await res.json();
    state.lastResult = data;
    renderCaptionResult(data);
  } catch (err) {
    console.error('Caption generation failed:', err);
    elements.txtCaption.textContent = `Error: ${err.message}`;
    showToast(`Generation failed: ${err.message}`);
  } finally {
    state.isGenerating = false;
    elements.btnGenerate.disabled = false;
    elements.btnGenerate.classList.remove('loading');
    elements.btnGenerate.querySelector('.btn-text').textContent = 'Generate Caption';
    elements.txtCaption.style.opacity = '1';
  }
}

// Render Results
function renderCaptionResult(data) {
  // 1. Hero Caption text with typewriter / reveal animation
  elements.txtCaption.textContent = data.caption;
  elements.badgeStrategyName.textContent = data.strategy === 'beam' 
    ? `Beam Search (K=${data.beam_width})` 
    : 'Greedy Search (Argmax)';

  // 2. Metrics Bar
  const confPct = Math.round((data.average_confidence || 0.85) * 100);
  elements.metricConfidence.textContent = `${confPct}%`;
  elements.metricLatency.textContent = `${data.latency_ms || 42} ms`;
  elements.metricTokensCount.textContent = `${data.tokens ? data.tokens.length : 0} tokens`;

  // 3. Token-by-Token Chips
  elements.tokensContainer.innerHTML = '';
  if (data.tokens && data.tokens.length > 0) {
    data.tokens.forEach(tok => {
      const chip = document.createElement('div');
      const conf = tok.confidence || 0.8;
      const confClass = conf >= 0.8 ? 'conf-high' : (conf >= 0.5 ? 'conf-med' : 'conf-low');
      chip.className = `token-chip ${confClass}`;
      chip.innerHTML = `
        <span>${tok.word}</span>
        <span class="token-prob">${Math.round(conf * 100)}%</span>
      `;
      chip.addEventListener('click', () => {
        showCandidateInspector(tok);
      });
      elements.tokensContainer.appendChild(chip);
    });
  }

  // 4. Beam Hypotheses (if in beam search)
  if (data.strategy === 'beam' && data.beam_candidates && data.beam_candidates.length > 0) {
    elements.cardBeamCandidates.style.display = 'block';
    elements.tbodyBeamCandidates.innerHTML = '';
    data.beam_candidates.forEach(beam => {
      const tr = document.createElement('tr');
      if (beam.is_selected) tr.className = 'selected-beam';
      tr.innerHTML = `
        <td>#${beam.rank} ${beam.is_selected ? '★' : ''}</td>
        <td>${beam.caption}</td>
        <td><code>${beam.log_prob}</code></td>
        <td><code>${beam.normalized_score}</code></td>
      `;
      elements.tbodyBeamCandidates.appendChild(tr);
    });
  } else {
    elements.cardBeamCandidates.style.display = 'none';
  }

  // 5. CNN Visual Concepts
  elements.conceptBarsContainer.innerHTML = '';
  if (data.metadata && data.metadata.top_concepts) {
    data.metadata.top_concepts.forEach(c => {
      const item = document.createElement('div');
      item.className = 'concept-item';
      const pct = Math.round(c.confidence * 100);
      item.innerHTML = `
        <span class="concept-name">${c.concept.replace(/_/g, ' ')}</span>
        <div class="concept-bar-track">
          <div class="concept-bar-val" style="width: ${Math.max(pct, 5)}%"></div>
        </div>
        <span class="concept-confidence">${pct}%</span>
      `;
      elements.conceptBarsContainer.appendChild(item);
    });
  }

  // 6. Flickr8k Reference BLEU Score Card (if available)
  if (data.bleu_evaluation) {
    elements.cardPresetBleu.style.display = 'block';
    const b = data.bleu_evaluation;
    elements.valBleu1.textContent = (b.bleu_1 * 100).toFixed(1);
    elements.valBleu2.textContent = (b.bleu_2 * 100).toFixed(1);
    elements.valBleu3.textContent = (b.bleu_3 * 100).toFixed(1);
    elements.valBleu4.textContent = (b.bleu_4 * 100).toFixed(1);

    elements.listReferences.innerHTML = '';
    if (data.reference_captions) {
      data.reference_captions.forEach(ref => {
        const li = document.createElement('li');
        li.textContent = ref;
        elements.listReferences.appendChild(li);
      });
    }

    // Also populate BLEU Lab with current values
    elements.txtEvalCandidate.value = data.caption;
    if (data.reference_captions) {
      elements.txtEvalReferences.value = data.reference_captions.join('\n');
    }
  } else {
    elements.cardPresetBleu.style.display = 'none';
  }
}

// Show Token Candidate Inspector
function showCandidateInspector(token) {
  elements.candidateInspector.style.display = 'block';
  elements.inspectorWord.textContent = `"${token.word}" (Token ID: ${token.token_id})`;
  elements.inspectorBars.innerHTML = '';

  const candidates = token.top_candidates || [
    { word: token.word, probability: token.confidence || 0.9 },
    { word: 'the', probability: 0.05 },
    { word: 'a', probability: 0.03 }
  ];

  candidates.forEach(cand => {
    const row = document.createElement('div');
    row.className = 'candidate-row';
    const pct = Math.round(cand.probability * 100);
    row.innerHTML = `
      <span>${cand.word}</span>
      <div class="candidate-bar-bg">
        <div class="candidate-bar-fill" style="width: ${Math.max(pct, 3)}%"></div>
      </div>
      <span>${pct}%</span>
    `;
    elements.inspectorBars.appendChild(row);
  });
}

// Text to Speech
function handleTtsClick() {
  if (!state.speechSynth) {
    showToast('Text-to-speech is not supported on this browser');
    return;
  }

  if (state.isSpeaking) {
    state.speechSynth.cancel();
    state.isSpeaking = false;
    elements.btnTts.classList.remove('speaking');
    return;
  }

  const text = elements.txtCaption.textContent;
  if (!text || text === 'Select an image or sample preset to generate its deep learning caption.') return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    state.isSpeaking = true;
    elements.btnTts.classList.add('speaking');
  };

  utterance.onend = utterance.onerror = () => {
    state.isSpeaking = false;
    elements.btnTts.classList.remove('speaking');
  };

  state.speechSynth.speak(utterance);
}

// BLEU Lab Evaluation
async function runBleuLabEvaluation() {
  const candidate = elements.txtEvalCandidate.value.trim();
  const rawRefs = elements.txtEvalReferences.value.trim();

  if (!candidate) {
    showToast('Please enter a candidate caption to evaluate');
    return;
  }
  const references = rawRefs.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  if (references.length === 0) {
    showToast('Please provide at least one reference caption');
    return;
  }

  elements.btnRunEval.disabled = true;
  elements.btnRunEval.querySelector('span:last-child').textContent = 'Calculating...';

  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate, references })
    });
    const data = await res.json();

    // Render BLEU bars
    const b1 = Math.round(data.bleu_1 * 100);
    const b2 = Math.round(data.bleu_2 * 100);
    const b3 = Math.round(data.bleu_3 * 100);
    const b4 = Math.round(data.bleu_4 * 100);

    elements.evalBleu1.textContent = (data.bleu_1 * 100).toFixed(1);
    elements.evalBleu2.textContent = (data.bleu_2 * 100).toFixed(1);
    elements.evalBleu3.textContent = (data.bleu_3 * 100).toFixed(1);
    elements.evalBleu4.textContent = (data.bleu_4 * 100).toFixed(1);

    elements.barBleu1.style.width = `${b1}%`;
    elements.barBleu2.style.width = `${b2}%`;
    elements.barBleu3.style.width = `${b3}%`;
    elements.barBleu4.style.width = `${b4}%`;

    elements.evalBp.textContent = data.brevity_penalty.toFixed(3);
    elements.evalLens.textContent = `${data.candidate_length} words vs ${data.reference_length} words (Ref)`;

    showToast('BLEU scores calculated!');
  } catch (err) {
    console.error('BLEU calculation failed:', err);
    showToast('BLEU evaluation failed');
  } finally {
    elements.btnRunEval.disabled = false;
    elements.btnRunEval.querySelector('span:last-child').textContent = 'Compute BLEU Metrics';
  }
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', initApp);
