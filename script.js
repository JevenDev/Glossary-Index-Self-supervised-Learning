// Machine-generated pseudo-labeling demo powered by ml5.js + MobileNet

const fileInput = document.getElementById('file');
const btnClearImages = document.getElementById('btn-clear-images');
const btnDefaultImages = document.getElementById('btn-default-images');
const btnLabelRandom = document.getElementById('btn-label-random');
const btnLabelAll = document.getElementById('btn-label-all');

const statusEl = document.getElementById('status');
const curimgEl = document.getElementById('curimg');
const imgsStatus = document.getElementById('imgs-status');
const thumbs = document.getElementById('thumbs');
const resultsEl = document.getElementById('results');
const logEl = document.getElementById('log');

const preview = document.getElementById('preview');
const previewCtx = preview.getContext('2d');

// hidden element for image loading
const img = document.getElementById('img');

const MAX_RESULTS = 5;

let classifier = null;
let imageEntries = [];
let labeling = false;
let bulkLabeling = false;
let modelReady = false;
let currentIndex = -1;

initUI();

(async function init() {
    try {
        statusEl.textContent = 'Loading ml5 + MobileNet…';

        await prepareTensorflowBackend();

        classifier = await ml5.imageClassifier('MobileNet');
        modelReady = true;
        statusEl.textContent = 'Model ready. Load images then request machine-generated labels.';
        updateControls();
    } catch (err) {
        console.error(err);
        statusEl.textContent = `Failed to load ml5 MobileNet. ${err?.message || ''}`.trim();
        disableLabelControls(true);
    }
})();

fileInput.onchange = () => {
    clearImages({ quiet: true });

    const files = Array.from(fileInput.files || []);
    if (!files.length) {
        imgsStatus.textContent = 'No images yet.';
        updateControls();
        return;
    }

    files.forEach(f => {
        const url = URL.createObjectURL(f);
        addImageEntry({
            url,
            name: f.name,
            objectURL: true
        });
    });

    imgsStatus.textContent = `Loaded ${files.length} image${files.length === 1 ? '' : 's'}. Click a thumbnail or let the model pick at random.`;
    updateControls();
};

btnDefaultImages.onclick = () => {
    clearImages({ quiet: true });

    for (let i = 1; i <= 10; i++) {
        const name = `cat${i}.png`;
        addImageEntry({
            url: `images/${name}`,
            name,
            objectURL: false
        });
    }

    imgsStatus.textContent = 'Loaded 10 default cat images. Ask the network to describe them or click a thumbnail.';
    updateControls();
};

btnClearImages.onclick = () => {
    clearImages();
    fileInput.value = '';
};

btnLabelRandom.onclick = () => {
    if (!imageEntries.length) return;
    const idx = randomIndex();
    labelImage(idx);
};

btnLabelAll.onclick = async () => {
    if (!modelReady || !imageEntries.length || bulkLabeling) return;
    bulkLabeling = true;
    updateControls();

    for (let i = 0; i < imageEntries.length; i++) {
        await labelImage(i, { force: true });
    }

    bulkLabeling = false;
    statusEl.textContent = 'Finished labeling every image currently loaded.';
    updateControls();
};

function initUI() {
    setLogEmpty();
    resetPreview();
    renderResults();
    updateControls();
}

function updateControls() {
    const hasImages = imageEntries.length > 0;
    btnClearImages.disabled = !hasImages;

    const allowLabeling = modelReady && hasImages && !labeling && !bulkLabeling;
    btnLabelRandom.disabled = !allowLabeling;
    btnLabelAll.disabled = !modelReady || !hasImages || labeling || bulkLabeling;
}

function disableLabelControls(disabled) {
    btnLabelRandom.disabled = disabled;
    btnLabelAll.disabled = disabled;
}

function addImageEntry(entry) {
    imageEntries.push(entry);
    const idx = imageEntries.length - 1;

    const thumb = document.createElement('img');
    setImageSource(thumb, entry.url);
    thumb.title = entry.name;
    thumb.dataset.index = String(idx);
    thumb.addEventListener('click', () => {
        labelImage(Number(thumb.dataset.index));
    });

    entry.thumb = thumb;
    thumbs.appendChild(thumb);
}

async function labelImage(index, { force = false } = {}) {
    if (!modelReady || labeling || (bulkLabeling && !force)) return;
    const entry = imageEntries[index];
    if (!entry) return;

    labeling = true;
    updateControls();
    highlightThumb(index);

    try {
        statusEl.textContent = `Loading "${entry.name}" into the network…`;
        await loadImageInto(img, entry.url);
        drawToPreview(img);
        curimgEl.textContent = entry.name || '(blob)';

        statusEl.textContent = 'Generating machine labels…';
        const results = await classifier.classify(img);
        renderResults(results);
        appendLog(entry, results);

        if (results && results[0]) {
            statusEl.textContent = `Top label for "${entry.name}" -> ${formatResult(results[0])}`;
        } else {
            statusEl.textContent = `No confident label for "${entry.name}".`;
        }

        currentIndex = index;
    } catch (err) {
        console.error(err);
        statusEl.textContent = `Failed to label "${entry?.name || 'image'}".`;
    } finally {
        labeling = false;
        updateControls();
    }
}

function renderResults(results = []) {
    resultsEl.innerHTML = '';

    if (!results.length) {
        const li = document.createElement('li');
        li.textContent = 'No predictions yet.';
        resultsEl.appendChild(li);
        return;
    }

    const sliced = results.slice(0, MAX_RESULTS);
    sliced.forEach(res => {
        const li = document.createElement('li');
        li.textContent = formatResult(res);
        resultsEl.appendChild(li);
    });
}

function appendLog(entry, results = []) {
    if (!results.length) return;

    if (logEl.firstElementChild && logEl.firstElementChild.classList.contains('log-empty')) {
        logEl.innerHTML = '';
    }

    const li = document.createElement('li');
    const title = document.createElement('div');
    title.className = 'log-title';
    title.textContent = entry.name || '(blob)';
    li.appendChild(title);

    const primary = document.createElement('div');
    primary.textContent = `top: ${formatResult(results[0])}`;
    li.appendChild(primary);

    const alt = results.slice(1, 3);
    if (alt.length) {
        const altLine = document.createElement('div');
        altLine.textContent = `alt: ${alt.map(formatResult).join(', ')}`;
        li.appendChild(altLine);
    }

    logEl.prepend(li);
}

function clearImages(options = {}) {
    const { quiet = false } = options;

    imageEntries.forEach(entry => {
        if (entry.objectURL) {
            URL.revokeObjectURL(entry.url);
        }
    });
    imageEntries = [];
    thumbs.innerHTML = '';
    currentIndex = -1;

    resetPreview();
    renderResults();
    setLogEmpty();
    curimgEl.textContent = 'N/A';

    if (!quiet) {
        imgsStatus.textContent = 'Removed all images.';
    }

    updateControls();
}

function setLogEmpty(message = 'No pseudo-labels yet. Ask the model to describe an image.') {
    logEl.innerHTML = '';
    const li = document.createElement('li');
    li.className = 'log-empty';
    li.textContent = message;
    logEl.appendChild(li);
}

function highlightThumb(index) {
    imageEntries.forEach((entry, idx) => {
        if (!entry.thumb) return;
        entry.thumb.classList.toggle('active', idx === index);
    });
}

function resetPreview() {
    previewCtx.save();
    previewCtx.clearRect(0, 0, preview.width, preview.height);
    previewCtx.fillStyle = '#0b0d1c';
    previewCtx.fillRect(0, 0, preview.width, preview.height);
    previewCtx.restore();
}

function drawToPreview(source) {
    const w = preview.width;
    const h = preview.height;

    previewCtx.save();
    previewCtx.clearRect(0, 0, w, h);
    previewCtx.fillStyle = '#0b0d1c';
    previewCtx.fillRect(0, 0, w, h);

    const iw = source.naturalWidth || source.width;
    const ih = source.naturalHeight || source.height;
    const scale = Math.min(w / iw, h / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    const dx = (w - dw) / 2;
    const dy = (h - dh) / 2;

    previewCtx.drawImage(source, 0, 0, iw, ih, dx, dy, dw, dh);
    previewCtx.restore();
}

function loadImageInto(el, url) {
    return new Promise((resolve, reject) => {
        el.onload = () => resolve();
        el.onerror = err => reject(err);
        setImageSource(el, url);
    });
}

function formatResult(res) {
    const conf = (res.confidence * 100).toFixed(1);
    return `${res.label} (${conf}%)`;
}

function randomIndex() {
    return Math.floor(Math.random() * imageEntries.length);
}

async function prepareTensorflowBackend() {
    const tf = window.tf;
    if (!tf || typeof tf.ready !== 'function') return;

    try {
        const shouldUseCPU = typeof window.location !== 'undefined'
            && window.location.protocol === 'file:'
            && typeof tf.setBackend === 'function'
            && tf.getBackend?.() !== 'cpu';

        if (shouldUseCPU) {
            await tf.setBackend('cpu');
        }

        await tf.ready();
    } catch (err) {
        console.warn('TensorFlow.js backend setup issue:', err);
    }
}

function setImageSource(el, url) {
    if (shouldUseCors(url)) {
        el.crossOrigin = 'anonymous';
    } else {
        el.removeAttribute('crossorigin');
    }
    el.src = url;
}

function shouldUseCors(url) {
    if (typeof window === 'undefined') return false;
    const protocol = window.location?.protocol;
    if (!protocol || protocol === 'file:') return false;
    if (url.startsWith('data:') || url.startsWith('blob:')) return false;
    return true;
}
