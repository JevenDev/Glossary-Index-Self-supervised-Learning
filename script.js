// self-supervised rotation prediction
// using TensorFlow.js, MobileNet, and KNN classifier

// DOM elements
const fileInput = document.getElementById('file');
const btnClearImages = document.getElementById('btn-clear-images');
const btnDefaultImages = document.getElementById('btn-default-images');

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnResetStats = document.getElementById('btn-reset-stats');
const btnClearModel = document.getElementById('btn-clear-model');
const delayInput = document.getElementById('delay');

const statusEl = document.getElementById('status');
const itEl = document.getElementById('it');
const okEl = document.getElementById('ok');
const badEl = document.getElementById('bad');
const accEl = document.getElementById('acc');
const curimgEl = document.getElementById('curimg');
const predEl = document.getElementById('pred');

const imgsStatus = document.getElementById('imgs-status');
const thumbs = document.getElementById('thumbs');

// hidden source image (not displayed)
// used for training / testing
const img = document.getElementById('img');

// canvases for rotated images
const c0   = document.getElementById('c0');
const c90  = document.getElementById('c90');
const c180 = document.getElementById('c180');
const c270 = document.getElementById('c270');
const ctest = document.getElementById('ctest');

const ctx0 = c0.getContext('2d');
const ctx90 = c90.getContext('2d');
const ctx180 = c180.getContext('2d');
const ctx270 = c270.getContext('2d');
const ctxTest = ctest.getContext('2d');

// models
let net = null; // MobileNet embedding model
let knn = null; // KNN classifier

// loop control
let running = false;

// data
let imageURLs = []; // blob urls from users files
let stats = { it: 0, ok: 0, bad: 0 };

// to cap memory usage
const MAX_EXAMPLES_PER_CLASS = 400;

// init
(async function init() {
    try {
        statusEl.textContent = 'Loading TensorFlow.js + MobileNet…';
        await tf.ready();
        net = await mobilenet.load({ version: 2, alpha: 1.0 });
        knn = knnClassifier.create();
        statusEl.textContent = 'Models ready. Load your images to enable the loop.';
    } catch (e) {
        console.error(e);
        statusEl.textContent = 'Failed to load models.';
        disableControls(true);
    }
})();

// disable / enable all controls
function disableControls(disabled) {
    [
        btnStart, 
        btnStop, 
        btnResetStats, 
        btnClearModel, 
        fileInput, 
        btnClearImages, 
        delayInput
    ]
    .forEach(el => el.disabled = disabled);
}

// image Loading 
fileInput.onchange = () => {
    clearImages();
    const files = Array.from(fileInput.files || []);
    if (!files.length) {
        imgsStatus.textContent = 'No images yet.';
        btnStart.disabled = true;
        btnClearImages.disabled = true;
        return;
    }
    files.forEach(f => {
        const url = URL.createObjectURL(f);
        imageURLs.push({ url, name: f.name });
        const t = document.createElement('img');
        t.src = url;
        t.title = f.name;
        thumbs.appendChild(t);
    });
    imgsStatus.textContent = `Loaded ${files.length} images.`;
    btnStart.disabled = false;
    btnClearImages.disabled = false;
};

// clear loaded images button
btnClearImages.onclick = () => {
    clearImages();
    fileInput.value = '';
    imgsStatus.textContent = 'Removed all images.';
    btnStart.disabled = true;
    btnClearImages.disabled = true;
};

// load default images button
btnDefaultImages.onclick = () => {
    clearImages(); // clear anything previously loaded

    // preload 10 default images from /images folder
    for (let i = 1; i <= 10; i++) {
        const name = `cat${i}.png`;
        const url = `images/${name}`;
        imageURLs.push({ url, name });

        const t = document.createElement('img');
        t.src = url;
        t.title = name;
        thumbs.appendChild(t);
    }

    imgsStatus.textContent = `Loaded 10 default cat images.`;
    btnStart.disabled = false;
    btnClearImages.disabled = false;
};

// clear loaded images utility
function clearImages() {
    imageURLs.forEach(o => URL.revokeObjectURL(o.url));
    imageURLs = [];
    thumbs.innerHTML = '';
}

// draw rotated image into canvas
function drawRot(ctx, sourceImageEl, deg) {
    const sz = 224;
    ctx.save();
    ctx.clearRect(0,0,sz,sz);
    ctx.translate(sz/2, sz/2);
    ctx.rotate(deg * Math.PI/180);
    const iw = sourceImageEl.naturalWidth || sourceImageEl.width;
    const ih = sourceImageEl.naturalHeight || sourceImageEl.height;
    const minSide = Math.min(iw, ih);
    const sx = (iw - minSide)/2;
    const sy = (ih - minSide)/2;
    ctx.drawImage(sourceImageEl, sx, sy, minSide, minSide, -sz/2, -sz/2, sz, sz);
    ctx.restore();
}

function embedFrom(elOrCanvas) {
    return tf.tidy(() => net.infer(elOrCanvas, true));
}

// train once on ALL 4 rotations of the image
async function trainOnceOnImage(sourceEl) {
    const specs = [
        {ctx: ctx0,   deg: 0,   label: 'rot_0'},
        {ctx: ctx90,  deg: 90,  label: 'rot_90'},
        {ctx: ctx180, deg: 180, label: 'rot_180'},
        {ctx: ctx270, deg: 270, label: 'rot_270'}
    ];

    for (const s of specs) {
        drawRot(s.ctx, sourceEl, s.deg);
        const act = embedFrom(s.ctx.canvas);
        knn.addExample(act, s.label);
        act.dispose();
        await tf.nextFrame();
    }

    // memory cap because KNN keeps growing indefinitely
    if (MAX_EXAMPLES_PER_CLASS) {
        const counts = knn.getClassExampleCount();
        for (const k of Object.keys(counts)) {
            if (counts[k] > MAX_EXAMPLES_PER_CLASS) {
                console.warn(`Cap reached for ${k}. Clearing KNN to avoid overgrowth.`);
                knn.clearAllClasses();
                break;
            }
        }
    }
}

// returns: { ok, trueDeg, predDeg, conf }
async function testOnceOnImage(sourceEl) {
    const angles = [0,90,180,270];
    const trueDeg = angles[Math.floor(Math.random()*angles.length)];
    drawRot(ctxTest, sourceEl, trueDeg);
    const act = embedFrom(ctest);
    const res = await knn.predictClass(act, 3);
    act.dispose();
    const predDeg = parseInt(res.label.replace('rot_',''), 10);
    const conf = (res.confidences[res.label] * 100).toFixed(1);
    return { ok: predDeg === trueDeg, trueDeg, predDeg, conf };
}

// loop control
btnStart.onclick = async () => {
    if (!imageURLs.length) {
        statusEl.textContent = 'Load images first.';
        return;
    }

    running = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    statusEl.textContent = 'Loop running. Training then testing on random images.';
    await runLoop();
};

btnStop.onclick = () => {
    running = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    statusEl.textContent = 'Loop stopped.';
};

btnResetStats.onclick = () => {
    stats = { it: 0, ok: 0, bad: 0 };
    updateStats();
    predEl.textContent = '—';
};

btnClearModel.onclick = () => {
    knn.clearAllClasses();
    statusEl.textContent = 'Cleared KNN examples.';
};

// main loop
async function runLoop() {
    while (running) {
        const pick = imageURLs[Math.floor(Math.random() * imageURLs.length)];
        await loadImageInto(img, pick.url);
        curimgEl.textContent = pick.name || '(blob)';
        await trainOnceOnImage(img);
        const out = await testOnceOnImage(img);
        stats.it += 1;
        if (out.ok) stats.ok += 1; else stats.bad += 1;
        updateStats(out);
        const ms = Math.max(0, Number(delayInput.value) || 0);
        if (ms) await sleep(ms);
        await tf.nextFrame();
    }
}

function updateStats(last = null) {
    itEl.textContent = String(stats.it);
    okEl.textContent = String(stats.ok);
    badEl.textContent = String(stats.bad);
    const acc = stats.it ? (100 * stats.ok / stats.it) : 0;
    accEl.textContent = `${acc.toFixed(2)}%`;
    if (last) {
        predEl.textContent = `pred ${last.predDeg}° (true ${last.trueDeg}°) - ${last.ok ? '✅ correct' : '❌ wrong'} - conf ${last.conf}%`;
    }
}

// utilities
function sleep(ms) { return new Promise(res => setTimeout(res, ms)); }
function loadImageInto(el, url) {
    return new Promise((resolve, reject) => {
        el.onload = () => resolve();
        el.onerror = err => reject(err);
        el.src = url;
    });
}