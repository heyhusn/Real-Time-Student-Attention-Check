/**
 * attention.js
 * ─────────────────────────────────────────────────────────────────────────
 * Browser-side real-time attention detection using MediaPipe Face Mesh.
 *
 * Pipeline:
 *   Webcam (user-selected) → MediaPipe FaceMesh → head-pose + EAR →
 *   attention score → WebSocket → FastAPI backend (every SEND_INTERVAL_MS)
 * ─────────────────────────────────────────────────────────────────────────
 */

'use strict';

// ─── Tunable constants ────────────────────────────────────────────────────
const HEAD_YAW_THR = 15;   // degrees – head turned left/right limit
const HEAD_PITCH_THR = 15;   // degrees – head tilted up/down limit
const EAR_THR = 0.22; // EAR below this → eyes considered closed
const SEND_INTERVAL_MS = 500;  // send score to server every 500 ms

// ─── MediaPipe landmark indices ───────────────────────────────────────────
const LM_NOSE_TIP = 1;
const LM_CHIN = 152;
const LM_L_EYE_L = 33;
const LM_R_EYE_R = 263;
const LM_L_MOUTH = 61;
const LM_R_MOUTH = 291;

const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

// Nose sits ~46% of the way between eyes and chin when facing straight
const PITCH_NEUTRAL_RATIO = 0.46;

// ─── Global state ─────────────────────────────────────────────────────────
let socket = null;
let framesSent = 0;
let scoreHistory = [];
let isConnected = false;
let frameLoopRunning = false;

const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('overlay');
const ctx = canvasEl.getContext('2d');

// ─── Utils ────────────────────────────────────────────────────────────────
function lm(landmarks, idx) { return landmarks[idx]; }

function dist2(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function earForEye(landmarks, indices) {
    const [p1, p2, p3, p4, p5, p6] = indices.map(i => lm(landmarks, i));
    const horiz = dist2(p1, p4);
    const vert1 = dist2(p2, p6);
    const vert2 = dist2(p3, p5);
    if (horiz < 1e-6) return 0.3;
    return (vert1 + vert2) / (2.0 * horiz);
}

/**
 * Head-pose approximation.
 * Yaw   = horizontal nose deviation from eye midpoint / face width * 55
 * Pitch = (nose ratio along eye→chin axis) - neutral(0.46), scaled to degrees
 */
function _computeHeadPose(landmarks) {
    const nose = lm(landmarks, LM_NOSE_TIP);
    const chin = lm(landmarks, LM_CHIN);
    const leye = lm(landmarks, LM_L_EYE_L);
    const reye = lm(landmarks, LM_R_EYE_R);

    const faceW = dist2(leye, reye);
    if (faceW < 1e-4) return { yaw: 0, pitch: 0 };

    const eyeMidX = (leye.x + reye.x) / 2;
    const eyeMidY = (leye.y + reye.y) / 2;
    const yaw = ((nose.x - eyeMidX) / faceW) * 55;

    const eyeChinDist = chin.y - eyeMidY;
    if (eyeChinDist < 1e-4) return { yaw, pitch: 0 };
    const noseRatio = (nose.y - eyeMidY) / eyeChinDist;
    const pitch = (noseRatio - PITCH_NEUTRAL_RATIO) * 120;

    return { yaw, pitch };
}

// ─── Attention score ──────────────────────────────────────────────────────
function computeAttentionScore(landmarks, faceDetected) {
    if (!faceDetected) return { score: 0, yaw: 0, pitch: 0, ear: 0 };

    const { yaw, pitch } = _computeHeadPose(landmarks);
    const leftEAR = earForEye(landmarks, LEFT_EYE_IDX);
    const rightEAR = earForEye(landmarks, RIGHT_EYE_IDX);
    const ear = (leftEAR + rightEAR) / 2;

    let score = 1.0;
    if (Math.abs(yaw) > HEAD_YAW_THR || Math.abs(pitch) > HEAD_PITCH_THR) score *= 0.5;
    if (ear < EAR_THR) score *= 0.2;
    score = Math.max(0, Math.min(1, score));

    return { score, yaw, pitch, ear };
}

// ─── Draw helpers ─────────────────────────────────────────────────────────
function scoreColor(score) {
    if (score >= 0.75) return '#22c55e';
    if (score >= 0.40) return '#f59e0b';
    return '#ef4444';
}

function drawLandmarkDot(lm, color, radius) {
    const x = lm.x * canvasEl.width;
    const y = lm.y * canvasEl.height;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawCanvas(landmarks, score, yaw, pitch, ear, faceDetected) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (faceDetected && landmarks) {
        [LM_NOSE_TIP, LM_CHIN, LM_L_EYE_L, LM_R_EYE_R, LM_L_MOUTH, LM_R_MOUTH]
            .forEach(i => drawLandmarkDot(lm(landmarks, i), '#ef4444', 4));
        [...LEFT_EYE_IDX, ...RIGHT_EYE_IDX]
            .forEach(i => drawLandmarkDot(lm(landmarks, i), '#e879f9', 2));
        const color = scoreColor(score);
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.fillStyle = color;
        ctx.fillText('Score: ' + score.toFixed(2), 12, 28);
        ctx.fillStyle = '#facc15';
        ctx.font = '13px Inter, sans-serif';
        ctx.fillText('Yaw: ' + yaw.toFixed(1) + '  Pitch: ' + pitch.toFixed(1) + '  EAR: ' + ear.toFixed(3), 12, 50);
    } else {
        ctx.font = 'bold 18px Inter, sans-serif';
        ctx.fillStyle = '#ef4444';
        ctx.fillText('No face detected', 12, 32);
    }
}

// ─── UI update ────────────────────────────────────────────────────────────
function updateUI(score, yaw, pitch, ear, faceDetected) {
    const color = scoreColor(score);
    document.getElementById('scoreValue').textContent = score.toFixed(2);
    document.getElementById('scoreValue').style.color = color;
    document.getElementById('scoreOverlay').textContent = score.toFixed(2);
    document.getElementById('scoreOverlay').style.color = color;
    document.getElementById('yawValue').textContent = yaw.toFixed(1) + '\u00b0';
    document.getElementById('pitchValue').textContent = pitch.toFixed(1) + '\u00b0';
    document.getElementById('earValue').textContent = ear.toFixed(3);
    document.getElementById('faceValue').textContent = faceDetected ? 'YES' : 'NO';

    scoreHistory.push(score);
    if (scoreHistory.length > 20) scoreHistory.shift();
    const bar = document.getElementById('historyBar');
    bar.innerHTML = scoreHistory
        .map(s => '<div class="bar-item" style="height:' + Math.max(4, s * 48) + 'px;background:' + scoreColor(s) + '"></div>')
        .join('');
}

function addLog(msg) {
    const pre = document.getElementById('logPre');
    const ts = new Date().toLocaleTimeString();
    pre.textContent = '[' + ts + '] ' + msg + '\n' + pre.textContent;
    if (pre.textContent.length > 3000) pre.textContent = pre.textContent.slice(0, 3000);
}

// ─── WebSocket ────────────────────────────────────────────────────────────
function connectSocket(studentId) {
    const wsUrl = 'ws://' + location.host + '/ws/' + encodeURIComponent(studentId);
    addLog('Connecting to ' + wsUrl + ' ...');
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        isConnected = true;
        document.getElementById('statusBadge').textContent = 'Connected';
        document.getElementById('statusBadge').className = 'badge connected';
        document.getElementById('connectBtn').textContent = 'Disconnect';
        addLog('WebSocket connected.');
    };
    socket.onmessage = () => { };
    socket.onerror = () => { addLog('WebSocket error - is the server running?'); };
    socket.onclose = () => {
        isConnected = false;
        document.getElementById('statusBadge').textContent = 'Disconnected';
        document.getElementById('statusBadge').className = 'badge disconnected';
        document.getElementById('connectBtn').textContent = 'Connect';
        addLog('WebSocket closed.');
    };
}

function disconnectSocket() {
    if (socket) { socket.close(); socket = null; }
}

window.toggleConnection = function () {
    if (isConnected) {
        disconnectSocket();
    } else {
        const studentId = document.getElementById('studentId').value.trim() || 'student_001';
        connectSocket(studentId);
    }
};

// ─── Score sender (throttled) ─────────────────────────────────────────────
let lastSentTime = 0;

function maybeSendScore(score) {
    if (!isConnected || !socket || socket.readyState !== WebSocket.OPEN) return;
    const now = Date.now();
    if (now - lastSentTime < SEND_INTERVAL_MS) return;
    lastSentTime = now;
    const payload = JSON.stringify({
        score: parseFloat(score.toFixed(4)),
        timestamp: new Date().toISOString(),
    });
    socket.send(payload);
    framesSent++;
    document.getElementById('sentValue').textContent = framesSent;
    addLog('Sent score=' + score.toFixed(2));
}

// ─── MediaPipe Face Mesh ──────────────────────────────────────────────────
const faceMesh = new FaceMesh({
    locateFile: (file) => 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/' + file,
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
});

faceMesh.onResults((results) => {
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;

    const faceDetected = !!(results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0);
    const landmarks = faceDetected ? results.multiFaceLandmarks[0] : null;

    const { score, yaw, pitch, ear } = computeAttentionScore(landmarks, faceDetected);

    drawCanvas(landmarks, score, yaw, pitch, ear, faceDetected);
    updateUI(score, yaw, pitch, ear, faceDetected);
    maybeSendScore(score);
});

// ─── Camera selector + auto-start ────────────────────────────────────────
// Virtual/software cameras — shown last in dropdown
const VIRTUAL_CAM = /obs|virtual|ndi|snap camera|manycam|xsplit|mmhmm|reincubate|iriun|droidcam/i;

/**
 * Populate the <select id="cameraSelect"> dropdown.
 * Labels each device as [Webcam], [Built-in], or [Virtual].
 * Pre-selects the first real non-built-in camera (external USB webcam).
 */
window.populateCameraList = async function () {
    const sel = document.getElementById('cameraSelect');
    sel.innerHTML = '<option value="">Detecting cameras...</option>';
    try {
        // Need getUserMedia first so labels are visible
        await navigator.mediaDevices.getUserMedia({ video: true });
        const all = await navigator.mediaDevices.enumerateDevices();
        const cams = all.filter(d => d.kind === 'videoinput');

        if (cams.length === 0) {
            sel.innerHTML = '<option value="">No cameras found</option>';
            addLog('No cameras found.');
            return;
        }

        sel.innerHTML = '';
        const BUILTIN = /integrated|built.?in|facetime|internal|ir camera/i;

        cams.forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            const isVirtual = VIRTUAL_CAM.test(d.label || '');
            const isBuiltin = BUILTIN.test(d.label || '');
            const tag = isVirtual ? ' [Virtual]'
                : isBuiltin ? ' [Built-in]'
                    : i === 0 ? ' [Built-in]'   // unknown at index 0 = built-in
                        : ' [Webcam]';    // unknown at index 1+ = external
            opt.text = (d.label || ('Camera ' + i)) + tag;

            // Pre-select the first external webcam (not virtual, index > 0)
            if (!isVirtual && !isBuiltin && i > 0 && sel.selectedIndex <= 0) {
                opt.selected = true;
            }
            sel.appendChild(opt);
        });

        // If nothing was pre-selected, fall back to index 0
        if (sel.value === '') sel.selectedIndex = 0;

        addLog('Found ' + cams.length + ' camera(s):');
        cams.forEach((d, i) => addLog('  [' + i + '] ' + (d.label || 'Unknown')));

    } catch (err) {
        sel.innerHTML = '<option value="">Error: ' + err.message + '</option>';
        addLog('Camera list error: ' + err.message);
    }
};

/**
 * Open a specific camera by deviceId and start the MediaPipe frame loop.
 */
async function openCamera(deviceId, label) {
    // Stop existing stream first
    if (videoEl.srcObject) {
        videoEl.srcObject.getTracks().forEach(t => t.stop());
        videoEl.srcObject = null;
    }

    const constraints = deviceId
        ? { video: { deviceId: { exact: deviceId }, width: 640, height: 480 } }
        : { video: { width: 640, height: 480 } };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoEl.srcObject = stream;
        await new Promise(resolve => { videoEl.onloadedmetadata = resolve; });
        await videoEl.play();
        addLog('Active camera: "' + label + '"');

        // Only start frame loop once; switching camera just changes srcObject
        if (!frameLoopRunning) {
            frameLoopRunning = true;
            (async function processFrame() {
                if (!videoEl.paused && !videoEl.ended && videoEl.srcObject) {
                    await faceMesh.send({ image: videoEl });
                }
                requestAnimationFrame(processFrame);
            })();
        }
    } catch (err) {
        addLog('Camera failed: ' + err.message);
    }
}

/**
 * Called by "Use This" button — switches to the selected dropdown camera.
 */
window.switchCamera = async function () {
    const sel = document.getElementById('cameraSelect');
    const deviceId = sel.value;
    const label = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : 'camera';
    if (!deviceId) { addLog('No camera selected in dropdown.'); return; }
    addLog('Switching to: "' + label + '"...');
    await openCamera(deviceId, label);
};

/**
 * Auto-start: populate dropdown, then open whoever is pre-selected.
 */
(async function init() {
    await populateCameraList();
    const sel = document.getElementById('cameraSelect');
    const label = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : 'default';
    addLog('Auto-starting with: "' + label + '"');
    await openCamera(sel.value, label);
})();
