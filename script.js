/**
 * script.js - 全状态感知 & 精准解剖定位版
 * 功能：TF.js Handpose + Fingerpose + OpenCV Worker
 * 核心逻辑：以食指掌骨(p5-p0)中点定位合谷，并追踪三条主线
 */

// --- 1. 全局配置与状态追踪 ---
const elements = {
    video: document.getElementById('webcamVideo'),
    canvas: document.getElementById('outputCanvas'),
    imgUpload: document.getElementById('imageUpload'),
    status: document.getElementById('systemStatus'),
    recStatus: document.getElementById('recognitionStatus'),
    btnStart: document.getElementById('startCameraButton'),
    btnRecognize: document.getElementById('recognizeButton'),
    timeLabel: document.getElementById('processingTime'),
    // 结果面板
    heartConf: document.getElementById('heartLineConf'),
    headConf: document.getElementById('headLineConf'),
    lifeConf: document.getElementById('lifeLineConf')
};

let statusTracker = { ai: false, opencv: false };
let handposeModel, algoWorker;

// 更新 UI 状态栏
function updateGlobalStatus() {
    const aiLabel = statusTracker.ai ? "✅ AI 模型就绪" : "⏳ AI 模型加载中...";
    const cvLabel = statusTracker.opencv ? "✅ 算法引擎就绪" : "⏳ 算法引擎(WASM)编译中...";
    
    elements.status.innerHTML = `<span style="color: ${statusTracker.ai ? '#00ff00':'#ffa500'}">${aiLabel}</span> | 
                                 <span style="color: ${statusTracker.opencv ? '#00ff00':'#ffa500'}">${cvLabel}</span>`;
    
    if (statusTracker.ai && statusTracker.opencv) {
        elements.btnRecognize.disabled = false;
        elements.btnRecognize.style.background = "#00d1b2";
        console.log("系统提示：全引擎初始化完成，可以开始检测。");
    }
}

// --- 2. 初始化 Worker (含 OpenCV 状态反馈) ---
const initWorker = () => {
    const localOpenCVPath = window.location.origin + window.location.pathname.replace(/\/[^\/]*$/, '/') + "opencv.js";

    const workerCode = `
        let cvReady = false;
        
        // 【保留】确认 OpenCV 加载成功的钩子
        self.cv = {
            onRuntimeInitialized: () => {
                console.log("Worker: 🔔 OpenCV 加载成功");
                cvReady = true;
                self.postMessage({ type: 'CV_READY' });
            }
        };

        try {
        importScripts("${localOpenCVPath}");
            // 【保留】二次心跳检测
            setInterval(() => {
                if (!cvReady && typeof cv !== 'undefined' && cv.Mat) {
                    cvReady = true;
                    self.postMessage({ type: 'CV_READY' });
                }
            }, 500);
        } catch (e) {
            self.postMessage({ type: 'ERROR', msg: '路径错误: ' + e.message });
        }

        self.onmessage = async (e) => {
            const { bitmap, landmarks, width, height } = e.data;
            if (!cvReady || !bitmap || !landmarks) return;

            const canvas = new OffscreenCanvas(width, height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0);
            const imgData = ctx.getImageData(0, 0, width, height);

            // --- 5. 环境反馈机制：光线检查 ---
            let totalBrightness = 0;
            for(let i=0; i<imgData.data.length; i+=4) {
                totalBrightness += (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3;
            }
            const avgBrightness = totalBrightness / (width * height);
            if (avgBrightness < 40) { self.postMessage({ type: 'ERROR', msg: '环境过暗，请开启闪光灯或移至明亮处' }); return; }
            if (avgBrightness > 220) { self.postMessage({ type: 'ERROR', msg: '环境过亮或反光，请调整角度' }); return; }

            try {
                let src = new cv.Mat(height, width, cv.CV_8UC4);
                src.data.set(imgData.data);

                // --- ROI 严格限域 (解决背景干扰) ---
                let mask = cv.Mat.zeros(height, width, cv.CV_8UC1);
                let roiCoords = new Int32Array([
                    landmarks[5].x, landmarks[5].y, landmarks[9].x, landmarks[9].y,
                    landmarks[17].x, landmarks[17].y, landmarks[0].x, landmarks[0].y, landmarks[2].x, landmarks[2].y
                ]);
                let pts = cv.matFromArray(roiCoords.length/2, 1, cv.CV_32SC2, roiCoords);
                let ptsVec = new cv.MatVector(); ptsVec.push_back(pts);
                cv.fillPoly(mask, ptsVec, new cv.Scalar(255));
                let maskedSrc = new cv.Mat(); src.copyTo(maskedSrc, mask);

                // 图像增强
                let gray = new cv.Mat(); cv.cvtColor(maskedSrc, gray, cv.COLOR_RGBA2GRAY);
                let clahe = new cv.CLAHE(5.0, new cv.Size(8, 8));
                let enhanced = new cv.Mat(); clahe.apply(gray, enhanced);
                let edges = new cv.Mat(); cv.Canny(enhanced, edges, 25, 60); // 降低阈值提高检出率

                let contours = new cv.MatVector();
                let hierarchy = new cv.Mat();
                cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                const p0 = landmarks[0], p2 = landmarks[2], p5 = landmarks[5], p9 = landmarks[9], p17 = landmarks[17];
                const hegu = { x: p5.x + (p0.x - p5.x) * 0.5, y: p5.y + (p0.y - p5.y) * 0.5 };

                let results = { 
                    heart: { path: [], score: 0 }, 
                    head: { path: [], score: 0 }, 
                    life: { path: [], score: 0 } 
                };

                for (let i = 0; i < contours.size(); ++i) {
                    let cnt = contours.get(i);
                    let ptsArr = [];
                    for (let j = 0; j < cnt.data32S.length; j += 2) {
                        ptsArr.push({ x: cnt.data32S[j], y: cnt.data32S[j+1] });
                    }
                    if (ptsArr.length < 20) continue;

                    const start = ptsArr[0];
                    const mid = ptsArr[Math.floor(ptsArr.length/2)];
                    const end = ptsArr[ptsArr.length-1];

                    // --- 逻辑微调：解决判错颜色问题 ---

                    // 1. 感情线 (红色): 起于小指下方区域，且整体水平高度高于合谷
                    let isHeart = (start.x > p17.x - 30) && (mid.y < hegu.y);
                    if (isHeart) {
                        results.heart.path = ptsArr;
                        results.heart.score = Math.min(100, Math.round(ptsArr.length * 1.5));
                    }

                    // 2. 智慧线 (绿色): 起于食指下方，终点向手掌中部延伸
                    let isHead = (start.x < p9.x) && (start.y > p5.y - 10) && (mid.y >= hegu.y && mid.y < p0.y);
                    if (isHead && ptsArr.length > results.head.path.length) {
                        results.head.path = ptsArr;
                        results.head.score = Math.min(100, Math.round(ptsArr.length * 1.2));
                    }

                    // 3. 生命线 (蓝色): 弧形判定，必须包绕 p2 (拇指根)
                    let isLife = (start.x < p5.x) && (end.y > p0.y - 50) && (mid.x < (p2.x + p5.x)/2);
                    if (isLife) {
                        results.life.path = ptsArr;
                        results.life.score = Math.min(100, Math.round(ptsArr.length * 1.1));
                    }
                }

                if (!results.heart.path.length && !results.head.path.length && !results.life.path.length) {
                    self.postMessage({ type: 'ERROR', msg: '未检测到清晰线条，请尝试侧光照射并保持手掌平齐' });
                }

                self.postMessage({ type: 'RESULT', results, hegu });
                
                // 释放
                src.delete(); mask.delete(); pts.delete(); ptsVec.delete(); maskedSrc.delete();
                gray.delete(); enhanced.delete(); edges.delete(); contours.delete(); hierarchy.delete();
            } catch (err) { self.postMessage({ type: 'ERROR', msg: '算法运行异常' }); }
        };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    algoWorker = new Worker(URL.createObjectURL(blob));

    algoWorker.onmessage = (e) => {
        if (e.data.type === 'CV_READY') { statusTracker.opencv = true; updateGlobalStatus(); }
        if (e.data.type === 'RESULT') {
            renderFinal(e.data.results, e.data.hegu);
            // --- 更新置信度 UI ---
            updateConfidenceUI(e.data.results);
        }
        if (e.data.type === 'ERROR') {
            showUserFeedback(e.data.msg); // 调用 UI 提示函数
        }
    };
};

// 辅助函数：显示失败原因
const showUserFeedback = (msg) => {
    const statusEl = document.getElementById('status-text');
    if (statusEl) {
        statusEl.innerText = "识别提醒: " + msg;
        statusEl.style.color = "#ff4d4d";
    }
};

// 辅助函数：更新 UI 置信度
const updateConfidenceUI = (results) => {
    if (document.getElementById('heart-score')) {
        document.getElementById('heart-score').innerText = results.heart.score + "%";
        document.getElementById('head-score').innerText = results.head.score + "%";
        document.getElementById('life-score').innerText = results.life.score + "%";
    }
};

// --- 3. 初始化 AI ---
async function setupAI() {
    try {
        statusTracker.ai = false;
        updateGlobalStatus();
        handposeModel = await handpose.load();
        statusTracker.ai = true;
        updateGlobalStatus();
    } catch (err) {
        elements.status.textContent = "AI 加载失败: " + err.message;
    }
}

// --- 4. 交互触发 ---
elements.btnStart.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        elements.video.srcObject = stream;
        elements.video.play();
        initWorker(); // 启动算法 Worker
        setupAI();    // 启动 AI 模型
    } catch (err) {
        alert("请允许摄像头权限以继续。");
    }
};

elements.imgUpload.onchange = (e) => {
    const reader = new FileReader();
    reader.onload = (f) => {
        const img = new Image();
        img.onload = () => {
            elements.canvas.width = img.width;
            elements.canvas.height = img.height;
            elements.canvas.getContext('2d').drawImage(img, 0, 0);
            if (!handposeModel) { initWorker(); setupAI(); }
        };
        img.src = f.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
};

elements.btnRecognize.onclick = async () => {
    if (!statusTracker.ai || !statusTracker.opencv) return;
    
    const startTime = performance.now();
    elements.recStatus.textContent = "正在捕捉手掌形态...";
    
    const source = elements.video.srcObject ? elements.video : elements.canvas;
    if (elements.video.srcObject) {
        elements.canvas.width = elements.video.videoWidth;
        elements.canvas.height = elements.video.videoHeight;
        elements.canvas.getContext('2d').drawImage(elements.video, 0, 0);
    }

    const predictions = await handposeModel.estimateHands(source);

    if (predictions.length > 0) {
        const landmarks = predictions[0].landmarks;
        const bitmap = await createImageBitmap(elements.canvas);
        
        algoWorker.postMessage({
            bitmap,
            landmarks: landmarks.map(p => ({ x: p[0], y: p[1] })),
            width: elements.canvas.width,
            height: elements.canvas.height
        }, [bitmap]);

        elements.timeLabel.dataset.start = startTime;
    } else {
        elements.recStatus.textContent = "未检测到手掌，请调整姿势。";
    }
};

// --- 5. 最终渲染 ---
function renderFinal(res, hegu) {
    const ctx = elements.canvas.getContext('2d');
    
    // 绘制合谷点 (金色)
    ctx.fillStyle = "#FFD700";
    ctx.beginPath(); ctx.arc(hegu.x, hegu.y, 8, 0, Math.PI * 2); ctx.fill();

    const drawLine = (path, color) => {
        if (!path || path.length < 2) return;
        ctx.strokeStyle = color; ctx.lineWidth = 5; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(path[0].x, path[0].y);
        path.forEach(p => ctx.lineTo(p.x, p.y)); ctx.stroke();
    };

    drawLine(res.heart.path, "#FF3B30"); // 感情线
    drawLine(res.head.path, "#4CD964");  // 智慧线
    drawLine(res.life.path, "#007AFF");  // 生命线

    const duration = performance.now() - parseFloat(elements.timeLabel.dataset.start);
    elements.timeLabel.textContent = Math.round(duration) + " ms";
    elements.recStatus.textContent = "✅ 分析完成";

    // 同步 UI 进度
    elements.heartConf.textContent = res.heart.score + "%";
    elements.headConf.textContent = res.head.score + "%";
    elements.lifeConf.textContent = res.life.score + "%";
}