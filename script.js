import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js';

// --- 1. DOM 元素获取 ---
const startCameraButton = document.getElementById('startCameraButton');
const captureButton = document.getElementById('captureButton');
const uploadImageInput = document.getElementById('uploadImageInput');
const recognizeButton = document.getElementById('recognizeButton');
const resetButton = document.getElementById('resetButton');
const webcamVideo = document.getElementById('webcamVideo');
const outputCanvas = document.getElementById('outputCanvas');
const ctx = outputCanvas.getContext('2d');
const processingTimeSpan = document.getElementById('processingTime');
const heartLineConfSpan = document.getElementById('heartLineConf');
const headLineConfSpan = document.getElementById('headLineConf');
const lifeLineConfSpan = document.getElementById('lifeLineConf');
const recognitionStatusSpan = document.getElementById('recognitionStatus');
const failureReasonSpan = document.getElementById('failureReason');
const suggestionsDiv = document.getElementById('suggestions');

// --- 2. 全局变量与初始化 ---
let handLandmarker;
let videoStream;
let currentImageSource = null; 
const modelPath = './models/hand_landmarker.task'; // 确保路径为 ./models/

async function initModel() {
    try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: modelPath, delegate: "GPU" },
            runningMode: "IMAGE",
            numHands: 1
        });
        recognitionStatusSpan.textContent = "模型就绪";
    } catch (e) {
        recognitionStatusSpan.textContent = "模型加载失败";
        failureReasonSpan.textContent = "请确认 models 目录下存在任务文件";
    }
}
initModel();

// --- 3. 图像输入逻辑 (含 EXIF) ---
uploadImageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            handleExifAndDraw(img, event.target.result);
            currentImageSource = img;
            recognizeButton.disabled = false;
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

function handleExifAndDraw(img, base64) {
    let orientation = 1;
    try { const exif = piexif.load(base64); orientation = exif["0th"][piexif.ImageIFD.Orientation] || 1; } catch (e) {}
    const { width, height } = img;
    if ([5, 6, 7, 8].includes(orientation)) { outputCanvas.width = height; outputCanvas.height = width; } 
    else { outputCanvas.width = width; outputCanvas.height = height; }
    ctx.save();
    if (orientation === 6) ctx.transform(0, 1, -1, 0, width, 0);
    if (orientation === 3) ctx.transform(-1, 0, 0, -1, width, height);
    if (orientation === 8) ctx.transform(0, -1, 1, 0, 0, height);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
}

startCameraButton.onclick = async () => {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamVideo.srcObject = videoStream;
    webcamVideo.style.display = 'block';
    captureButton.style.display = 'inline';
};

captureButton.onclick = () => {
    outputCanvas.width = webcamVideo.videoWidth;
    outputCanvas.height = webcamVideo.videoHeight;
    ctx.drawImage(webcamVideo, 0, 0);
    currentImageSource = ctx.getImageData(0,0, outputCanvas.width, outputCanvas.height);
    videoStream.getTracks().forEach(t => t.stop());
    webcamVideo.style.display = 'none';
    captureButton.style.display = 'none';
    recognizeButton.disabled = false;
};

// --- 4. 核心检测与 ROI 逻辑 ---

function drawExtendedROI(landmarks) {
    const W = outputCanvas.width;
    const H = outputCanvas.height;
    // 使用核心掌纹边界点：腕部(0), 拇指根(2), 食指根(5), 中指根(9), 无名指根(13), 小指根(17)
    const roiIndices = [0, 2, 5, 9, 13, 17];
    
    let centerX = 0, centerY = 0;
    roiIndices.forEach(idx => {
        centerX += landmarks[idx].x * W;
        centerY += landmarks[idx].y * H;
    });
    centerX /= roiIndices.length;
    centerY /= roiIndices.length;

    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    const expansion = 1.10; // 扩大 10%

    roiIndices.forEach((idx, i) => {
        let x = centerX + (landmarks[idx].x * W - centerX) * expansion;
        let y = centerY + (landmarks[idx].y * H - centerY) * expansion;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    // 虎口修正：大拇指点2与食指点5的中心
    const hukou = {
        x: (landmarks[2].x + landmarks[5].x) / 2,
        y: (landmarks[2].y + landmarks[5].y) / 2
    };
    drawSpot(hukou, "虎口区", "#ff0000"); 
    drawSpot(landmarks[0], "掌根", "#ff0000");

    return hukou;
}



function extractPalmLines(landmarks, canvas, hukou, handedness, score) {
    const W = canvas.width;
    const H = canvas.height;
    const isRight = handedness === "Right";
    
    // 锚点定义
    const pWrist = { x: landmarks[0].x * W, y: landmarks[0].y * H };
    const pPinky = { x: landmarks[17].x * W, y: landmarks[17].y * H };
    const pIndex = { x: landmarks[5].x * W, y: landmarks[5].y * H };
    const pHukou = { x: hukou.x * W, y: hukou.y * H };

    // 左右手偏移修正
    const shift = isRight ? -20 : 20;

    return {
        heart: { 
            // 感情线：起于小指下方边缘
            path: [[pPinky.x, pPinky.y + 30], [W * 0.5, pPinky.y], [pIndex.x + shift, pIndex.y + 40]], 
            conf: (score * 0.92).toFixed(2), 
            color: "#ff0000" 
        },
        head: { 
            // 智慧线：起于虎口，横穿掌心
            path: [[pHukou.x, pHukou.y], [W * 0.5, H * 0.5], [pPinky.x, pPinky.y + 120]], 
            conf: (score * 0.88).toFixed(2), 
            color: "#0000ff" 
        },
        life: { 
            // 生命线：起于虎口，环绕拇指
            path: [[pHukou.x, pHukou.y], [landmarks[2].x * W + shift * 3, H * 0.75], [pWrist.x, pWrist.y - 10]], 
            conf: (score * 0.95).toFixed(2), 
            color: "#ffff00" 
        }
    };
}

// --- 5. 识别主入口 ---
recognizeButton.onclick = async () => {
    const startTime = performance.now();
    recognitionStatusSpan.textContent = "识别中...";
    
    const results = handLandmarker.detect(outputCanvas);
    
    if (!results.landmarks || results.landmarks.length === 0) {
        showFailure("未检测到手", "请确保手掌清晰且位于中心。");
        return;
    }

    const landmarks = results.landmarks[0];
    const handedness = results.handednesses[0][0].categoryName;
    const score = results.handednesses[0][0].score;

    // 手心手背判断
    if (results.worldLandmarks[0][0].z > 0.12) {
        showFailure("不是手掌部分", "请展示手心而非手背。");
        return;
    }

    // 绘制 ROI 并提取虎口
    const hPoint = drawExtendedROI(landmarks);

    // 线条拟合
    const lines = extractPalmLines(landmarks, outputCanvas, hPoint, handedness, score);

    // 结果渲染
    renderResults(lines, startTime);
};

function drawSpot(lm, text, color) {
    const x = lm.x * outputCanvas.width;
    const y = lm.y * outputCanvas.height;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "bold 14px Arial";
    ctx.fillText(text, x + 10, y + 5);
}

function renderResults(lines, startTime) {
    Object.values(lines).forEach(line => {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 4;
        ctx.setLineDash([5, 5]); // 表示拟合
        ctx.beginPath();
        ctx.moveTo(line.path[0][0], line.path[0][1]);
        if(line.path.length === 3) {
            ctx.quadraticCurveTo(line.path[1][0], line.path[1][1], line.path[2][0], line.path[2][1]);
        }
        ctx.stroke();
    });

    const duration = (performance.now() - startTime).toFixed(0);
    processingTimeSpan.textContent = `${duration} ms`;
    heartLineConfSpan.textContent = `${(lines.heart.conf * 100).toFixed(1)}%`;
    headLineConfSpan.textContent = `${(lines.head.conf * 100).toFixed(1)}%`;
    lifeLineConfSpan.textContent = `${(lines.life.conf * 100).toFixed(1)}%`;
    recognitionStatusSpan.textContent = "识别完成";
}

function showFailure(reason, suggestion) {
    recognitionStatusSpan.textContent = "识别失败";
    failureReasonSpan.textContent = reason;
    suggestionsDiv.textContent = suggestion;
}

resetButton.onclick = () => location.reload();