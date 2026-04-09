const API_BASE = "http://127.0.0.1:8001";

const elements = {
    video: document.getElementById("webcamVideo"),
    canvas: document.getElementById("outputCanvas"),
    imgUpload: document.getElementById("imageUpload"),
    status: document.getElementById("systemStatus"),
    recStatus: document.getElementById("recognitionStatus"),
    btnStart: document.getElementById("startCameraButton"),
    btnRecognize: document.getElementById("recognizeButton"),
    timeLabel: document.getElementById("processingTime"),
    heartConf: document.getElementById("heartLineConf"),
    headConf: document.getElementById("headLineConf"),
    lifeConf: document.getElementById("lifeLineConf"),
    reasonList: document.getElementById("reasonList"),
};

let latestImageBlob = null;

function setStatus(text, ok = true) {
    elements.status.textContent = text;
    elements.status.style.color = ok ? "#00d1b2" : "#ff6b6b";
}

function updateProgressUI(id, val) {
    const bar = document.getElementById("bar-" + id);
    if (bar) {
        bar.style.width = `${Math.max(0, Math.min(100, val))}%`;
    }
}

function drawPolygon(ctx, polygon, strokeColor, fillColor) {
    if (!polygon || polygon.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1]);
    for (let i = 1; i < polygon.length; i += 1) {
        ctx.lineTo(polygon[i][0], polygon[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
}

function drawPrediction(predictions) {
    const ctx = elements.canvas.getContext("2d");
    if (!ctx) return;

    if (predictions.palm?.detected) {
        drawPolygon(ctx, predictions.palm.polygon, "rgb(255,255,0)", "rgba(255,255,0,0.10)");
    }
    if (predictions.heart_line?.detected) {
        drawPolygon(ctx, predictions.heart_line.polygon, "rgb(255,59,48)", "rgba(255,59,48,0.20)");
    }
    if (predictions.head_line?.detected) {
        drawPolygon(ctx, predictions.head_line.polygon, "rgb(76,217,100)", "rgba(76,217,100,0.20)");
    }
    if (predictions.life_line?.detected) {
        drawPolygon(ctx, predictions.life_line.polygon, "rgb(0,122,255)", "rgba(0,122,255,0.20)");
    }

    if (predictions.hegu?.detected && predictions.hegu.point?.length === 2) {
        const [x, y] = predictions.hegu.point;
        ctx.fillStyle = "#ffd700";
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fill();
    }
}

function setScores(predictions) {
    const heart = Math.round((predictions.heart_line?.confidence || 0) * 100);
    const head = Math.round((predictions.head_line?.confidence || 0) * 100);
    const life = Math.round((predictions.life_line?.confidence || 0) * 100);
    elements.heartConf.textContent = `${heart}%`;
    elements.headConf.textContent = `${head}%`;
    elements.lifeConf.textContent = `${life}%`;
    updateProgressUI("heart", heart);
    updateProgressUI("head", head);
    updateProgressUI("life", life);
}

function setReasons(reasons) {
    elements.reasonList.innerHTML = "";
    if (!reasons || reasons.length === 0) {
        const li = document.createElement("li");
        li.textContent = "识别通过：未检测到明显异常。";
        elements.reasonList.appendChild(li);
        return;
    }
    reasons.forEach((reason) => {
        const li = document.createElement("li");
        li.textContent = `${reason.code}: ${reason.message}`;
        elements.reasonList.appendChild(li);
    });
}

function canvasToBlob(canvas) {
    return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.95);
    });
}

async function callInferenceApi(imageBlob) {
    const formData = new FormData();
    formData.append("file", imageBlob, "palm.jpg");
    const resp = await fetch(`${API_BASE}/api/infer`, {
        method: "POST",
        body: formData,
    });
    if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`推理失败(${resp.status}): ${txt}`);
    }
    return resp.json();
}

elements.btnStart.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        elements.video.srcObject = stream;
        await elements.video.play();
        elements.btnRecognize.disabled = false;
        setStatus("摄像头就绪，后端模型待调用。");
        elements.recStatus.textContent = "实时画面已开启，可点击执行深度分析。";
    } catch (_err) {
        setStatus("摄像头权限失败，请允许访问摄像头。", false);
    }
};

elements.imgUpload.onchange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const bitmap = await createImageBitmap(file);
    elements.canvas.width = bitmap.width;
    elements.canvas.height = bitmap.height;
    const ctx = elements.canvas.getContext("2d");
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    ctx.drawImage(bitmap, 0, 0);
    latestImageBlob = await canvasToBlob(elements.canvas);
    elements.btnRecognize.disabled = false;
    elements.recStatus.textContent = "已加载本地图片，可点击执行深度分析。";
};

elements.btnRecognize.onclick = async () => {
    const start = performance.now();
    try {
        elements.recStatus.textContent = "正在调用模型推理...";

        if (elements.video.srcObject) {
            elements.canvas.width = elements.video.videoWidth;
            elements.canvas.height = elements.video.videoHeight;
            const ctx = elements.canvas.getContext("2d");
            ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
            ctx.drawImage(elements.video, 0, 0);
            latestImageBlob = await canvasToBlob(elements.canvas);
        }

        if (!latestImageBlob) {
            elements.recStatus.textContent = "请先开启摄像头或上传图片。";
            return;
        }

        const result = await callInferenceApi(latestImageBlob);
        setScores(result.predictions);
        setReasons(result.reasons);
        drawPrediction(result.predictions);

        const elapsed = Math.round(performance.now() - start);
        elements.timeLabel.textContent = `${elapsed} ms`;
        elements.recStatus.textContent = result.success ? "✅ 分析完成" : "⚠️ 分析完成，但存在识别风险";
    } catch (err) {
        elements.recStatus.textContent = "推理失败，请检查服务是否运行。";
        setReasons([{ code: "API_ERROR", message: String(err.message || err) }]);
        setStatus("后端不可用，请先启动 ml/api.py。", false);
    }
};

window.addEventListener("load", async () => {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        if (!resp.ok) throw new Error("health check failed");
        const data = await resp.json();
        setStatus(`模型服务在线：${data.status}`);
    } catch (_err) {
        setStatus("模型服务离线，请先启动 ml/api.py。", false);
    }
});