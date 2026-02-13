// worker.js
// 1. 使用标准 import 语法，彻底移除 importScripts
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js";

let handLandmarker;

self.onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'INIT_AI') {
        try {
            // 2. 在 Worker 线程中解析 WASM，这里的 CPU 飙升不会影响网页操作
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
            );
            
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "CPU" // 在线程隔离环境下，CPU 模式最稳定
                },
                runningMode: "IMAGE",
                numHands: 1
            });
            
            self.postMessage({ type: 'AI_READY' }); // 通知主线程已就绪
        } catch (err) {
            self.postMessage({ type: 'ERROR', msg: err.message });
        }
    }

    if (type === 'DETECT') {
        if (!handLandmarker) return;
        try {
            // 3. 执行识别并将结果传回
            const result = handLandmarker.detect(data.imageBitmap);
            self.postMessage({ type: 'DETECTION_RESULT', result: result });
        } catch (err) {
            self.postMessage({ type: 'ERROR', msg: "检测失败: " + err.message });
        }
    }
};