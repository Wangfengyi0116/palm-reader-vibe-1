# 掌纹主线自动化识别系统 (MVP)

本课题旨在通过前端技术与 AI 模型结合，实现手机端拍照/导入手掌照片后，自动提取掌纹三大主线（感情线、智慧线、生命线）并进行可视化展示。

## 功能特性
* **多模式输入**：支持调用摄像头实时拍照截图及本地图片导入。
* **EXIF 自动校正**：解决手机拍摄照片上传后出现的横竖颠倒问题。
* **AI 关键点定位**：集成 MediaPipe Hands，精准识别掌根、虎口等 21 个手部关键点。
* **ROI 与 轮廓绘制**：自动提取掌心区域并绘制手掌边缘轮廓。
* **主线提取算法**：结合手部解剖学模型与图像增强技术，拟合生成三大主线。
* **异常处理**：针对“未检测到手”、“手背拍摄”、“光线过暗”等场景提供明确的错误反馈。

## 目录结构
- `index.html` - 主界面结构
- `script.js` - 核心算法逻辑（含 MediaPipe 初始化与掌纹拟合）
- `style.css` - 响应式交互样式
- `models/` - 存放 `hand_landmarker.task` 模型文件
- `docs/design.md` - 技术实现方案与调研报告

## 开发者快速上手
1.  确保 `models/hand_landmarker.task` 已下载并放置。
2.  使用 Cursor 的终端运行 `python -m http.server 8000`。
3.  在浏览器打开 `localhost:8000`。
4.  建议在充足光线下测试，确保掌纹清晰可见。

阶段,采用方法,结果,失败原因分析
1. 基础版,主线程同步加载所有库,卡死 (100% CPU),WASM 编译是 CPU 密集型任务，强行占用 UI 线程导致画面完全停滞。
2. 异步版,setTimeout & 异步加载,依然卡死,WASM 编译是原子操作，一旦开始，浏览器无法在中途强行切换任务。
3. Worker 版 (v1),importScripts 引入库,报错 (SyntaxError),现代库（MediaPipe）使用了 ES Module 语法，而传统 Worker 脚本不支持 export。
4. Worker 版 (v2),"type: ""module"" Worker",报错 (Security),模块化 Worker 不允许跨域加载库脚本，且与旧版 importScripts 冲突。
5. 内联 Blob 版,将代码转为 Blob 运行,报错 (CORS/Network),库内部尝试加载 .wasm 二进制文件时，因路径不匹配被浏览器拦截。