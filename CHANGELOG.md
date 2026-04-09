# 更新日志

## 2026-03-30

### 一、深度学习模块重建

- 新增 `ml/train.py`：YOLO 训练脚本，支持分割模型与关键点模型训练。
- 新增 `ml/infer.py`：离线推理脚本，输出可视化结果与 JSON。
- 新增 `ml/api.py`：FastAPI 推理服务，提供：
  - `GET /api/health`
  - `POST /api/infer`
- 新增 `ml/datasets/palm_lines_seg.yaml`、`ml/datasets/palm_hegu_pose.yaml` 数据配置模板。
- 新增 `ml/requirements.txt` 与 `ml/README.md`。

### 二、前端改造（切换到后端推理）

- 更新 `index.html`：移除旧前端模型依赖，补充失败原因展示区域。
- 更新 `script.js`：
  - 移除 Handpose + OpenCV Worker 前端推理逻辑
  - 新增上传当前帧到 `/api/infer` 的流程
  - 新增识别结果绘制、置信度更新、失败原因展示
  - 新增 `/api/health` 服务状态检测
- 更新 `style.css`：适配新页面结构与失败原因列表样式。

### 三、项目瘦身与整理

- 已扫描并清理本地模型文件：当前仓库中未保留本地模型权重（如 `.pt/.onnx/.task`）。
- 清理临时缓存产物（如 `__pycache__` 中的编译缓存）。
- `README.md` 已同步今天的改造内容与启动方式。
