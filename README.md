# 掌纹主线自动化识别系统

当前版本已从前端启发式识别升级为 **YOLO 后端推理 + 前端可视化** 架构。

## 当前能力

- 支持摄像头实时采集与本地图片上传
- 识别并绘制：手掌区域、感情线、智慧线、生命线、虎口点
- 输出三线置信度与推理耗时
- 给出详细失败原因（过暗、过亮、模糊、手掌未检测、部分线缺失等）

## 项目结构

- `index.html`：前端页面结构
- `script.js`：前端交互与 API 调用（`/api/health`、`/api/infer`）
- `style.css`：前端样式
- `ml/train.py`：YOLO 训练脚本（分割 + 关键点）
- `ml/infer.py`：离线推理脚本（输出可视化与 JSON）
- `ml/api.py`：FastAPI 推理服务
- `ml/datasets/*.yaml`：数据配置模板
- `ml/README.md`：ML 子模块说明
- `docs/design.md`：设计文档

## 快速启动

### 1) 启动后端推理服务

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8001
```

### 2) 启动前端

```bash
cd ..
python -m http.server 8000
```

浏览器打开：`http://localhost:8000`

## 今日修改（2026-03-30）

- 重建 `ml/` 深度学习模块（训练、推理、API、数据配置）
- 前端改造为调用后端 YOLO 推理接口（移除前端模型推理链路）
- 清理本地模型与缓存，减轻项目负担（当前仓库无本地模型权重文件）
- 新增并更新 `CHANGELOG.md` 记录变更详情