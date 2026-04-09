# YOLO 掌纹模型方案

该目录为你重新搭建的深度学习识别方案，包含：

- `train.py`：训练脚本
- `infer.py`：离线推理脚本（输出可视化和 JSON）
- `api.py`：FastAPI 推理服务（前端可直接调用）
- `datasets/*.yaml`：数据配置模板

## 1) 安装依赖

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 数据准备

### 分割数据（手掌 + 三条线）

```
ml/data/seg/
  images/train|val|test
  labels/train|val|test
```

类别：
- `0: palm`
- `1: heart_line`
- `2: head_line`
- `3: life_line`

### 虎口关键点数据（Pose）

```
ml/data/pose/
  images/train|val|test
  labels/train|val|test
```

关键点定义：`kpt_shape: [1, 3]`（1个点：x,y,visible）

## 3) 训练

```bash
python train.py --task all --epochs 120 --imgsz 960 --batch 8 --device 0
```

输出权重：
- `runs/palm_seg/weights/best.pt`
- `runs/palm_pose/weights/best.pt`

## 4) 离线推理

```bash
python infer.py ^
  --image ..\test.jpg ^
  --seg-weights runs\palm_seg\weights\best.pt ^
  --pose-weights runs\palm_pose\weights\best.pt ^
  --save-vis outputs\predict_vis.jpg ^
  --save-json outputs\predict.json
```

推理失败会给详细原因（如过暗、过曝、模糊、手掌未检测、三线缺失、虎口点低置信等）。

## 5) 启动 API 服务

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

接口：
- `GET /api/health`
- `POST /api/infer`（上传 `file`）

