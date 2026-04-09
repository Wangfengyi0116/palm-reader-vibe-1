from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from infer import run_inference


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEG_WEIGHTS = BASE_DIR / "runs" / "palm_seg" / "weights" / "best.pt"
DEFAULT_POSE_WEIGHTS = BASE_DIR / "runs" / "palm_pose" / "weights" / "best.pt"


class ModelStore:
    def __init__(self) -> None:
        self.seg_model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None
        self.seg_path: Optional[Path] = None
        self.pose_path: Optional[Path] = None

    def load(self, seg_weights: Path, pose_weights: Path) -> None:
        if self.seg_model is None or self.seg_path != seg_weights:
            self.seg_model = YOLO(str(seg_weights))
            self.seg_path = seg_weights
        if self.pose_model is None or self.pose_path != pose_weights:
            self.pose_model = YOLO(str(pose_weights))
            self.pose_path = pose_weights


store = ModelStore()
app = FastAPI(title="Palm Reader Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "palm-ml-api"}


@app.post("/api/infer")
async def infer_image(
    file: UploadFile = File(...),
    seg_weights: str = Form(str(DEFAULT_SEG_WEIGHTS)),
    pose_weights: str = Form(str(DEFAULT_POSE_WEIGHTS)),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
):
    seg_path = Path(seg_weights)
    pose_path = Path(pose_weights)
    if not seg_path.exists():
        raise HTTPException(status_code=400, detail=f"Seg weights not found: {seg_path}")
    if not pose_path.exists():
        raise HTTPException(status_code=400, detail=f"Pose weights not found: {pose_path}")

    data = await file.read()
    image_arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法读取上传图片")

    try:
        store.load(seg_path, pose_path)
        result = run_inference(
            image_bgr=image,
            seg_model=store.seg_model,
            pose_model=store.pose_model,
            conf=conf,
            iou=iou,
        )
        result.pop("_vis_image_bgr", None)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理异常: {exc}") from exc
