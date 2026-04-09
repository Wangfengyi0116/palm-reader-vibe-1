import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ["palm", "heart_line", "head_line", "life_line"]
LINE_NAME_MAP = {1: "heart_line", 2: "head_line", 3: "life_line"}
LINE_COLOR_MAP = {
    "heart_line": (48, 59, 255),
    "head_line": (100, 217, 76),
    "life_line": (255, 122, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run palm line + hegu inference.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--seg-weights", required=True, help="Seg model weights path")
    parser.add_argument("--pose-weights", required=True, help="Pose model weights path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save-vis", default="outputs/predict_vis.jpg")
    parser.add_argument("--save-json", default="outputs/predict.json")
    return parser.parse_args()


def image_quality_check(image_bgr: np.ndarray) -> List[Dict]:
    reasons = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_brightness < 45:
        reasons.append(
            {
                "code": "LOW_LIGHT",
                "message": "图像过暗，掌纹对比度不足。建议提高环境光或开启补光。",
                "metric": {"mean_brightness": round(mean_brightness, 2), "threshold": 45},
            }
        )
    elif mean_brightness > 215:
        reasons.append(
            {
                "code": "OVER_EXPOSURE",
                "message": "图像过亮或高反光，导致掌纹细节丢失。建议降低曝光或调整角度。",
                "metric": {"mean_brightness": round(mean_brightness, 2), "threshold": 215},
            }
        )

    if blur_score < 55:
        reasons.append(
            {
                "code": "BLURRY_IMAGE",
                "message": "图像模糊，无法稳定提取掌纹。建议保持手和相机静止后重拍。",
                "metric": {"laplacian_var": round(blur_score, 2), "threshold": 55},
            }
        )

    return reasons


def collect_seg_predictions(result, image_shape: Tuple[int, int]) -> Dict:
    h, w = image_shape[:2]
    out = {
        "palm": {"detected": False, "confidence": 0.0, "polygon": []},
        "heart_line": {"detected": False, "confidence": 0.0, "polygon": []},
        "head_line": {"detected": False, "confidence": 0.0, "polygon": []},
        "life_line": {"detected": False, "confidence": 0.0, "polygon": []},
    }

    if result.boxes is None or len(result.boxes) == 0:
        return out

    masks_xy = result.masks.xy if result.masks is not None else [None] * len(result.boxes)
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        name = CLASS_NAMES[cls_id]
        if conf < out[name]["confidence"]:
            continue

        poly = []
        if i < len(masks_xy) and masks_xy[i] is not None:
            pts = np.asarray(masks_xy[i], dtype=np.float32)
            if len(pts) > 0:
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                poly = pts.astype(int).tolist()

        out[name] = {"detected": True, "confidence": conf, "polygon": poly}

    return out


def collect_hegu_prediction(result) -> Dict:
    out = {"detected": False, "confidence": 0.0, "point": []}
    if result.keypoints is None or result.keypoints.xy is None:
        return out
    if len(result.keypoints.xy) == 0:
        return out

    points = result.keypoints.xy[0].cpu().numpy()
    confs = (
        result.keypoints.conf[0].cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(points.shape[0], dtype=np.float32)
    )
    if len(points) == 0:
        return out

    p = points[0]
    c = float(confs[0])
    return {"detected": c >= 0.2, "confidence": c, "point": [float(p[0]), float(p[1])]}


def polygon_area(poly: List[List[int]]) -> float:
    if len(poly) < 3:
        return 0.0
    pts = np.asarray(poly, dtype=np.float32)
    return float(abs(cv2.contourArea(pts)))


def explain_failures(image_shape, seg_pred, hegu_pred) -> List[Dict]:
    reasons = []
    h, w = image_shape[:2]
    image_area = float(h * w)

    if not seg_pred["palm"]["detected"] or seg_pred["palm"]["confidence"] < 0.30:
        reasons.append(
            {
                "code": "PALM_NOT_DETECTED",
                "message": "未可靠识别到手掌区域，建议手掌完整入镜并避免背景干扰。",
                "metric": {"palm_confidence": round(seg_pred["palm"]["confidence"], 3), "threshold": 0.30},
            }
        )
        return reasons

    palm_area = polygon_area(seg_pred["palm"]["polygon"])
    palm_ratio = palm_area / image_area if image_area > 0 else 0.0
    if palm_ratio < 0.08:
        reasons.append(
            {
                "code": "PALM_TOO_SMALL",
                "message": "手掌区域过小，掌纹像素不足。建议靠近镜头重拍。",
                "metric": {"palm_area_ratio": round(palm_ratio, 4), "threshold": 0.08},
            }
        )

    missing_lines = []
    for line_name in ["heart_line", "head_line", "life_line"]:
        if not seg_pred[line_name]["detected"] or seg_pred[line_name]["confidence"] < 0.25:
            missing_lines.append(line_name)

    if len(missing_lines) == 3:
        reasons.append(
            {
                "code": "LINES_NOT_DETECTED",
                "message": "三条主线都未被可靠识别，可能由光照、模糊或手掌角度导致。",
                "metric": {
                    "heart_conf": round(seg_pred["heart_line"]["confidence"], 3),
                    "head_conf": round(seg_pred["head_line"]["confidence"], 3),
                    "life_conf": round(seg_pred["life_line"]["confidence"], 3),
                    "threshold": 0.25,
                },
            }
        )
    elif len(missing_lines) > 0:
        reasons.append(
            {
                "code": "PARTIAL_LINES_MISSING",
                "message": f"部分掌纹未可靠识别：{', '.join(missing_lines)}。建议调整手掌平整度与光线方向。",
                "metric": {"missing_lines": missing_lines},
            }
        )

    if not hegu_pred["detected"] or hegu_pred["confidence"] < 0.25:
        reasons.append(
            {
                "code": "HEGU_NOT_DETECTED",
                "message": "虎口关键点置信度不足，建议拇指与食指自然张开。",
                "metric": {"hegu_confidence": round(hegu_pred["confidence"], 3), "threshold": 0.25},
            }
        )

    return reasons


def draw_predictions(image_bgr: np.ndarray, seg_pred: Dict, hegu_pred: Dict) -> np.ndarray:
    vis = image_bgr.copy()

    def draw_poly(poly, color, label, conf):
        if not poly:
            return
        pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
        x, y = pts[0, 0, 0], pts[0, 0, 1]
        cv2.putText(vis, f"{label}:{conf:.2f}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if seg_pred["palm"]["detected"]:
        draw_poly(seg_pred["palm"]["polygon"], (255, 255, 0), "palm", seg_pred["palm"]["confidence"])
    for cls_id, name in LINE_NAME_MAP.items():
        if seg_pred[name]["detected"]:
            draw_poly(seg_pred[name]["polygon"], LINE_COLOR_MAP[name], name, seg_pred[name]["confidence"])

    if hegu_pred["detected"] and len(hegu_pred["point"]) == 2:
        x, y = int(hegu_pred["point"][0]), int(hegu_pred["point"][1])
        cv2.circle(vis, (x, y), 7, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"hegu:{hegu_pred['confidence']:.2f}",
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            2,
        )
    return vis


def run_inference(image_bgr: np.ndarray, seg_model: YOLO, pose_model: YOLO, conf: float = 0.25, iou: float = 0.5):
    quality_reasons = image_quality_check(image_bgr)
    seg_result = seg_model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)[0]
    pose_result = pose_model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)[0]

    seg_pred = collect_seg_predictions(seg_result, image_bgr.shape)
    hegu_pred = collect_hegu_prediction(pose_result)
    fail_reasons = explain_failures(image_bgr.shape, seg_pred, hegu_pred)
    all_reasons = quality_reasons + fail_reasons

    return {
        "success": len(all_reasons) == 0,
        "predictions": {
            "palm": seg_pred["palm"],
            "heart_line": seg_pred["heart_line"],
            "head_line": seg_pred["head_line"],
            "life_line": seg_pred["life_line"],
            "hegu": hegu_pred,
        },
        "reasons": all_reasons,
        "advice": "识别失败时请优先检查光照、对焦和手掌占画面比例。",
        "_vis_image_bgr": draw_predictions(image_bgr, seg_pred, hegu_pred),
    }


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")

    seg_model = YOLO(args.seg_weights)
    pose_model = YOLO(args.pose_weights)
    result = run_inference(image, seg_model, pose_model, args.conf, args.iou)
    vis = result.pop("_vis_image_bgr")

    vis_path = Path(args.save_vis)
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), vis)

    output = dict(result)
    output["image"] = str(image_path)
    output["visualization"] = str(vis_path)

    json_path = Path(args.save_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
