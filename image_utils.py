"""
قص كل ورقة اتكشفت في صورة الكاميرا، باستخدام الـ polygon points
الراجعة من Roboflow segmentation workflow، مع فلترة الورق الضعيف/الصغير/الناقص.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def _leaf_metrics(polygon: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    بترجع:
      area_ratio    -> مساحة الورقة نسبة لمساحة الصورة الكاملة
      completeness  -> 1.0 لو الورقة كاملة جوه الصورة، وبتقل 0.25 عن كل حافة
                       (يمين/شمال/فوق/تحت) الورقة لامستها (يعني ممكن تكون مقطوعة)
    """
    area = cv2.contourArea(polygon)
    area_ratio = area / float(img_w * img_h)

    x_min, y_min = polygon[:, 0].min(), polygon[:, 1].min()
    x_max, y_max = polygon[:, 0].max(), polygon[:, 1].max()

    touches = 0
    touches += 1 if x_min <= 0 else 0
    touches += 1 if y_min <= 0 else 0
    touches += 1 if x_max >= img_w - 1 else 0
    touches += 1 if y_max >= img_h - 1 else 0

    completeness = 1.0 - (touches * 0.25)
    return area_ratio, completeness


def crop_leaves_from_predictions(
    image: Image.Image,
    predictions: List[Dict[str, Any]],
    mask_background: bool = True,
    min_confidence: float = 0.5,
    min_area_ratio: float = 0.01,
    min_completeness: float = 0.5,
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """
    بترجع (leaf_crops, filtered_out):
      leaf_crops   -> صور PIL للورق اللي عدّى الفلترة، جاهزة تدخل step 3
      filtered_out -> تفاصيل أي ورقة اتشالت وسبب الاستبعاد (مفيدة للتشخيص/اللوج)
    """
    img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_np.shape[:2]

    leaf_crops: List[Image.Image] = []
    filtered_out: List[Dict[str, Any]] = []

    for idx, pred in enumerate(predictions):
        points = pred.get("points")
        if not points:
            filtered_out.append({"index": idx, "reason": "من غير polygon points"})
            continue

        confidence = float(pred.get("confidence", 1.0))
        if confidence < min_confidence:
            filtered_out.append({
                "index": idx,
                "reason": f"ثقة منخفضة ({confidence:.2f} < {min_confidence})",
            })
            continue

        polygon = np.array(
            [[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32
        )

        area_ratio, completeness = _leaf_metrics(polygon, w, h)

        if area_ratio < min_area_ratio:
            filtered_out.append({
                "index": idx,
                "reason": f"مساحة صغيرة جدًا ({area_ratio:.4f} < {min_area_ratio})",
            })
            continue

        if completeness < min_completeness:
            filtered_out.append({
                "index": idx,
                "reason": f"ورقة ناقصة/مقطوعة من الحافة (completeness={completeness:.2f} < {min_completeness})",
            })
            continue

        x_min = max(int(polygon[:, 0].min()), 0)
        x_max = min(int(polygon[:, 0].max()), w)
        y_min = max(int(polygon[:, 1].min()), 0)
        y_max = min(int(polygon[:, 1].max()), h)

        if x_max <= x_min or y_max <= y_min:
            filtered_out.append({"index": idx, "reason": "bounding box غير صالح"})
            continue

        if mask_background:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            masked = np.full_like(img_np, 255)  # خلفية بيضا
            masked[mask == 255] = img_np[mask == 255]
            cropped = masked[y_min:y_max, x_min:x_max]
        else:
            cropped = img_np[y_min:y_max, x_min:x_max]

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        leaf_crops.append(Image.fromarray(cropped_rgb))

    return leaf_crops, filtered_out
