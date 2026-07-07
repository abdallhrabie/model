"""
معالجة صورة واحدة حسب مصدرها، مصمّمة عشان تتنادى لكل صورة في القايمة
اللي المستخدم بعتها، وبعدين نجمع النتايج كلها في aggregation.py
"""

import base64
from typing import Any, Dict

from PIL import Image

from config import MIN_LEAF_AREA_RATIO, MIN_LEAF_COMPLETENESS, MIN_LEAF_CONFIDENCE
from external_apis import check_is_tomato, get_leaf_predictions
from image_utils import crop_leaves_from_predictions
from model_predict import predict_disease, predict_leaf_health


def process_upload_image(pil_image: Image.Image) -> Dict[str, Any]:
    """Step 3 مباشرة، لصورة upload واحدة."""
    disease_name, confidence, probs = predict_disease(pil_image)
    return {
        "disease": disease_name,
        "confidence": confidence,
        "probabilities": probs,
        "is_healthy": disease_name.lower() == "healthy",
    }


def process_camera_image(pil_image: Image.Image, image_bytes: bytes) -> Dict[str, Any]:
    """
    Step 1 -> Step 2 -> Step 3 لصورة كاميرا واحدة.
    لو PlantNet فشل تقنيًا (اتصال، DNS، إلخ) بنتخطى الخطوة دي بس ونكمل الباقي عادي.
    لو PlantNet رد بنجاح وقال إنها مش طماطم، الصورة دي بتتستبعد من التجميع.
    """
    result: Dict[str, Any] = {
        "plantnet_status": "ok",
        "is_tomato": None,
        "detected_species": None,
        "plantnet_score": None,
        "leaves": [],
        "leaves_filtered_out": [],
        "excluded": False,
        "exclude_reason": None,
    }

    # ---- Step 1: PlantNet (اختياري - فشله التقني مش بيوقف الصورة) ----
    try:
        is_tomato, sci_name, plantnet_score = check_is_tomato(image_bytes)
        result["is_tomato"] = is_tomato
        result["detected_species"] = sci_name
        result["plantnet_score"] = plantnet_score

        if not is_tomato:
            result["excluded"] = True
            result["exclude_reason"] = "الصورة مش لنبات طماطم حسب PlantNet"
            return result
    except Exception as e:
        result["plantnet_status"] = f"skipped: {e}"
        # مفيش return هنا، بنكمل عادي على step 2

    # ---- Step 2: قص الورق (مع فلترة الثقة/المساحة/الاكتمال) ----
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        predictions = get_leaf_predictions(image_b64)
        leaf_crops, filtered_out = crop_leaves_from_predictions(
            pil_image,
            predictions,
            min_confidence=MIN_LEAF_CONFIDENCE,
            min_area_ratio=MIN_LEAF_AREA_RATIO,
            min_completeness=MIN_LEAF_COMPLETENESS,
        )
        result["leaves_filtered_out"] = filtered_out
    except Exception as e:
        result["excluded"] = True
        result["exclude_reason"] = f"فشل قص الورق: {e}"
        return result

    if not leaf_crops:
        result["excluded"] = True
        result["exclude_reason"] = "متعرفش أي ورق مقبول في الصورة (كل الورق اتفلتر أو مفيش ورق أصلًا)"
        return result

    # ---- Step 3: anomaly detector لكل ورقة ----
    for idx, leaf in enumerate(leaf_crops):
        is_healthy, score = predict_leaf_health(leaf)
        result["leaves"].append({
            "leaf_index": idx,
            "is_healthy": is_healthy,
            "anomaly_score": score,
        })

    return result
