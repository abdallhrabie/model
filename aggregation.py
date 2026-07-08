"""
دمج نتايج كذا صورة في رد واحد مبسّط:
عدد الصور، عدد الورق، النتيجة (سليم/مرض)، نسبة الثقة، ونسبة الخطورة لو فيه مرض.
"""

from typing import Any, Dict, List

import numpy as np

from config import DISEASE_CLASSES

HEALTHY_IDX = DISEASE_CLASSES.index("healthy")


def aggregate_upload_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    متوسط متجهات الاحتمالات (probabilities) عبر كل الصور المرفوعة،
    وبعدين استخراج النتيجة النهائية منه.
    """
    prob_matrix = np.array([r["probabilities"] for r in results])
    avg_probs = prob_matrix.mean(axis=0)

    predicted_idx = int(avg_probs.argmax())
    disease_name = DISEASE_CLASSES[predicted_idx]
    is_healthy = disease_name.lower() == "healthy"

    confidence = round(float(avg_probs[predicted_idx] * 100), 2)

    response: Dict[str, Any] = {
        "images_count": len(results),
        "leaves_count": len(results),  # كل صورة upload = ورقة واحدة جاهزة
        "result": "healthy" if is_healthy else disease_name,
        "confidence": confidence,
    }

    if not is_healthy:
        # نسبة الخطورة = احتمال إن الورقة مش سليمة إجمالًا (كل احتمالات الأمراض مع بعض)
        risk = round(float((1 - avg_probs[HEALTHY_IDX]) * 100), 2)
        response["risk_percentage"] = risk

    return response


def aggregate_camera_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    بيجمع كل الورق المكتشف من كل الصور، ويحسب متوسط anomaly_score على مستوى الورقة،
    وبيحوّله لنسبة ثقة ونسبة خطورة مفهومة (بدل الرقم الخام غير المحدود).
    """
    included = [r for r in results if not r["excluded"]]

    all_scores = [leaf["anomaly_score"] for r in included for leaf in r["leaves"]]
    leaves_count = len(all_scores)

    response: Dict[str, Any] = {
        "images_count": len(results),
        "leaves_count": leaves_count,
    }

    if leaves_count == 0:
        response["result"] = "unknown"
        response["confidence"] = 0.0
        return response

    average_score = float(np.mean(all_scores))

    # تحويل الـ anomaly score (غير محدود) لاحتمالية سليم بين 0 و1 (logistic squashing)
    prob_healthy = 1.0 / (1.0 + np.exp(-average_score))
    is_healthy = prob_healthy >= 0.5

    confidence = round(float(max(prob_healthy, 1 - prob_healthy) * 100), 2)

    response["result"] = "healthy" if is_healthy else "unhealthy"
    response["confidence"] = confidence

    if not is_healthy:
        response["risk_percentage"] = round(float((1 - prob_healthy) * 100), 2)

    return response
