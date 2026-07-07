"""
دمج نتايج كذا صورة في رد واحد (متوسط).
"""

from typing import Any, Dict, List

import numpy as np

from config import DISEASE_CLASSES


def aggregate_upload_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    بياخد متوسط متجهات الاحتمالات (probabilities) عبر كل الصور،
    وبعدين يختار المرض صاحب أعلى احتمال في المتوسط - ده أدق من متوسط
    الـ confidence لوحدها لأنه بياخد بالِه من توزيع الاحتمالات كامل مش رقم واحد بس.
    """
    prob_matrix = np.array([r["probabilities"] for r in results])
    avg_probs = prob_matrix.mean(axis=0)

    predicted_idx = int(avg_probs.argmax())
    disease_name = DISEASE_CLASSES[predicted_idx]
    confidence = float(avg_probs[predicted_idx] * 100)

    return {
        "images_processed": len(results),
        "disease": disease_name,
        "confidence": round(confidence, 2),
        "is_healthy": disease_name.lower() == "healthy",
        "per_image": [
            {"disease": r["disease"], "confidence": round(r["confidence"], 2)}
            for r in results
        ],
    }


def aggregate_camera_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    بيجمع كل الورق المكتشف من كل الصور المبعوتة، ويحسب متوسط anomaly_score
    على مستوى كل الورق (مش على مستوى الصور)، عشان صورة فيها ورقتين
    توزن أكتر من صورة فيها ورقة واحدة.
    """
    included = [r for r in results if not r["excluded"]]
    excluded = [r for r in results if r["excluded"]]

    all_scores = [leaf["anomaly_score"] for r in included for leaf in r["leaves"]]
    total_leaves = len(all_scores)

    average_score = float(np.mean(all_scores)) if total_leaves else None
    overall_healthy = (average_score > 0) if average_score is not None else None

    return {
        "images_processed": len(results),
        "images_included": len(included),
        "images_excluded": len(excluded),
        "exclude_reasons": [r["exclude_reason"] for r in excluded],
        "total_leaves_detected": total_leaves,
        "average_anomaly_score": round(average_score, 4) if average_score is not None else None,
        "overall_healthy": overall_healthy,
        "per_image": [
            {
                "plantnet_status": r["plantnet_status"],
                "is_tomato": r["is_tomato"],
                "detected_species": r["detected_species"],
                "excluded": r["excluded"],
                "exclude_reason": r["exclude_reason"],
                "leaves": r["leaves"],
                "leaves_filtered_out": r["leaves_filtered_out"],
            }
            for r in results
        ],
    }
