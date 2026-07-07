"""
دوال التنبؤ باستخدام الموديلات المحمّلة مرة واحدة بس عند بدء تشغيل السيرفر.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from config import DISEASE_CLASSES
from model_loader import (
    DEVICE,
    disease_transform,
    load_anomaly_detector,
    load_disease_model,
    load_feature_extractor,
)

disease_model = load_disease_model()
feature_extractor, feature_transform = load_feature_extractor()
anomaly_detector = load_anomaly_detector()


def predict_disease(image: Image.Image):
    """Step 3 (upload path): بيرجع اسم المرض، نسبة الثقة، ومتجه الاحتمالات كامل (لازم للـ averaging)."""
    tensor = disease_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = disease_model(tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    predicted_idx = int(probs.argmax())
    class_name = DISEASE_CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx] * 100)
    return class_name, confidence, probs.tolist()


def predict_leaf_health(leaf_image: Image.Image):
    """Step 3 (camera path): بيرجع هل الورقة سليمة ولا شاذة (anomaly)، ودرجة الثقة الموقّعة."""
    tensor = feature_transform(leaf_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = feature_extractor(tensor).cpu().numpy()

    prediction = anomaly_detector.predict(features)[0]      # 1 = سليم، -1 = شاذ
    score = anomaly_detector.decision_function(features)[0]  # موجب = سليم، سالب = شاذ
    return bool(prediction == 1), float(score)
