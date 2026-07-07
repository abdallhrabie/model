"""
تحميل الموديلات مرة واحدة بس عند بدء تشغيل السيرفر، وتجهيز الـ preprocessing
بنفس الطريقة اللي اتدرب بيها كل موديل بالظبط.
"""

import joblib
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from config import DISEASE_MODEL_PATH, ANOMALY_MODEL_PATH, DISEASE_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Preprocessing بتاع موديل كشف المرض (tomato_model.h5)
# نفس الـ transform المستخدم وقت التدريب بالظبط (شوف كود التدريب الأصلي)
# ============================================================
disease_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def load_disease_model():
    """موديل MobileNetV2 معدّل بـ 11 class، لصور الـ upload."""
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(DISEASE_CLASSES))
    state_dict = torch.load(DISEASE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_feature_extractor():
    """
    نفس MobileNetV2 المستخدم في تدريب healthy_anomaly_detector:
    أوزان ImageNet جاهزة، وشيل آخر طبقة تصنيف (نستخدمه كمستخرج خصائص بس).
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()
    model.to(DEVICE)
    model.eval()
    feature_transform = weights.transforms()  # نفس الـ preprocessing المستخدم وقت التدريب
    return model, feature_transform


def load_anomaly_detector():
    """OneClassSVM المدرب على خصائص صور الورق السليم، لصور الكاميرا."""
    return joblib.load(ANOMALY_MODEL_PATH)
