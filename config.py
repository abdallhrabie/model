"""
إعدادات المشروع: مفاتيح الـ APIs الخارجية، مسارات الموديلات، وأسماء الأمراض.
"""

# ---------------- PlantNet (Step 1: تأكيد إن النبات طماطم) ----------------
PLANTNET_API_KEY = "2b10ofHx591TKtn7On84ZTUZUO"
PLANTNET_PROJECT = "all"

# ---------------- Roboflow (Step 2: قص الورق من صورة الكاميرا) ----------------
ROBOFLOW_API_KEY = "M4R6LrwAnDzY1fNQDZSC"
ROBOFLOW_WORKSPACE = "ahmed-ragab-t5yhy"
ROBOFLOW_WORKFLOW_ID = "general-segmentation-api-7"

# ---------------- فلترة الورق المكتشف قبل ما يدخل step 3 ----------------
MIN_LEAF_CONFIDENCE = 0.5     # اقل ثقة مقبولة من Roboflow لكل ورقة (0 -> 1)
MIN_LEAF_AREA_RATIO = 0.01    # اقل نسبة مساحة الورقة لمساحة الصورة الكاملة
MIN_LEAF_COMPLETENESS = 0.5   # اقل درجة اكتمال (1.0 = الورقة كاملة جوه الصورة، مش ملامسة أي حافة)

DISEASE_MODEL_PATH = "models/tomato_model.h5"          # PyTorch state_dict (MobileNetV2) - لصور الـ upload
ANOMALY_MODEL_PATH = "models/healthy_anomaly_detector.joblib"  # OneClassSVM - لصور الكاميرا

# ---------------- أسماء الأمراض (بنفس الترتيب اللي اتدرب بيه الموديل) ----------------
# الترتيب ده جاي من train_data.classes بتاع ImageFolder (ترتيب أبجدي تلقائي)
DISEASE_CLASSES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew",
]
