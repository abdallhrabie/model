# Tomato Health Check API

## 1. تجهيز البيئة
```bash
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. حط ملفات الموديلات
انسخ الملفين دول جوه مجلد `models/`:
```
tomato_api/
  models/
    tomato_model.h5                  <- موديل كشف المرض (upload)
    healthy_anomaly_detector.joblib  <- anomaly detector (camera)
```

## 3. شغّل السيرفر
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

هيبقى شغال على: `http://127.0.0.1:8000`
وتقدر تفتح `http://127.0.0.1:8000/docs` عشان تجرب الـ API من متصفح (Swagger UI جاهز تلقائي مع FastAPI).

## 4. تجربة سريعة (curl) - أكتر من صورة في نفس الطلب

### صور upload (من المعرض):
```bash
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -F "images=@C:\path\to\leaf1.jpg" ^
  -F "images=@C:\path\to\leaf2.jpg" ^
  -F "source=upload"
```

### صور camera (لسه في الشجرة، محتاجة قص ورق):
```bash
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -F "images=@C:\path\to\plant1.jpg" ^
  -F "images=@C:\path\to\plant2.jpg" ^
  -F "source=camera"
```

**ملحوظة:** تقدر تبعت صورة واحدة أو عدد أكبر، الـ key دايمًا اسمه `images` (بالجمع) في كل الحالات.

### في Postman
في تبويب Body -> form-data:
- ضيف صف باسم `images`، نوعه **File**، واختار أول صورة
- ضيف صف تاني بنفس الاسم بالظبط `images`، نوعه **File**، واختار الصورة التانية
- كرر لأي عدد صور عايزه (كلهم بنفس اسم الحقل `images`)
- وضيف صف `source` (Text) بقيمة `camera` أو `upload`

## 5. شكل الرد

### لو source=upload:
```json
{
  "source": "upload",
  "images_processed": 3,
  "disease": "Early_blight",
  "confidence": 82.4,
  "is_healthy": false,
  "per_image": [
    {"disease": "Early_blight", "confidence": 79.1},
    {"disease": "Early_blight", "confidence": 85.0},
    {"disease": "Early_blight", "confidence": 83.1}
  ]
}
```

### لو source=camera:
```json
{
  "source": "camera",
  "images_processed": 2,
  "images_included": 2,
  "images_excluded": 0,
  "exclude_reasons": [],
  "total_leaves_detected": 14,
  "average_anomaly_score": 0.32,
  "overall_healthy": true,
  "per_image": [ ... تفاصيل كل صورة والورق اللي فيها ... ]
}
```

لو PlantNet فشل تقنيًا (مشكلة نت/DNS) في أي صورة، `plantnet_status` هيبقى `"skipped: <سبب الخطأ>"` جوه تفاصيل الصورة دي، والـ pipeline هيكمل عادي على step 2 و3 من غير ما يوقف الطلب كله.

## 6. من الموبايل (Flutter / React Native / إلخ)
الطلب لازم يبقى `multipart/form-data` فيه:
- `images`: ملف أو أكتر (كل الصور بنفس اسم الحقل `images`)
- `source`: النص `"camera"` أو `"upload"` حسب مصدر الصور

## ملاحظات أمان قبل النشر (production)
- انقل `PLANTNET_API_KEY` و`ROBOFLOW_API_KEY` من `config.py` لمتغيرات بيئة (`.env`) بدل ما يفضلوا مكتوبين في الكود.
- ضيف حد أقصى لحجم الصورة المرفوعة.
- لو هتستضيف السيرفر على استضافة فيها GPU، هيبقى الأداء أسرع بكتير من CPU.
