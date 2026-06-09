from fastapi import FastAPI, Request
import joblib
import pandas as pd
import time

app = FastAPI(
    title="Сбер Автоподписка — Сервис предсказания конверсий",
    description="Промышленный API на базе числовых частотных признаков."
)

# Загружаем артефакты
MODEL_PATH = "model/best_lightgbm_model.joblib"
model, encoding_maps = joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Sber Autopodpiska Predictor"}


@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()

    # Получаем JSON
    payload = await request.json()

    # Извлекаем текстовые значения из JSON
    visit_number = int(payload.get("visit_number", 1))
    utm_source_str = str(payload.get("utm_source", "unknown"))
    utm_medium_str = str(payload.get("utm_medium", "unknown"))
    device_category_str = str(payload.get("device_category", "unknown"))
    geo_city_str = str(payload.get("geo_city", "unknown"))

    # Меняем текст на числа по нашим словарям
    utm_source_num = encoding_maps["utm_source"].get(utm_source_str, 0.0)
    utm_medium_num = encoding_maps["utm_medium"].get(utm_medium_str, 0.0)
    device_category_num = encoding_maps["device_category"].get(device_category_str, 0.0)
    geo_city_num = encoding_maps["geo_city"].get(geo_city_str, 0.0)

    # Формируем словарь, состоящий строго из чисел
    numeric_data = {
        "visit_number": visit_number,
        "utm_source": utm_source_num,
        "utm_medium": utm_medium_num,
        "device_category": device_category_num,
        "geo_city": geo_city_num
    }

    # Превращаем в DataFrame из одной строки
    input_df = pd.DataFrame([numeric_data])
    features_list = ['visit_number', 'utm_source', 'utm_medium', 'device_category', 'geo_city']

    # Делаем предсказание и берем первый элемент, превращая его в число
    raw_prediction = model.predict(input_df[features_list])
    prediction = int(raw_prediction[0])

    latency_seconds = time.time() - start_time

    return {
        "result": prediction,
        "latency_seconds": round(latency_seconds, 5),
        "status": "Успешно уложились в лимит 3 сек" if latency_seconds < 3.0 else "Превышен лимит скорости"
    }
