import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

print("Загрузка датасетов")
df_sessions = pd.read_csv("D:/DATA_сеты/Final_work_'Data_Science'/ga_sessions.csv", low_memory=False)
df_hist = pd.read_csv("D:/DATA_сеты/Final_work_'Data_Science'/ga_hits.csv", low_memory=False)

print("Базовая очистка и разметка таргета")
df_sessions = df_sessions.drop_duplicates()
df_hist = df_hist.drop_duplicates()

target_actions = [
    'sub_car_claim_click', 'sub_car_claim_submit_click',
    'sub_open_dialog_click', 'sub_custom_question_submit_click',
    'sub_call_number_click', 'sub_callback_submit_click',
    'sub_submit_success', 'sub_car_request_submit_click'
]

target_sessions = df_hist[df_hist['event_action'].isin(target_actions)]['session_id'].unique()
df_sessions['target'] = df_sessions['session_id'].isin(target_sessions).astype(int)

# Заполняем пропуски
df_sessions['utm_source'] = df_sessions['utm_source'].fillna('unknown')
df_sessions['utm_medium'] = df_sessions['utm_medium'].fillna('unknown')
df_sessions['geo_city'] = df_sessions['geo_city'].fillna('unknown')
df_sessions['device_category'] = df_sessions['device_category'].fillna('unknown')

print("Частотное кодирование текстов в числа...")
categorical_features = ['utm_source', 'utm_medium', 'device_category', 'geo_city']

# Сюда сохраним наши словари перевода текста в числа
encoding_maps = {}

# Для каждой текстовой колонки считаем частоту появления значений
for col in categorical_features:
    freq_map = df_sessions[col].value_counts(normalize=True).to_dict()
    encoding_maps[col] = freq_map
    # Заменяем текст на число-частоту
    df_sessions[col] = df_sessions[col].map(freq_map)

# Выделяем фичи
features_list = ['visit_number', 'utm_source', 'utm_medium', 'device_category', 'geo_city']
X = df_sessions[features_list]
y = df_sessions['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Обучение")
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

model = LGBMClassifier(n_estimators=150, scale_pos_weight=ratio, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Сохранение в model")
if not os.path.exists("model"):
    os.makedirs("model")

# Сохраняем в один файл словари кодирования И саму модель как кортеж (tuple)
artifacts = (model, encoding_maps)
joblib.dump(artifacts, "model/best_lightgbm_model.joblib", compress=3)

print("Обучение завершено")
