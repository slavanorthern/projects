#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from catboost import CatBoostRegressor

# Путь к сохраненной модели и тестовым данным
model_save_path = 'best_model.cbm'
test_data_path = 'test.csv'
submission_save_path = 'submission.csv'  # Путь, куда сохранять предсказания

# Загрузка модели
model = CatBoostRegressor()
model.load_model(model_save_path)

# Загрузка тестовых данных
test_data = pd.read_csv(test_data_path)
X_test = test_data.drop(columns=['id'])

# Генерация предсказаний
predictions = model.predict(X_test)

# Создание файла предсказаний
submission = pd.DataFrame({
    'id': test_data['id'],
    'score': predictions
})

# Сохранение файла предсказаний
submission.to_csv(submission_save_path, index=False)
print(f'Файл предсказаний сохранён как {submission_save_path}')


# In[ ]:




