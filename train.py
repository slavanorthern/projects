#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Загрузка обучающих данных
train_data_path = 'train.csv' #Путь к файлу
train_data = pd.read_csv(train_data_path)

# Путь для сохранения обученной модели
model_save_path = 'best_model.cbm'


# Подготовка данных
X = train_data.drop(columns=['id', 'score'])
y = train_data['score']

# Разделение данных на обучающий и валидационный наборы
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12345)

# Обучение модели
# Инициализация и обучение модели
model = CatBoostRegressor(
    loss_function='MAE', 
    eval_metric='MAE', 
    depth=10, 
    learning_rate=0.1, 
    l2_leaf_reg=7, 
    random_state=12345, 
    verbose=False
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

# Сохранение модели на диск
model.save_model(model_save_path)
print(f'Модель сохранена как {model_save_path}')

