import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, load_model
from sklearn.decomposition import PCA

base_dir = os.path.dirname(__file__)
print(base_dir)

file_path = os.path.join(base_dir, 'dataset.xlsx')
print(file_path)

# Загрузка данных
df = pd.read_excel(file_path)
print(df)

# Шаг 1: Предобработка данных
# Пример: кодирование категориальных признаков и нормализация
df_encoded = pd.get_dummies(df, drop_first=True)

# Масштабирование (нормализация данных)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Шаг 2: Кластеризация с помощью K-Means
k = 5  # Количество кластеров, можно подобрать
kmeans = KMeans(n_clusters=k, random_state=42)
clusters_kmeans = kmeans.fit_predict(df_scaled)

# Метрики для K-Means
silhouette_kmeans = silhouette_score(df_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(df_scaled, clusters_kmeans)

print(f"K-Means Silhouette Score: {silhouette_kmeans}")
print(f"K-Means Davies-Bouldin Score: {davies_bouldin_kmeans}")

# Шаг 3: Кластеризация с использованием DEC

# Определение автоэнкодера для извлечения нелинейного латентного представления
input_dim = df_scaled.shape[1]
latent_dim = 10  # Размер латентного пространства

input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation="relu")(input_layer)
encoder = Dense(64, activation="relu")(encoder)
latent = Dense(latent_dim, activation="relu")(encoder)

decoder = Dense(64, activation="relu")(latent)
decoder = Dense(128, activation="relu")(decoder)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Обучение автоэнкодера
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=32, validation_split=0.2)

# Извлечение латентного представления
encoder_model = Model(inputs=input_layer, outputs=latent)
df_latent = encoder_model.predict(df_scaled)

# Применение K-Means на латентных представлениях
kmeans_dec = KMeans(n_clusters=k, random_state=42)
clusters_dec = kmeans_dec.fit_predict(df_latent)

# Метрики для DEC
silhouette_dec = silhouette_score(df_latent, clusters_dec)
davies_bouldin_dec = davies_bouldin_score(df_latent, clusters_dec)

print(f"DEC Silhouette Score: {silhouette_dec}")
print(f"DEC Davies-Bouldin Score: {davies_bouldin_dec}")

# Сравнение результатов кластеризации
print(f"\nСравнение метрик кластеризации:")
print(f"K-Means Silhouette: {silhouette_kmeans}, DEC Silhouette: {silhouette_dec}")
print(f"K-Means Davies-Bouldin: {davies_bouldin_kmeans}, DEC Davies-Bouldin: {davies_bouldin_dec}")
