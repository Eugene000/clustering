import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.src.models import Model
from keras.src.layers import Input, Dense

# Загрузка данных
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'dataset.xlsx')

df = pd.read_excel(file_path)

# Обработка данных
df = df.drop(columns=['client_id'])
df_clean = df.dropna()
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Разделение данных для обучения автоэнкодера
X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)

# Преобразование данных в формат float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Построение автоэнкодера
input_dim = df_encoded.shape[1]
input_layer = Input(shape=(input_dim,))

# Кодировщик
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
latent_space = Dense(3, activation='linear', name='latent_space')(encoded)

# Декодировщик
decoded = Dense(16, activation='relu')(latent_space)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

# Определение модели
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Компиляция автоэнкодера
autoencoder.compile(optimizer='adam', loss='mse')

# Обучение автоэнкодера
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=32, 
                shuffle=True, 
                validation_data=(X_test, X_test))

# Извлечение кодировочной части автоэнкодера
encoder = Model(inputs=input_layer, outputs=latent_space)

# Преобразование данных в латентное пространство
latent_data = encoder.predict(df_encoded.astype('float32'))

# Кластеризация в латентном пространстве
kmeans = KMeans(n_clusters=9, random_state=42)
clusters = kmeans.fit_predict(latent_data)

# Метрики
silhouette_kmeans = silhouette_score(latent_data, clusters)
davies_bouldin_kmeans = davies_bouldin_score(latent_data, clusters)
print(f"K-Means Silhouette Score: {silhouette_kmeans}")
print(f"K-Means Davies-Bouldin Score: {davies_bouldin_kmeans}")

# Визуализация кластеров в 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'pink']
labels = [f'Кластер {i}' for i in range(9)]

for i in range(9):
    ax.scatter(latent_data[clusters == i, 0], 
               latent_data[clusters == i, 1], 
               latent_data[clusters == i, 2], 
               color=colors[i], 
               label=labels[i], 
               s=50)

ax.set_xlabel('Latent Space Dim 1')
ax.set_ylabel('Latent Space Dim 2')
ax.set_zlabel('Latent Space Dim 3')

ax.legend(loc='best')
plt.title('3D Clustering with Autoencoder Latent Space')
plt.show()

# Сохранение кластеров
df_clean['Cluster'] = clusters
df_clean.to_excel('clustered_data_with_autoencoder.xlsx', index=False)

# Анализ кластеров
cluster_summary = df_clean.groupby('Cluster').mean()
print(cluster_summary)
