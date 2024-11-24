# Автоматическое выявление кластеров аудитории сервиса доставки и формирование бизнес-описаний (портретов) к ним

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'dataset.xlsx')

df = pd.read_excel(file_path)
# print(df.head())
# print(df.columns)
# print(df.shape)

df = df.drop(columns=['client_id'])

df_clean = df.dropna()

df_encoded = pd.get_dummies(df_clean, drop_first=True)
# print(df_encoded)
# print(df_encoded.columns)

wcss = []
for i in range(2, 30):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_encoded)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 30), wcss)
plt.title('Метод Локтя')
plt.xlabel('Кол-во кластеров')
plt.ylabel('WCSS')
plt.show()

silhouette_scores = []
davies_bouldin_scores = []
for k in range(2, 30):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters_kmeans = kmeans.fit_predict(df_encoded)
    
    silhouette_kmeans = silhouette_score(df_encoded, clusters_kmeans)
    davies_bouldin_kmeans = davies_bouldin_score(df_encoded, clusters_kmeans)
    print(f"K-Means Silhouette Score (K={k}): {silhouette_kmeans}")
    print(f"K-Means Davies-Bouldin Score (K={k}): {davies_bouldin_kmeans}")
    
    silhouette_scores.append(silhouette_kmeans)
    davies_bouldin_scores.append(davies_bouldin_kmeans)

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(range(2, 30), silhouette_scores, marker='o')
plt.title('График по метрике Silhouette Score')
plt.xlabel('Кол-во кластеров')
plt.ylabel('Значение Silhouette Score')
plt.xticks(range(2, 30))

plt.subplot(1, 2, 2)
plt.plot(range(2, 30), davies_bouldin_scores, marker='o', color='orange')
plt.title('График по метрике Davies-Bouldin Score')
plt.xlabel('Кол-во кластеров')
plt.ylabel('Значение Davies-Bouldin Score')
plt.xticks(range(2, 30))

plt.tight_layout()
plt.show()

pca = PCA(n_components=3)
df_pca_3d = pca.fit_transform(df_encoded)

kmeans = KMeans(n_clusters=9, random_state=42)
clusters = kmeans.fit_predict(df_pca_3d)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'pink'] 
labels = [f'Кластер {i}' for i in range(9)]

for i in range(9):
    ax.scatter(df_pca_3d[clusters == i, 0], df_pca_3d[clusters == i, 1], df_pca_3d[clusters == i, 2], 
               color=colors[i], label=labels[i], s=50)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

ax.legend(loc='best')

plt.title('3D PCA Plot K-Means')
plt.show()
