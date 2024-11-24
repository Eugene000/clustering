import pandas as pd
import os
from keras.src.models import Model
from keras.src.layers import Input, Dense
from keras.src.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import itertools

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'dataset.xlsx')

df = pd.read_excel(file_path)

df = df.drop(columns=['client_id'])
df_clean = df.dropna()
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Преобразование данных в формат float32
X_train = df_encoded.astype('float32')

# Функция для обучения модели и оценки кластеризации
def train_and_evaluate(architecture, latent_dim, epochs=50, batch_size=32, learning_rate=0.001, activation='relu'):
    input_dim = df_encoded.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Построение кодировщика
    x = input_layer
    for units in architecture:
        x = Dense(units, activation=activation)(x)
    latent_space = Dense(latent_dim, activation='linear', name='latent_space')(x)
    
    # Построение декодировщика
    x = latent_space
    for units in reversed(architecture):
        x = Dense(units, activation=activation)(x)
    output_layer = Dense(input_dim, activation='sigmoid')(x)
    
    # Определение модели
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    # Обучение автоэнкодера
    autoencoder.fit(X_train, X_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=True,
                    verbose=0)
    
    # Извлечение кодировочной части
    encoder = Model(inputs=input_layer, outputs=latent_space)
    latent_data = encoder.predict(df_encoded.astype('float32'))
    
    # Кластеризация
    kmeans = KMeans(n_clusters=9, random_state=42)
    clusters = kmeans.fit_predict(latent_data)
    silhouette = silhouette_score(latent_data, clusters)
    davies_bouldin = davies_bouldin_score(latent_data, clusters)
    
    return silhouette, davies_bouldin

# Базовая настройка
def run_baseline():
    baseline_architecture = [64, 32, 16]
    baseline_latent_dim = 3
    baseline_epochs = 50
    baseline_batch_size = 32
    baseline_learning_rate = 0.001
    baseline_activation = 'relu'
    
    silhouette, davies_bouldin = train_and_evaluate(
        architecture=baseline_architecture,
        latent_dim=baseline_latent_dim,
        epochs=baseline_epochs,
        batch_size=baseline_batch_size,
        learning_rate=baseline_learning_rate,
        activation=baseline_activation
    )
    
    return {
        "Architecture": "Baseline",
        "Epochs": baseline_epochs,
        "Batch Size": baseline_batch_size,
        "Learning Rate": baseline_learning_rate,
        "Activation": baseline_activation,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin
    }

# Экспериментальные параметры
architectures = {
    "Extra Layer": {"architecture": [64, 32, 16, 8], "latent_dim": 3},
    "Simplified": {"architecture": [64, 16], "latent_dim": 3},
    "Larger Latent": {"architecture": [64, 32, 16], "latent_dim": 5}
}

epochs_list = [20, 50, 100]
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.005, 0.01]
activations = ['relu', 'tanh']

# Хранение результатов
results = []

# Добавление базового запуска
results.append(run_baseline())

# Перебор всех комбинаций
for arch_name, arch_params in architectures.items():
    for epochs, batch_size, learning_rate, activation in itertools.product(
        epochs_list, batch_sizes, learning_rates, activations
    ):
        silhouette, davies_bouldin = train_and_evaluate(
            architecture=arch_params["architecture"],
            latent_dim=arch_params["latent_dim"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            activation=activation
        )
        
        # Сохранение результатов в список
        results.append({
            "Architecture": arch_name,
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Learning Rate": learning_rate,
            "Activation": activation,
            "Silhouette Score": silhouette,
            "Davies-Bouldin Score": davies_bouldin
        })

# Преобразование результатов в таблицу
results_df = pd.DataFrame(results)

# Сохранение таблицы в Excel
results_df.to_excel('experiment_results.xlsx', index=False)

