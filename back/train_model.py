from keras.src.models import Model
from keras.src.layers import Input, Dense
from keras.src.optimizers import Adam
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

def train_and_evaluate(architecture, latent_dim, epochs=50, batch_size=32, learning_rate=0.001, activation='relu'):
    split_architecture = list(map(int, architecture.split(',')))
    base_dir = os.path.dirname(__file__)
    processed_file = os.path.join(base_dir, 'processed', 'processed_dataset.xlsx')
    X_train = pd.read_excel(processed_file)
    
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Построение кодировщика
    x = input_layer
    for units in split_architecture:
        x = Dense(units, activation=activation)(x)
    latent_space = Dense(latent_dim, activation='linear', name='latent_space')(x)
    
    # Построение декодировщика
    x = latent_space
    for units in reversed(split_architecture):
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
    latent_data = encoder.predict(X_train)
    
    # Кластеризация
    kmeans = KMeans(n_clusters=9, random_state=42)
    clusters = kmeans.fit_predict(latent_data)
    silhouette = silhouette_score(latent_data, clusters)
    davies_bouldin = davies_bouldin_score(latent_data, clusters)
    
    return silhouette, davies_bouldin