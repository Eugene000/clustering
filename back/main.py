from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
import io
from fastapi.middleware.cors import CORSMiddleware
from train_model import train_and_evaluate
from pydantic import BaseModel
from typing import List

# Создаем экземпляр приложения
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Разрешаем запросы с вашего фронтенда
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)

# Папка для хранения файлов
PROCESSED_DIR = "./processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Эндпоинт для загрузки и обработки датасета
@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile):
    contents = await file.read()
    file_like = io.BytesIO(contents)

    df = pd.read_excel(file_like)

    if "client_id" in df.columns:
        df = df.drop(columns=["client_id"])
    df_clean = df.dropna()
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    X_train = df_encoded.astype('float32')

    # Сохранение обработанного файла
    processed_file_name = "processed_dataset.xlsx"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)
    X_train.to_excel(processed_file_path, index=False)

    return {
        "message": "Файл успешно загружен и обработан."
    }

class ExperimentParams(BaseModel):
    architecture: str  # Архитектура модели
    latentDim: int  # Размер латентного пространства
    epoch: int  # Количество эпох
    batchSize: int  # Размер батча
    learningRate: float # Скорость обучения
    activation: str  # Функция активации

class ExperimentResult(BaseModel):
    silhouette: float
    davies_bouldin: float

# Эндпоинт для запуска эксперимента
@app.post("/experiment/", response_model=ExperimentResult)
async def run_experiment(params: ExperimentParams):
    silhouette, davies_bouldin = train_and_evaluate(
        architecture=params.architecture,
        latent_dim=params.latentDim,
        epochs=params.epoch,
        batch_size=params.batchSize,
        learning_rate=params.learningRate,
        activation=params.activation
    )
    
    return ExperimentResult(silhouette=silhouette, davies_bouldin=davies_bouldin)