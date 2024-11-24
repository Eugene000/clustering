from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os

# Создаем экземпляр приложения
app = FastAPI()

# Папка для хранения файлов
PROCESSED_DIR = "./processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Эндпоинт для загрузки и обработки датасета
@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile):
    # Проверка типа файла
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")

    try:
        # Чтение содержимого файла в Pandas DataFrame
        df = pd.read_excel(file.file)

        # Удаление client_id
        if "client_id" in df.columns:
            df = df.drop(columns=["client_id"])

        # Удаление пропусков
        df_clean = df.dropna()

        # Кодирование категориальных переменных
        df_encoded = pd.get_dummies(df_clean, drop_first=True)

        # Сохранение обработанного файла
        processed_file_name = f"processed_{file.filename}"
        processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)
        df_encoded.to_excel(processed_file_path, index=False)

        return {
            "message": "File uploaded and processed successfully.",
            "processed_file_path": processed_file_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
