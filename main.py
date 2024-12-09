from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Загрузка вашей обученной нейросети
model = tf.keras.models.load_model('F:\Downloads\projects\AIAPtiktaktoe\tiktaktoeAIAPI')  # Укажите путь к вашей модели

# Инициализация FastAPI
app = FastAPI()

# Модель для данных доски
class BoardState(BaseModel):
    board: list[int]  # Массив из 9 элементов: 0, 1, 2 (0 — пусто, 1 — игрок, 2 — бот)

@app.post("/predict")
def predict_move(state: BoardState):
    # Преобразуем доску в формат для модели
    board = np.array(state.board).reshape(1, -1)

    # Получаем предсказание модели
    predictions = model.predict(board)
    best_move = int(np.argmax(predictions))  # Индекс лучшего хода

    # Возвращаем ответ в виде JSON
    return {"move": best_move}
