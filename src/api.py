import joblib
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/logreg.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: Message):
    X = vectorizer.transform([msg.text])

    start = time.time()
    pred = model.predict(X)[0]
    latency = time.time() - start

    return {
        "prediction": int(pred),
        "latency_ms": latency * 1000
    }