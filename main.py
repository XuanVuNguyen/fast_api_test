from fastapi import FastAPI
import uvicorn
from predict import Predictor

predictor = Predictor("lightning_logs/version_0/checkpoints/epoch=19-step=1680.ckpt", "vocab.pth")

app = FastAPI()
@app.post("/predict")
async def predict(text: str):
    label, prob = predictor.predict(text)
    label_dict = {
        "b": "business",
        "e": "entertainment",
        "m": "medical",
        "t": "science and technology"
    }
    return {"label": label_dict[label], "probability": prob}