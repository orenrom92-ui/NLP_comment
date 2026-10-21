from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.config import DEVICE, MAX_LENGTH
from src.predict import load_model, predict_texts


model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    model, tokenizer = load_model()

    yield

    model = None
    tokenizer = None


app = FastAPI(
    lifespan=lifespan
)


class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NLP Comment API работает!"}

@app.post("/predict")
def predict(request: TextRequest):
    results = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=[request.text],
        device=DEVICE,
        max_length=MAX_LENGTH
    )

    result = results[0]

    return {
        "text": result["text"],
        "pred_class": result["pred_class"],
        "pred_label": result["pred_label"],
        "probs": result["probs"].tolist()
    }