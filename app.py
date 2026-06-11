from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.responses import FileResponse

app=FastAPI(title="duplicate question detection")

tokenizer = AutoTokenizer.from_pretrained(
    "anya2407/bert-duplicate-detector"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "anya2407/bert-duplicate-detector"
)

model.eval()

class QuestionPairs(BaseModel):
    ques1:str
    ques2:str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/predict")
def predict(pair:QuestionPairs):
    enc = tokenizer(
        pair.ques1, pair.ques2,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    
    return {
        "duplicate": bool(probs[1] > 0.5),
        "confidence": round(float(probs[1]), 4)
    }


