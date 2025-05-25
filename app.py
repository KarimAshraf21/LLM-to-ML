from fastapi import FastAPI
from pydantic import BaseModel
from llm import predict_churn

app = FastAPI()

class ChurnRequest(BaseModel):
    input_text: str

@app.post("/predict")
def predict(request: ChurnRequest):
    return {"result": predict_churn(request.input_text)}