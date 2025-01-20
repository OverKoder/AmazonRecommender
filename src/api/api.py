from src.config.config import MAX_LENGTH, WEIGHTS_PATH, IDX2CATEGORY
from src.models.tinymodel import TinyModel

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad

#Perform parsing
class Review(BaseModel):
    also_buy: list[str]
    also_view: list[str]
    asin: str
    brand: str
    category: list[str] 
    description: list[str]
    feature: list[str]
    image: list[str]
    price: str
    title: str


api = FastAPI()

TOKENIZER = AutoTokenizer.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification")
TOKENIZER.pad_token = TOKENIZER.eos_token

model = AutoModelForSequenceClassification.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification", num_labels = len(IDX2CATEGORY), ignore_mismatched_sizes=True)
model = TinyModel.load_from_checkpoint(WEIGHTS_PATH, map_location='cpu', model = model, num_classes = len(IDX2CATEGORY))
model.eval()

@api.get('/')
async def root():
    return {"message":"Welcome to category predicter"}

# Prediction function
@api.post('/')
async def predict_category(review: Review):
	
    data = review.model_dump()
	
    data = '. '.join([data['brand']] + data['category'] + data['description'] + [data['title']])
    data = TOKENIZER(data, return_tensors="pt", max_length = MAX_LENGTH, padding = "max_length", truncation = True)

	# Make prediction
    with no_grad():
        prediction = model(data).logits.argmax().item()

    return {'Predicted category': IDX2CATEGORY[prediction]}

if __name__ == '__main__':
	uvicorn.run(api, host='0.0.0.0', port=8000)