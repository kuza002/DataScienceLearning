import pandas as pd
from pydantic import BaseModel
import dill
from fastapi import FastAPI
import pickle


class Forma(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int


class Prediction(BaseModel):
    id: int
    pred: str
    price: int


with open('model/data/loan_pipe.pkl', 'rb') as f:
    model_pkl = dill.load(f)
    model = model_pkl['model']
    model_metadata = model_pkl['model_metadate']


app = FastAPI()

@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model_metadata


@app.post('/predict', response_model=Prediction)
def predict(form:Forma):
    df = pd.DataFrame.from_dict([form.dict()])
    predict = model.predict(df)
    return {'id': df['id'],
            'pred': predict[0],
            'price': df['price']}

