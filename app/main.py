from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import MarianMTModel, MarianTokenizer
from app.model_functions import traduz_sentenca
from fastapi.middleware.cors import CORSMiddleware

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "app/model"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    ml_models['modelo_traducao_sentenca'] = model
    ml_models['tokenizer'] = tokenizer
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


class Data(BaseModel):
    texto_ingles: str

@app.post("/translate")
async def translate(data: Data):
    sentenca_traduzida = traduz_sentenca(
        ml_models["modelo_traducao_sentenca"],
        ml_models['tokenizer'],
        data.texto_ingles
    )
    return {"tradução": sentenca_traduzida}
