from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

BASE_MODEL = "FacebookAI/roberta-base"
MODEL_DIR = Path("models/roberta-base")
DATA_DIR = Path("data/squad_v2")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

print("Descargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.save_pretrained(MODEL_DIR)

print("Descargando modelo base...")
model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)
model.save_pretrained(MODEL_DIR)

print("Descargando SQuAD 2.0...")
dataset = load_dataset("rajpurkar/squad_v2")
dataset.save_to_disk(str(DATA_DIR))

print("Descarga completada.")
print(dataset)