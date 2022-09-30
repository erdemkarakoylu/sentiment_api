from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""This script downloads and saves all models offered as options in the app"""

def load_and_save_model(model_path):
    model_name = model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(f'models/tokenizer/{model_name}')
    model.save_pretrained(f'models/classifier/{model_name}')

models = [
    'philschmid/roberta-large-sst2', 
    'bhadresh-savani/distilbert-base-uncased-sentiment-sst2',
    'j-hartmann/sentiment-roberta-large-english-3-classes',
    'gchhablani/fnet-base-finetuned-sst2',
    'finiteautomata/bertweet-base-sentiment-analysis'
    ]
    
for model in models:
    load_and_save_model(model)
