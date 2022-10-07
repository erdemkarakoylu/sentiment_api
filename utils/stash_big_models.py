from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""This script downloads and saves all models offered as options in the app"""

def load_and_save_model(model_hub_path):
    model_name = model_hub_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_hub_path)
    clf = AutoModelForSequenceClassification.from_pretrained(model_hub_path)
    tokenizer.save_pretrained(f'/app/models/tokenizer/{model_name}')
    clf.save_pretrained(f'/app/models/classifier/{model_name}')

models = [
    'philschmid/roberta-large-sst2', 
    'bhadresh-savani/distilbert-base-uncased-sentiment-sst2',
    'j-hartmann/sentiment-roberta-large-english-3-classes',
    'gchhablani/fnet-base-finetuned-sst2',
    'finiteautomata/bertweet-base-sentiment-analysis'
    ]
    
for model in models:
    load_and_save_model(model)
