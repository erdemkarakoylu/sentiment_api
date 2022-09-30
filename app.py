from annotated_text import annotated_text
import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline


#TODO 1: find 3 models for sst2
#TODO 2: find 3 models for sst3(?)
#TODO 3: Test models
#TODO 4: Add interpretability
#TODO 5: Add model footprint information


@st.cache(persist=True, allow_output_mutation=True, show_spinner=False)
def get_sentiment_pipeline(model_name):
    """Build sentiment analysis pipeline based on model name."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_path = f'/app/models/tokenizer/{model_name}'
    classifier_path = f'/app/models/classifier/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_path).to(device)
    sent_pipeline = pipeline(
        'sentiment-analysis', model=model, tokenizer=tokenizer, device=device)
    return sent_pipeline

def get_model_path(model_selection):
    """Map model selection to model path."""
    """model_dict ={
        'roberta-2': 'philschmid/roberta-large-sst2', 
        'distilbert': 'bhadresh-savani/distilbert-base-uncased-sentiment-sst2',
        'roberta-3': 'j-hartmann/sentiment-roberta-large-english-3-classes',
        'fnet': 'gchhablani/fnet-base-finetuned-sst2',
        'bertweet': 'finiteautomata/bertweet-base-sentiment-analysis'
        }"""
    model_dict = {
        'roberta-2': 'roberta-large-sst2', 
        'distilbert': 'distilbert-base-uncased-sentiment-sst2',
        'roberta-3': 'sentiment-roberta-large-english-3-classes',
        'fnet': 'fnet-base-finetuned-sst2',
        'bertweet': 'bertweet-base-sentiment-analysis'
        }
    return model_dict[model_selection.lower()]

def parse_prediction(prediction:str)->str :
    """Normalize three-letter labels some models put out."""
    pred_dict = dict(
        NEU='NEUTRAL', POS='POSITIVE', NEG='NEGATIVE')
    return pred_dict.get(prediction, prediction.upper())

st.header("Sentiment Classifier")
st.subheader('Choose Model')
model_string = st.radio(
        "", (

            "RoBERTa-2: Large, more accurate (2 classes)",
            "RoBERTa-3: Large model, more accurate (3 classes)", 
            "Distilbert: Moderate size, somewhat lower accuracy (2 classes)",
            "FNet: Lighter, slightly less accurate (2 classes)", 
            "Bertweet: Tweet-specific (3 classes)"
            )
        )
model_string = model_string.split(':')[0]
#st.write(f'Model Selected: {model_string}')

with st.spinner("Loading Model..."):
    model_path = get_model_path(model_string)
    sent_pipe = get_sentiment_pipeline(
        model_path)

st.markdown("")
st.markdown("")

st.write(f"Model loaded on {sent_pipe.model.device}")

with st.form("text_input_form", clear_on_submit=False):
    text_input = st.text_area("Enter Input Text:", )
    run_click = st.form_submit_button('RUN MODEL')
if run_click:
    prediction = sent_pipe(text_input)
    prediction_label = parse_prediction(prediction[0]['label'])
    st.write(prediction_label)
