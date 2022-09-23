from annotated_text import annotated_text
import streamlit as st

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline


#TODO 1: find 3 models for sst2
#TODO 2: find 3 models for sst3(?)
#TODO 3: Test models
#TODO 4: Add interpretability
#TODO 5: Add model footprint information


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentiment_pipeline(
    model_name#, num_labels
    ):
    #config = AutoConfig.from_pretrained(
    #    model_name, num_labels=num_labels)
    
    #if num_labels ==2:
    #    config.id2label = {0:'NEGATIVE', 1: 'POSITIVE'}
    #elif num_labels == 3:
    #    config.id2label = {0:'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name#, config=config
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name#, config=config
        )
    sent_pipeline = pipeline(
        'text-classification', model=model_name, tokenizer=tokenizer)
    return sent_pipeline

def get_model_path(model_selection):
    model_dict ={
        'roberta-2': 'philschmid/roberta-large-sst2', 
        #'distilbert'='bhadresh-savani/distilbert-base-uncased-sentiment-sst2'
        'robeta-3': 'j-hartmann/sentiment-roberta-large-english-3-classes',
        'fnet': 'gchhablani/fnet-base-finetuned-sst2',
        'bertweet': 'finiteautomata/bertweet-base-sentiment-analysis'
        }
    return model_dict[model_selection.lower()]


st.header("Sentiment Classifier")
#left_col, right_col = st.columns(2)
#with left_col:
model_string = st.radio(
        "Choose Model", (
            #"BERT: Bigger, more accurate (2 classes)",
            "RoBERTa-2: Large, more accurate (2 classes)",
            "RoBERTa-3: Large model, more accurate (3 classes)", 
            "FNet: Lighter, slightly less accurate (2 classes)", 
            "Bertweet: Tweet-specific (3 classes)")
        )
model_string = model_string.split(':')[0]
st.write(f'Model Selected: {model_string}')
   
#with right_col:
#    num_labels = st.radio('Choose Number of Sentiment Classes', 
#d    (2, 3))

with st.spinner("Loading Model..."):
    model_path = get_model_path(model_string)
    sent_pipe = get_sentiment_pipeline(
        model_path)
text_input = st.text_input("Enter Input Text:")
prediction = sent_pipe(text_input)
st.write(prediction[0]['label'])