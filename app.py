from annotated_text import annotated_text
import streamlit as st
from streamlit import components

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
from transformers_interpret import SequenceClassificationExplainer as SCE



#TODO 1: find 3 models for sst2
#TODO 2: find 3 models for sst3(?)
#TODO 3: Test models
#TODO 4: Add interpretability
#TODO 5: Add model footprint information

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False)
def get_sentiment_pipeline(model_name):
    """Build sentiment analysis pipeline based on model name."""
    
    tokenizer_path = f'/app/models/tokenizer/{model_name}'
    classifier_path = f'/app/models/classifier/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_path).to(DEVICE)
    device_map = torch.cuda.current_device() if DEVICE.type=='cuda' else 'cpu'
    sent_pipeline = pipeline(
        'sentiment-analysis', model=model, tokenizer=tokenizer, device=device)
    return sent_pipeline

def get_model_path(model_selection):
    """Map model selection to model path."""

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
st.sidebar.subheader('Choose Model')
model_string = st.sidebar.radio(
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
    sent_pipe = get_sentiment_pipeline(model_path)

st.markdown("")
st.markdown("")
    st.warning("Running on CPU", icon="⚠️")

with st.form("text_input_form", clear_on_submit=False):
    text_input = st.text_area("Enter Input Text:", )
    run_click = st.form_submit_button('CLASSIFY INPUT')
if run_click:
    prediction = sent_pipe(text_input)
    prediction_label = parse_prediction(prediction[0]['label'])
    st.write(prediction_label)

st.markdown("---")
st.markdown("---")
# ---- Prediction Interpreter ----- #
st.subheader("Prediction Interpretation")
cls_explainer = SCE(model=sent_pipe.model, tokenizer=sent_pipe.tokenizer)
if cls_explainer.accepts_position_ids:
    emb_type_name = st.sidebar.selectbox(
        "Choose embedding type for attribution.", ["word", "position"]
    )
    if emb_type_name == "word":
        emb_type_num = 0
    if emb_type_name == "position":
        emb_type_num = 1
else:
    emb_type_num = 0
explanation_classes = ["predicted"] + list(
    sent_pipe.model.config.label2id.keys())
explanation_class_choice = st.sidebar.selectbox(
    "Explanation class: The class you would like to explain output with respect to.",
    explanation_classes
    )
if st.button("INTERPRET PREDICTION"):
    with st.spinner("Interpreting your text (This may take some time)"):
        if explanation_class_choice != "predicted":
            word_attributions = cls_explainer(
                text_input,
                class_name=explanation_class_choice,
                embedding_type=emb_type_num,
                internal_batch_size=2,
            )
        else:
            word_attributions = cls_explainer(
                text_input, embedding_type=emb_type_num, internal_batch_size=2
                )

    if word_attributions:
        word_attributions_expander = st.expander(
            "Click here for raw word attributions"
        )
        with word_attributions_expander:
            st.json(word_attributions)
        components.v1.html(
            cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
        )
