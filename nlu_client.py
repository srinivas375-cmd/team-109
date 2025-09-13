# backend/nlu_client.py
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
from transformers import pipeline
import os

# Configure IBM Watson NLU (set env vars or replace here)
WATSON_APIKEY = os.getenv("WATSON_APIKEY")
WATSON_URL = os.getenv("WATSON_URL")

nlu = None
if WATSON_APIKEY and WATSON_URL:
    nlu = NaturalLanguageUnderstandingV1(
        version='2023-08-01',
        iam_apikey=WATSON_APIKEY,
        url=WATSON_URL
    )

# Hugging Face pipelines (fallback)
ner_pipe = pipeline("ner", grouped_entities=True)
# summarizer = pipeline("summarization")  # or text2text-generation with a simplifier model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#classifier = pipeline("text-classification", return_all_scores=True)
from transformers import pipeline

# For clause simplification → summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# For entity extraction → NER
ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


def watson_entities(text):
    if not nlu:
        return {"warning": "Watson not configured; using HF NER", "entities": ner_pipe(text)}
    resp = nlu.analyze(text=text, features=Features(entities=EntitiesOptions(limit=50))).get_result()
    return resp.get("entities", [])

def hf_simplify(text):
    # Use summarization as a quick simplifier; for better results use a fine-tuned model like "t5-small" style simplifier
    out = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return out[0]["summary_text"]

def hf_classify(text):
    # Example: return best label, score
    out = classifier(text, top_k=3)
    # pick highest score label
    best = sorted(out, key=lambda x: x[0].get("score", 0), reverse=True)[0]
    # pipeline return format may differ by transformers version; adjust accordingly in real code
    # fallback simple:
    if isinstance(out, list) and len(out)>0 and isinstance(out[0], dict):
        label = out[0]["label"]
        score = out[0]["score"]
    else:
        label = "UNKNOWN"
        score = 0.0
    return label, float(score)
