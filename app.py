
import requests
import json
import re
import random
import scapy
import string
import scipy
from string import punctuation
from nltk import tokenize
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk, simple_lesk, cosine_lesk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import yake
from keybert import KeyBERT
from summarizer import Summarizer
import nltk
import io
from IPython.display import Markdown, display
from random import shuffle
from io import StringIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import benepar
benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")
import streamlit as st
import os

import import_ipynb
import mcq
import Generatingblanks

st.set_page_config(
    page_title = "MCQ Generator"
)

uploaded_file = st.file_uploader("Upload the text file", type={"txt"})
if uploaded_file is not None:
    full_text = uploaded_file.getvalue().decode('utf-8')
    if st.button('Start'):
        st.write('Started...')
    summarized_text = Generatingblanks.bert_summ(full_text)
    kw_summ = Generatingblanks.keyword_selection(summarized_text)
    kw_full = Generatingblanks.keyword_selection(full_text)
    filtered_keys1 = []
    for keyword in kw_summ:
        if keyword.lower() in summarized_text.lower():
            filtered_keys1.append(keyword)
    filtered_keys2 = []
    for keyword in kw_full:
        if keyword.lower() in full_text.lower():
            filtered_keys2.append(keyword)
    sentences1 = Generatingblanks.SentenceMapping.tokenize_sentences(summarized_text)
    keyword_sentence_mapping1 = Generatingblanks.SentenceMapping.get_sentences_for_keyword(filtered_keys1, sentences1)
    sentences2 = Generatingblanks.SentenceMapping.tokenize_sentences(full_text)
    keyword_sentence_mapping2 = Generatingblanks.SentenceMapping.get_sentences_for_keyword(filtered_keys2, sentences2)
    kws_m =  keyword_sentence_mapping2.copy()
    kws_m.update(keyword_sentence_mapping1)

    mcq.Distractors.generate_option(kws_m)

    
