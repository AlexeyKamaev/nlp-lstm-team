import json 
import numpy as np
import joblib

from transformers import AutoTokenizer, AutoModel
import torch
import pickle

import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt

with open('srcs/vocab_to_int.json', encoding='utf-8') as f:
    vocab_to_int = json.load(f)

with open('srcs/int_to_vocab.json', encoding='utf-8') as f:
    int_to_vocab = json.load(f)

VOCAB_SIZE = len(vocab_to_int) + 1
EMBEDDING_DIM = 64 # embedding_dim
SEQ_LEN = 350
HIDDEN_SIZE = 64

with open('srcs/embedding_matrix.npy', 'rb') as f:
    embedding_matrix = np.load(f)



from srcs.srcs import LSTMConcatAttentionB
from srcs.srcs import Text_ex, clean


log_reg_vec = joblib.load('srcs/log_reg_vec.sav')

log_reg_bert = joblib.load('srcs/log_reg_bert.sav')

texter = Text_ex(clean, vocab_to_int, SEQ_LEN)
lstm = LSTMConcatAttentionB()

lstm.load_state_dict(torch.load('srcs/lstm.pt'))


vectorizer = pickle.load(open("srcs/vectorizer.pickle", "rb"))
tokenizer = AutoTokenizer.from_pretrained('srcs/tokenzier', local_files_only=True)
bert = torch.load('srcs/bert.pt')



from srcs.srcs import PredMaker
predM = PredMaker(model1=log_reg_vec, model2=lstm, rubert=bert, model3=log_reg_bert, vectorizer=vectorizer, texter=texter, clean_func=clean, tokenizer=tokenizer, itc=int_to_vocab)




import streamlit as st
st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)


st.title('Кинопоиск')
st.page_link("app.py", label="Home", icon='🏠')

import streamlit as st

txt = st.text_area(
    "Введите сюда отзыв на фильм:",
    "",
    )

# st.write(f'Введено {len(txt)} символов.')

if txt == '' or len(txt) < 12:
    if len(txt) >= 1:
        st.write('Введи что-нибудь нормальное')
else:
    text = txt
    res1, res2, res3, t, att, *times = predM(text)
    t_ = t[0].numpy()[0]
    k = len(t[1].split()) + 1

    labels = [int_to_vocab[str(x)] for x in t_ if int_to_vocab.get(str(x))]

    if list(set(labels[-k:])) == ["<pad>"]:
        st.write('Давай по новой миша, всё @**##')
        st.write(set(labels[-k:]))
    else:
        st.toast('!', icon='🎉')
        di = {0:'Плохо',1:'Нейтрально',2:'Хорошо'}
        d = {0: st.error, 1: st.warning, 2: st.success}

        d[res1](f'Предсказание 1-й модели: {di[res1]}')
        st.write(f'время = {round(times[0],3)}c, f1-score = 0.64')

        d[res2](f'Предсказание 2-й модели: {di[res2]}')
        st.write(f'время = {round(times[0],3)}c, f1-score = 0.70')

        d[res3](f'Предсказание 3-й модели: {di[res3]}')
        st.write(f'время = {round(times[0],3)}c, f1-score = 0.66')


        plt.figure(figsize=(8, 8))
        plt.barh(np.arange(len(t_))[-k:], att[-k:])
        plt.yticks(ticks = np.arange(len(t_))[-k:], labels = labels[-k:])
        plt.title(f'f1-score = 0.7\npred = {di[res2]}\ntime = {round(times[1],3)}c');
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()