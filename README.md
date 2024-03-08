---
title: Project1
emoji: 🏢
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.31.1
app_file: app.py
pinned: false
license: apache-2.0
---
# Streamlit app for NLP tasks 💡
1. Film review classification on 'Good', 'Bad', 'Neutral' classes
2. Assessing the toxicity of a user message
3. Text generation using the GPT model using a custom prompt
   
## Team 🧑🏻‍💻
1. [Alexey Kamaev](https://github.com/AlexeyKamaev)
2. [Marina Kochetova](https://github.com/neonanet)
3. [Valeriia Dashieva](https://github.com/valeriedaash)

## Used models 🤖
1. For classification task we trained 3 models: TF-IDF vectorizer + LogReg, Word2Vec + LogReg, RuBert + LogReg 
2. For toxicity assessing task we trained rubert-tiny-toxicity
3. For Text generation task we trained rugpt3small_based_on_gpt2

## Used instruments 🧰
1. Python.
2. Pytorch.
3. Transformers
4. [Streamlit](https://nnproject1-6tcyg5tnwo2we6fg8gqbgt.streamlit.app).
