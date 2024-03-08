import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

model_path = 'srcs/model_modify.pth'

#  токенизатор
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-toxicity')

model = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-toxicity', num_labels=1, ignore_mismatched_sizes=True)
# весов модифицированной модели
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
image = Image.open("media/oritoxic.jpg")

df = pd.read_csv("media/Toxic_labeled.csv")
loss_values = [0.4063596375772262, 0.402279906166038, 0.3998144585561736, 0.39567733055365567,
               0.3921396666608141, 0.38956182373070186, 0.3866641920902114, 0.3879134839351564,
               0.38288725781591604, 0.38198364493999004]

#Боковая панель
selected_option = st.sidebar.selectbox("Выберите из списка", ["Определение токсичность текста", "Информация о датасете", "Информация о модели"])

#st.title("Главная страница")


if selected_option == "Определение токсичность текста":


    st.markdown("<h1 style='text-align: center;'>Приложение для определения токсичности текста</h1>",
                unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    user_input = st.text_area("")


    # Функция предсказания токсичности

    def predict_toxicity(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probability = torch.sigmoid(logits).item()
        prediction = "токсичный" if probability >= 0.5 else "не токсичный"
        return prediction, probability
    # Тык на кнопу
    if st.button("Оценить токсичность"):
        if user_input:
            prediction, toxicity_probability = predict_toxicity(user_input)
            st.write(f'Вероятность токсичности: {toxicity_probability:.4f}')

    # Прогресс бар
    if 'toxicity_probability' in locals():
        progress_percentage = int(toxicity_probability * 100)
        progress_bar_color = f'linear-gradient(to right, rgba(0, 0, 255, 0.5) {progress_percentage}%, rgba(255, 0, 0, 0.5) {progress_percentage}%)'
        st.markdown(f'<div style="background: {progress_bar_color}; height: 20px; border-radius: 5px;"></div>',
                    unsafe_allow_html=True)

elif selected_option == "Информация о датасете":
    st.header("Информация о датасете:")
    st.dataframe(df.head())
    st.write(f"Объем выборки: 14412")
    st.subheader("Баланс классов в датасете:")
    st.write(f"Количество записей в классе 0.0: {len(df[df['toxic'] == 0.0])}")
    st.write(f"Количество записей в классе 1.0: {len(df[df['toxic'] == 1.0])}")
    fig, ax = plt.subplots()
    df['toxic'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
    ax.set_xticklabels(['Не токсичный', 'Токсичный'], rotation=0)
    ax.set_xlabel('Класс')
    ax.set_ylabel('Количество записей')
    ax.set_title('Распределение по классам')
    st.pyplot(fig)

elif selected_option == "Информация о модели":
    st.subheader("Информация о модели:")
    st.write(f"Модель: Rubert tiny toxicity")
    st.subheader("Информация о процессе обучения")

# график лосса
#st.subheader("График потерь в процессе обучения")
#st.line_chart([0.5181976270121774, 0.4342067330899996, 0.41386983832460666])  # Замените данными из ваших эпох
    for epoch, loss in enumerate(loss_values, start=1):
        st.write(f"<b>Epoch {epoch}/{len(loss_values)}, Loss:</b> {loss}<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <b>Количество эпох:</b> 10
        <b>Размер батча:</b> 8
        <b>Оптимизатор:</b> Adam
        <b>Функция потерь:</b> BCEWithLogitsLoss
        <b>learning rate:</b> 0.00001
        """,
        unsafe_allow_html=True
    )

    st.subheader("Метрики модели:")
    st.write(f"Accuracy: {0.8366:.4f}")
    st.write(f"Precision: {0.8034:.4f}")
    st.write(f"Recall: {0.6777:.4f}")
    st.write(f"F1 Score: {0.7352:.4f}")


    st.subheader("Код")


    bert_model_code = """
    
    model = BertModel(
        embeddings=BertEmbeddings(
            word_embeddings=Embedding(29564, 312, padding_idx=0),
            position_embeddings=Embedding(512, 312),
            token_type_embeddings=Embedding(2, 312),
            LayerNorm=LayerNorm((312,), eps=1e-12, elementwise_affine=True),
            dropout=Dropout(p=0.1, inplace=False),
        ),
        encoder=BertEncoder(
            layer=ModuleList(
                BertLayer(
                    attention=BertAttention(
                        self=BertSelfAttention(
                            query=Linear(in_features=312, out_features=312, bias=True),
                            key=Linear(in_features=312, out_features=312, bias=True),
                            value=Linear(in_features=312, out_features=312, bias=True),
                            dropout=Dropout(p=0.1, inplace=False),
                        ),
                        output=BertSelfOutput(
                            dense=Linear(in_features=312, out_features=312, bias=True),
                            LayerNorm=LayerNorm((312,), eps=1e-12, elementwise_affine=True),
                            dropout=Dropout(p=0.1, inplace=False),
                        ),
                    ),
                    intermediate=BertIntermediate(
                        dense=Linear(in_features=312, out_features=600, bias=True),
                        intermediate_act_fn=GELUActivation(),
                    ),
                    output=BertOutput(
                        dense=Linear(in_features=600, out_features=312, bias=True),
                        LayerNorm=LayerNorm((312,), eps=1e-12, elementwise_affine=True),
                        dropout=Dropout(p=0.1, inplace=False),
                    ),
                )
            )
        ),
        pooler=BertPooler(
            dense=Linear(in_features=312, out_features=312, bias=True),
            activation=Tanh(),
        ),
        dropout=Dropout(p=0.1, inplace=False),
        classifier=Linear(in_features=312, out_features=1, bias=True),
    )
    """

    # Отображение кода в Streamlit
    st.code(bert_model_code, language="python")