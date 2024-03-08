import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import torch

def generate_text(model, tokenizer, prompt, max_length, num_generations, temperature):
    generated_texts = []

    for _ in range(num_generations):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    
    </style>
"""

DEVICE = 'cpu' 


tokenizer_path = "sberbank-ai/rugpt3small_based_on_gpt2"

model = torch.load('srcs/gpt_weights.pth', map_location=torch.device('cpu'))
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

st.write("## Text generator")
st.page_link("app.py", label="Home", icon='🏠')
st.markdown(
        """
        This streamlit-app can generate text using your prompt 
    """
)
# Ввод пользовательского prompt
prompt = st.text_area("Enter your prompt:")

# Параметры генерации
max_length = st.slider("Max length of generated text:", min_value=10, max_value=500, value=100, step=10)
num_generations = st.slider("Number of generations:", min_value=1, max_value=10, value=3, step=1)
temperature = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Generate text"):
    start_time = time.time()
    generated_texts = generate_text(model, tokenizer, prompt, max_length, num_generations, temperature)
    end_time = time.time()

    st.subheader("Сгенерированный текст:")
    for i, text in enumerate(generated_texts, start=1):
        st.write(f"Генерация {i}:\n{text}")

    generation_time = end_time - start_time
    st.write(f"\nВремя генерации: {generation_time:.2f} секунд")

st.markdown(button_style, unsafe_allow_html=True)  # Применяем стиль к кнопке
st.markdown(
    """
    <style>
        div[data-baseweb="textarea"] {
            border: 2px solid #3498db;  /* Цвет границы */
            border-radius: 5px;  /* Закругленные углы */
            background-color: #ecf0f1;  /* Цвет фона */
            padding: 10px;  /* Поля вокруг текстового поля */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# except:
#     st.write('Модель в разработке ( ﾉ ﾟｰﾟ)ﾉ( ﾉ ﾟｰﾟ)ﾉ( ﾉ ﾟｰﾟ)ﾉ( ﾉ ﾟｰﾟ)ﾉ')  
