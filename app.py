import streamlit as st


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('📝 &  ⚡💨🍃🪫💡')
st.title('AI-Apps for NLP tasks')


st.write('Choose app below')



st.page_link("pages/task1.py", label="Film review sentiment", icon='🎥')
st.page_link("pages/toxicapp.py", label="Message toxicity", icon='☠️')
st.page_link("pages/task3.py", label="Text generation", icon='📲')

st.subheader(f'made by: Alexey Kamaev & Marina Kochetova & Valerie Dashieva')
