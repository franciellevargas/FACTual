#Python3
import streamlit as st
import numpy as np
import pandas as pd
import spacy 
import ktrain
from ktrain import text
from PIL import Image
from annotated_text import annotated_text, annotation
nlp = spacy.load('pt_core_news_sm') 


#Insert logo
left_co, cent_co,last_co = st.columns([0.2, 0.4, 0.3])
with cent_co:
  st.image('factual-logo.jpeg')
  

#Insert text input
input_txt = st.text_area("", help='Insira o texto da notícia', height=120) 
doc = nlp(input_txt) 

#load BERT fine-tuned model
predictor = ktrain.load_predictor('factual')

#use model
model = ktrain.get_predictor(predictor.model, predictor.preproc)

st.markdown("""
        <style>
        .font-label{
            font-size:14px;
            padding: 3px;
            color: red;
        }
        </style>
    """, unsafe_allow_html=True)


st.markdown("""
        <style>
        .font-label1{
            font-size:14px;
            padding: 3px;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

#Expander
with st.expander("**Score de Credibilidade:** :blue[85%]"):

  #sentence segmentation
  for sentences in doc.sents: 
    
    predictions = model.predict(str(sentences))
    if predictions == 'not_classe':
      annotated_text(annotation(str(sentences), "-factual- ", "$gray-200", font_family="Comic Sans MS"),)
      #st.write('<p class="font-label1">'+str(sentences)+'</p>', unsafe_allow_html=True)      
    if predictions == 'classe':
      #st.write('<p class="font-label">'+str(sentences)+'</p>', unsafe_allow_html=True)
      annotated_text(annotation(str(sentences), "biased", "#f0f2f6", font_family="Comic Sans MS", border="2px dashed red"),)


      #Menus
      tab1, tab2, tab3 = st.tabs(["Score", "Análise", "Gráfico"])

      with tab1:
        st.header("Score")
        st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

      with tab2:
        st.header("Análise")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

      with tab3:
        st.header("Gráfico")
        chart_data = pd.DataFrame(
          np.random.randn(20, 3),
          columns=['a', 'b', 'c'])

        st.line_chart(chart_data

#Button
col1, col2, col3 = st.columns([0.3, 0.3, 0.1])
with col1:
   pass
with col2:
   btn = st.button('Checar', help='Checar a crediblidade da notícia') 
   if btn:
    progress_text = "Checando a crediblidade da notícia. Por favor, aguarde."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
      time.sleep(0.1)
      my_bar.progress(percent_complete + 1, text=progress_text))
with col3:
   pass
