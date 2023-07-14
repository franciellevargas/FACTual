import streamlit as st
import numpy as np
import pandas as pd
import spacy 
import ktrain
import re
from ktrain import text
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from annotated_text import annotated_text, annotation

#st.markdown("""
#        <style>
#        .font-label{
#            font-size:14px;
#            padding: 3px;
#            color: red;
#        }
#        </style>
#    """, unsafe_allow_html=True)


#st.markdown("""
#        <style>
#        .font-label1{
#            font-size:14px;
#            padding: 3px;
#            color: black;
#        }
#        </style>
#    """, unsafe_allow_html=True)
#st.write('<p class="font-label">'+str(sentences)+'</p>', unsafe_allow_html=True)      
        

#load and use BERT fine-tuned model
predictor = ktrain.load_predictor('factual')
model = ktrain.get_predictor(predictor.model, predictor.preproc)

#spacy language model PT
nlp = spacy.load('pt_core_news_sm')

#functions
def printProgress(progress):
  if(progress>=50):
    color1='#262730'
    color2='rgba(0,0,0,0)'
    text_progress = str(progress)+'%'
  if(progress<50):
    color2='red'
    color1='rgba(0,0,0,0)'
    text_progress = str(progress)+'%'
  df_progress = pd.DataFrame({'names' : ['progress',' '],'values' :  [progress, 100 - progress]}) 
  fig = px.pie(df_progress, values ='values', names = 'names', hole = 0.7,color_discrete_sequence = [color1, color2])
  fig.data[0].textfont.color = 'white'
  fig.update_traces(textinfo='none', marker=dict(line=dict(color='#0068c9', width=0.1)))
  fig.update_layout(showlegend=False, hovermode=False,annotations=[dict(text=('<b>'+text_progress+'</b>'), x=0.5, y=0.5, font_size=25, showarrow=False, hovertext='Score de credibilidade')],height=170, margin=dict(r=5, l=5, t=5, b=5))
  return fig

def identify_quotes(input_quotes):
  quotes = re.findall(r'"([^"]*)"', input_quotes)
  return quotes


def checked_facts(sentence_to_check):
  checked_claims = 'O Nordeste tem a maior número acidentes com vítimas do Brasil. Estudos mostram que limonada cura o COVID'
  doc = nlp(checked_claims)
  for checked_sentences in doc.sents:
    if (sentence_to_check in str(checked_sentences)) or (str(checked_sentences) in sentence_to_check):
      return True
    else:
      return False


def porcentagem(votos, total):
  return (votos / total) * 100 if total > 0 else 0
  

#Insert logo
left_co, cent_co,last_co = st.columns([0.2, 0.4, 0.3])
with cent_co:
  st.image('factual-logo.jpeg')
  
#Text input
input_txt = st.text_area("", help='Insira o texto da notícia', height=120) 

#Button
col1, col2, col3 = st.columns([0.35, 0.35, 0.1])
with col1:
   pass
with col2:
  btn = st.button('Checar', help='Checar a crediblidade da notícia')  
with col3:
   pass


#Results
if btn:
  tab1, tab2, tab3 = st.tabs(["Análise","Score", "Gráfico"])
  count_quotes = 0
  count_bias = 0
  count_fake = 0
  count_factual = 0   
  
  with tab1:
    doc = nlp(input_txt)
    for sentences in doc.sents:
      temp = identify_quotes(str(sentences))
      #check if quotes function return is empty
      if not temp:
        #classification of factuality
        predictions = model.predict(str(sentences))
        
        if predictions == 'classe':   
          annotated_text(annotation(str(sentences), "BIASED", "#f0f2f6", font_family="Comic Sans MS", border="2px dashed red"),)
          count_bias +=1
          
        if predictions == 'not_classe':
          temp2 = checked_facts(str(sentences))
          if temp2 == False:
            annotated_text(annotation(str(sentences), "FACTUAL ", "#f0f2f6", font_family="Comic Sans MS"),)
            count_factual +=1
          else:
            annotated_text(annotation(str(sentences), "FAKE ", "#faa", font_family="Comic Sans MS"),)   
            count_fake +=1
      else:        
        annotated_text(annotation(str(sentences), "QUOTES ", "#f0f2f6", font_family="Comic Sans MS"),)
        count_quotes +=1
      temp = ''
      temp2 = ''
      temp3 = 0
      high_credibility = 0
      tot_sente = 0

  with tab2:
    st.subheader("Score de Credibilidade")
    high_credibility = count_quotes + count_factual
    tot_sente = count_bias + count_fake + count_quotes + count_factual
    temp3 = porcentagem(high_credibility,tot_sente)
    graph = printProgress(temp3)
    st.plotly_chart(graph, use_container_width=True)

  with tab3:
    #st.header("Gráfico")
    labels = 'Factual', 'Biased', 'Fake'
    sizes = [high_credibility, count_bias, count_fake]
    explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.set_figwidth(3) 
    fig1.set_figheight(1) 
    st.pyplot(fig1)
    
    #chart_data = pd.DataFrame(
    #  np.random.randn(4, 4),
    #  columns=['Factual', 'Quotes', 'Biased', 'Fake'])
      
    #st.bar_chart(chart_data)

  
