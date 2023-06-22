#Python3
import streamlit as st
import numpy as np
from PIL import Image

#Insert logo
left_co, cent_co,last_co = st.columns([0.2, 0.4, 0.29])
with cent_co:
  st.image('factual-logo-new.jpeg')
  

#Insert text input
txt = st.text_area("", height=150, value="Cole Aqui o Texto da Notícia")


col1, col2, col3 = st.columns([0.3, 0.3, 0.1])
with col1:
   pass
with col2:
   btn = st.button('Checar', help='Clica aqui para checar a crediblidade da notícia') 
   if btn:
      st.write('Checado')
with col3:
   pass