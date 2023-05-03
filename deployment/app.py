import streamlit as st
import numpy as np
import joblib

#from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'../models/RTA_model.joblib')

st.set_page_config(
    page_title="Road Traffic Severity Clasification",
    page_icon="🚨",
    layout="wide"
)
st.header('🚨Road Traffic Severity Classification🚨')
st.markdown("""This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.\n
 The target feature is **Accident_severity** which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. The metric for evaluation will be f1-score""")
st.sidebar.write('<hr>', unsafe_allow_html=True)
st.sidebar.markdown("""Author : Horacio Morales González \n
Date : April  2023\n
https://github.com/hmorales21/Road-Traffic-Severity-Classification\n
https://www.linkedin.com/in/hmorales1970/
""")