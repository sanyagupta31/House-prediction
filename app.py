import streamlit as st
import pickle
import numpy as np
model= pickle.load(open('model.pkl', 'rb'))
st.title("House Price Prediction")
area=st.number_input('enter area in sq ft',min_value=500)
bedrooms=st.number_input('enter number of bedrooms',min_value=1)
bathrooms=st.number_input('enter number of bathrooms',min_value=1)
if st.button('Predict'):
    features=np.array([[area,bedrooms,bathrooms]])
    prediction=model.predict(features)
    st.success(f'The predicted house price is ${prediction[0]:,.2f}')
