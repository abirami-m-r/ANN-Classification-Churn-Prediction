import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

##Load the pickle file - trained model
model = load_model('regression_model.h5')

##Load the scaler and encoder pickle files
with open('label_encoder_gender.pkl','rb') as f:
    gender_encoder = pickle.load(f)

with open('onehot_encoder_geo.pkl','rb') as f:
    geo_encoder = pickle.load(f)

with open('std_scaler.pkl','rb') as f:
    scaler = pickle.load(f)

##Streamlit app
st.title('Customer Salary Prediction')

#User input
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

##Apply encoders to categorical data
geo_data = geo_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_data,columns=geo_encoder.get_feature_names_out(['Geography']))

#combine onehot encoded columns with input data
input_data = pd.concat([input_data,geo_df],axis=1)
input_data

##Scaling the data
input_scaled = scaler.transform(input_data)

##Perform prediction
predict = model.predict(input_scaled)
predict = predict[0][0]

st.write(f'Customer estimated salary : {predict} ')


##To run a streamlit file
#streamlit run app.py