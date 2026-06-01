import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#Loading the model
model = tf.keras.models.load_model('model.h5')

#Loading encoder and scaler
with open('Label_encoder_gender.pkl', 'rb') as file:  #loading Geoencoder
    Label_encoder_gender = pickle.load(file)

with open('ohe_geo.pkl', 'rb') as file:  #loading Geography one hot encoder
    ohe_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file: #loding scaler
    scaler  = pickle.load(file)

#Streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', Label_encoder_gender.classes_)
age = st.slider('Age', 18, 90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#Prepare the input

input_data = pd.DataFrame({

    'CreditScore': [credit_score],
    'Gender': [Label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})

geo_encoded = ohe_geo.transform([[geography]]).toarray() #Encode geography data
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out()) #Turn it in a data frame

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1) #Replace transformed geography data

input_scaled = scaler.transform(input_data) #scale the input

prediction = model.predict(input_scaled) #Predict using model
prediction_prob = prediction[0][0] #First index of prediction is turned into probability

st.write(f'Churn Probability : {prediction_prob:.2f}')

if prediction_prob > 0.5:
    print('Customer is likely to churn')
else:
    print('Customer is not likely to churn')