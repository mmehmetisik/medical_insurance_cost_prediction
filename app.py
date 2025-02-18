import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Sayfa konfigürasyonu
st.set_page_config(
   page_title="Medical Insurance Cost Prediction",
   page_icon="💉",
   layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
   padding: 0rem 0rem;
}
.title {
   text-align: center;
}
.header-style {
   font-size: 1.5rem;
   font-weight: bold;
   margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Model ve diğer dosyaları yükleme
@st.cache_resource()
def load_model_and_components():
   try:
       model_path = os.path.join('models', 'final_model.pkl')
       scaler_path = os.path.join('models', 'scaler.pkl')
       features_path = os.path.join('models', 'feature_columns.pkl')

       with open(model_path, 'rb') as file:
           model = pickle.load(file)

       with open(scaler_path, 'rb') as file:
           scaler = pickle.load(file)

       with open(features_path, 'rb') as file:
           features = pickle.load(file)

       return model, scaler, features
   except Exception as e:
       st.error(f"Error loading model components: {str(e)}")
       return None, None, None

# Model ve bileşenleri yükle
model, scaler, features = load_model_and_components()

# Model yüklenmediyse uygulama durdur
if None in (model, scaler, features):
   st.error("Failed to load model components. Please check the model files.")
   st.stop()

# Ana başlık
st.title('💉 Medical Insurance Cost Prediction')

# Sidebar bilgileri
st.sidebar.header('About the Application')
st.sidebar.markdown("""
This application predicts medical insurance costs based on various factors:

**Features:**
* Age, BMI, and number of children
* Smoking status and gender
* Region and other health factors
* Instant cost estimation
* Detailed factor analysis

**Data Source:**
* Medical Insurance Dataset
* 1,338 real insurance records
* Up-to-date analysis
""")

# Input alanlarını iki kolona bölelim
col1, col2 = st.columns(2)

with col1:
   st.markdown("### Personal Information")
   
   # Yaş seçimi
   age = st.slider("Age", min_value=18, max_value=64, value=30)
   
   # Cinsiyet seçimi
   sex = st.radio("Gender", options=['female', 'male'])
   
   # BMI girişi
   bmi = st.number_input("BMI (Body Mass Index)", 
                        min_value=15.0, 
                        max_value=54.0, 
                        value=25.0, 
                        step=0.1,
                        help="Normal BMI range is 18.5 to 24.9")
   
   # Çocuk sayısı
   children = st.slider("Number of Children", 
                       min_value=0, 
                       max_value=5, 
                       value=0)

with col2:
   st.markdown("### Additional Factors")
   
   # Sigara kullanımı
   smoker = st.radio("Smoking Status", 
                     options=['no', 'yes'],
                     help="Do you smoke?")
   
   # Bölge seçimi
   region = st.selectbox("Region", 
                        options=['southwest', 'southeast', 'northwest', 'northeast'],
                        help="Select your residential area")

# Tahmin butonu
st.markdown("### ")  # Boşluk eklemek için
predict_button = st.button("Predict Insurance Cost", 
                         help="Click to predict the insurance cost")

# Tahmin fonksiyonu
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
   # Risk skoru hesaplama
   if smoker == 'yes' and bmi > 30:
       risk_score = 3
   elif smoker == 'yes' and bmi > 25:
       risk_score = 2
   elif smoker == 'yes' or bmi > 30:
       risk_score = 1
   else:
       risk_score = 0
   
   # Aile büyüklüğü
   family_size = children + 1
   
   # Etkileşim değişkenleri
   age_bmi_interaction = (age * bmi) / 100
   smoker_bmi_interaction = bmi if smoker == 'yes' else 0
   smoker_age_interaction = age if smoker == 'yes' else 0
   health_score = (bmi * age * (3 if smoker == 'yes' else 1)) / 100
   
   # Tahmin için DataFrame oluşturma
   data = {
       'age': [age],
       'bmi': [bmi],
       'children': [children],
       'family_size': [family_size],
       'age_bmi_interaction': [age_bmi_interaction],
       'smoker_bmi_interaction': [smoker_bmi_interaction],
       'smoker_age_interaction': [smoker_age_interaction],
       'health_score': [health_score],
       'sex_male': [1 if sex == 'male' else 0],
       'smoker_yes': [1 if smoker == 'yes' else 0],
       'region_northwest': [1 if region == 'northwest' else 0],
       'region_southeast': [1 if region == 'southeast' else 0],
       'region_southwest': [1 if region == 'southwest' else 0],
       'risk_score_1': [1 if risk_score == 1 else 0],
       'risk_score_2': [1 if risk_score == 2 else 0],
       'risk_score_3': [1 if risk_score == 3 else 0],
       'has_children_1': [1 if children > 0 else 0]
   }
   
   df = pd.DataFrame(data)
   
   # Numerik kolonları scale etme
   num_cols = ['age', 'bmi', 'children', 'family_size', 'age_bmi_interaction', 
               'smoker_bmi_interaction', 'smoker_age_interaction', 'health_score']
   df[num_cols] = scaler.transform(df[num_cols])
   
   # Tahmin
   prediction = model.predict(df)[0]
   
   return prediction, risk_score

# Tahmin butonu tıklandığında
if predict_button:
   prediction, risk_score = predict_insurance_cost(
       age, sex, bmi, children, smoker, region
   )
   
   # Sonuçları göster
   st.markdown("### Results")
   
   # Tahmin edilen maliyet
   st.markdown(
       f"""
       <div style='padding:10px; border-radius:10px; background-color: rgba(0,100,0,0.1)'>
       <h3 style='color: #0f5132; margin:0'>Predicted Insurance Cost: ${prediction:,.2f}</h3>
       </div>
       """, 
       unsafe_allow_html=True
   )
   
   # Faktör analizi
   st.markdown("### Factors Affecting the Price")
   col1, col2 = st.columns(2)
   
   with col1:
       st.markdown(f"""
       * Age: {age} years
       * BMI: {bmi:.1f}
       * Number of Children: {children}
       """)
       
   with col2:
       st.markdown(f"""
       * Smoking Status: {smoker}
       * Region: {region}
       * Risk Score: {risk_score}
       """)
