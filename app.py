import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('data.csv')

# Data preprocessing
# Check if 'Unnamed: 32' column exists and drop if it does
if 'Unnamed: 32' in data.columns:
    data.drop(['Unnamed: 32'], axis=1, inplace=True)

# Check if 'id' column exists and drop if it does
if 'id' in data.columns:
    data.drop(['id'], axis=1, inplace=True)

# Label encoding for the diagnosis column
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Split data
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Breast Cancer Prediction App")

st.write("""
### Predict Breast Cancer
Please answer the following questions to help us predict the likelihood of breast cancer.
""")

# User input features
def user_input_features():
    st.sidebar.header('User Input Features')
    
    # Personal and Family Medical History
    st.sidebar.subheader('Personal and Family Medical History')
    previous_cancer = st.sidebar.selectbox('Have you ever been diagnosed with breast cancer or any other type of cancer before?', ['No', 'Yes'])
    previous_biopsies = st.sidebar.selectbox('Have you had any previous breast biopsies or surgeries?', ['No', 'Yes'])
    family_history = st.sidebar.selectbox('Do you have a family history of breast cancer (e.g., mother, sister, daughter)?', ['No', 'Yes'])
    
    # Symptoms and Physical Changes
    st.sidebar.subheader('Symptoms and Physical Changes')
    lumps_changes = st.sidebar.selectbox('Have you noticed any lumps or changes in your breast tissue?', ['No', 'Yes'])
    pain_tenderness = st.sidebar.selectbox('Have you experienced any pain or tenderness in your breasts?', ['No', 'Yes'])
    nipple_discharge = st.sidebar.selectbox('Do you have any nipple discharge or changes in the appearance of your nipples?', ['No', 'Yes'])
    size_changes = st.sidebar.selectbox('Have you observed any changes in the size, shape, or appearance of your breasts?', ['No', 'Yes'])
    skin_changes = st.sidebar.selectbox('Have you noticed any skin changes on your breasts, such as dimpling or redness?', ['No', 'Yes'])
    
    # Screening and Preventive Measures
    st.sidebar.subheader('Screening and Preventive Measures')
    mammogram = st.sidebar.selectbox('Have you had a mammogram before, and if so, when was your last one?', ['No', 'Yes'])
    other_screening = st.sidebar.selectbox('Have you undergone any other breast cancer screening tests, such as MRI or ultrasound?', ['No', 'Yes'])
    
    # Collecting user input in a dictionary
    user_data = {
        'previous_cancer': previous_cancer,
        'previous_biopsies': previous_biopsies,
        'family_history': family_history,
        'lumps_changes': lumps_changes,
        'pain_tenderness': pain_tenderness,
        'nipple_discharge': nipple_discharge,
        'size_changes': size_changes,
        'skin_changes': skin_changes,
        'mammogram': mammogram,
        'other_screening': other_screening
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Prediction
input_encoded = input_df.apply(LabelEncoder().fit_transform)
prediction = model.predict(input_encoded)
prediction_proba = model.predict_proba(input_encoded)

st.subheader('Prediction')
breast_cancer_labels = np.array(['Benign', 'Malignant'])
st.write(breast_cancer_labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write('Model Accuracy: ', accuracy)

# Providing suggestions based on prediction
st.subheader('Personalized Suggestions')
if prediction == 0:
    st.write("Based on your responses, it is less likely that you have breast cancer. However, it is important to continue regular screenings and consultations with your healthcare provider.")
else:
    st.write("Based on your responses, it could have potential on cancerous. Otherwise, it is recommended that you consult with a healthcare provider for further evaluation and possible diagnostic tests.")
