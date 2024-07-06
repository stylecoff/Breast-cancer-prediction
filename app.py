import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Train a simple model (for demonstration purposes)
# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'history_diagnosed': np.random.randint(0, 2, 100),
    'history_biopsies': np.random.randint(0, 2, 100),
    'history_family': np.random.randint(0, 2, 100),
    'symptoms_lumps': np.random.randint(0, 2, 100),
    'symptoms_pain': np.random.randint(0, 2, 100),
    'symptoms_discharge': np.random.randint(0, 2, 100),
    'symptoms_size_change': np.random.randint(0, 2, 100),
    'symptoms_skin_change': np.random.randint(0, 2, 100),
    'screening_mammogram': np.random.randint(0, 2, 100),
    'screening_other_tests': np.random.randint(0, 2, 100),
    'diagnosis': np.random.randint(0, 2, 100)
})

# Preprocessing
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Train a simple decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit app layout
st.title('Breast Cancer Risk Assessment App')

st.header('Personal and Family Medical History')
history_diagnosed = st.selectbox('Have you ever been diagnosed with breast cancer or any other type of cancer before?', ['No', 'Yes'])
history_biopsies = st.selectbox('Have you had any previous breast biopsies or surgeries?', ['No', 'Yes'])
history_family = st.selectbox('Do you have a family history of breast cancer (e.g., mother, sister, daughter)?', ['No', 'Yes'])

st.header('Symptoms and Physical Changes')
symptoms_lumps = st.selectbox('Have you noticed any lumps or changes in your breast tissue?', ['No', 'Yes'])
symptoms_pain = st.selectbox('Have you experienced any pain or tenderness in your breasts?', ['No', 'Yes'])
symptoms_discharge = st.selectbox('Do you have any nipple discharge or changes in the appearance of your nipples?', ['No', 'Yes'])
symptoms_size_change = st.selectbox('Have you observed any changes in the size, shape, or appearance of your breasts?', ['No', 'Yes'])
symptoms_skin_change = st.selectbox('Have you noticed any skin changes on your breasts, such as dimpling or redness?', ['No', 'Yes'])

st.header('Screening and Preventive Measures')
screening_mammogram = st.selectbox('Have you had a mammogram before, and if so, when was your last one?', ['No', 'Yes'])
screening_other_tests = st.selectbox('Have you undergone any other breast cancer screening tests, such as MRI or ultrasound?', ['No', 'Yes'])

# Convert user inputs to binary values
input_data = pd.DataFrame({
    'history_diagnosed': [1 if history_diagnosed == 'Yes' else 0],
    'history_biopsies': [1 if history_biopsies == 'Yes' else 0],
    'history_family': [1 if history_family == 'Yes' else 0],
    'symptoms_lumps': [1 if symptoms_lumps == 'Yes' else 0],
    'symptoms_pain': [1 if symptoms_pain == 'Yes' else 0],
    'symptoms_discharge': [1 if symptoms_discharge == 'Yes' else 0],
    'symptoms_size_change': [1 if symptoms_size_change == 'Yes' else 0],
    'symptoms_skin_change': [1 if symptoms_skin_change == 'Yes' else 0],
    'screening_mammogram': [1 if screening_mammogram == 'Yes' else 0],
    'screening_other_tests': [1 if screening_other_tests == 'Yes' else 0]
})

# Make prediction
prediction = model.predict(input_data)[0]

# Define personalized suggestions based on user inputs
suggestions = []
if history_diagnosed == 'Yes':
    suggestions.append("Consult your healthcare provider for regular follow-ups and screenings.")
if history_biopsies == 'Yes':
    suggestions.append("Inform your doctor about any previous biopsies or surgeries during consultations.")
if history_family == 'Yes':
    suggestions.append("Discuss genetic testing and preventive measures with your doctor.")
if symptoms_lumps == 'Yes':
    suggestions.append("Schedule a clinical breast exam to evaluate the lump.")
if symptoms_pain == 'Yes':
    suggestions.append("Report any persistent pain to your healthcare provider.")
if symptoms_discharge == 'Yes':
    suggestions.append("Seek medical advice for any unusual nipple discharge.")
if symptoms_size_change == 'Yes':
    suggestions.append("Monitor changes and report them to your doctor.")
if symptoms_skin_change == 'Yes':
    suggestions.append("Get a clinical evaluation for any skin changes on your breasts.")
if screening_mammogram == 'Yes':
    suggestions.append("Maintain a regular schedule for mammograms as advised by your healthcare provider.")
if screening_other_tests == 'Yes':
    suggestions.append("Follow up with your doctor regarding any additional tests.")

# Display the suggestions
if st.button('Get Personalized Suggestions!'):
    st.write("Based on your inputs, here are some personalized suggestions:")
    for suggestion in suggestions:
        st.write("- " + suggestion)
    
    # Provide recommendations based on prediction result
    if prediction == 0:
        st.write("Based on the prediction, your risk of breast cancer is low. Keep maintaining regular check-ups and follow preventive measures.")
    else:
        st.write("Based on the prediction, your risk of breast cancer is higher. Please consult a healthcare provider for further evaluation and testing.")

# Provide a general recommendation to see a doctor
st.write("Regardless of the results, it's always a good idea to discuss any concerns or changes with your healthcare provider.")
