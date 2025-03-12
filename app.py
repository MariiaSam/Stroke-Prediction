import streamlit as st
import joblib
import numpy as np
import pandas as pd


logistic_model = joblib.load('model/LogisticRegression.pkl') 
encoder = joblib.load('model/encoder.pkl')
scaler = joblib.load('model/scaler.pkl') 

def preprocess_data(df, encoder, scaler):
    columns_to_encode = ['ever_married', 'gender', 'work_type', 'Residence_type', 'smoking_status']
    df[columns_to_encode] = df[columns_to_encode].astype(str)
    
    encoded_array = encoder.transform(df[columns_to_encode])
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)
    
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
    
    df = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    
    num_features = ['age', 'avg_glucose_level', 'bmi']
    df[num_features] = scaler.transform(df[num_features])
    
    return df


def main():
    st.title("Прогнозування інсульту у пацієнта")
    st.write("Цей додаток визначає ймовірність виникнення інсульту у людини")

    with st.form("data_form"):
        st.header("Введіть дані")

        gender = st.selectbox('gender', ["Male", "Female"])
        age = st.number_input('age', min_value=0)
        hypertension = st.selectbox('hypertension', [0, 1])
        heart_disease = st.selectbox('heart_disease', [0, 1])
        ever_married = st.selectbox('ever_married', ["No", "Yes"])
        work_type = st.selectbox('work_type', ["children", "Govt_jov", "Never_worked", "Private", "Self-employed"] )
        Residence_type = st.selectbox('Residence_type', ["Rural", "Urban"])
        avg_glucose_level = st.number_input('avg_glucose_level', min_value=0 )
        bmi =  st.number_input('bmi', min_value=0)
        smoking_status = st.selectbox('smoking_status', ["formerly smoked", "never smoked", "smokes", "Unknown"]
    
    )
        # "Підписник кіно-пакету",
        # [0, 1],
        # help="0 - Ні, 1 - Так.


        submitted = st.form_submit_button("Передбачити")

        if submitted:
            
            # Підготовка даних для моделі
            input_data = pd.DataFrame({
                "gender": [gender],
                "age": [age],
                "hypertension": [hypertension],
                "heart_disease": [heart_disease],
                "ever_married": [ever_married],
                "work_type": [work_type],
                "Residence_type": [Residence_type],
                "avg_glucose_level": [avg_glucose_level],
                "bmi": [bmi],
                "smoking_status": [smoking_status],
            }) 
        

            processed_data = preprocess_data(input_data, encoder, scaler)


            prediction = logistic_model.predict(processed_data.values)
            probability = logistic_model.predict_proba(processed_data.values)

            if prediction[0] == 1:
                st.error(f"Пацієнт з відповідним ознаками має високу ймовірність інсульту ({probability[0][1]*100:.2f}%).")
            else:
                st.success(f"Пацієнт маж низьку йомвірність інсульту({probability[0][1]*100:.2f}%).")

if __name__ == "__main__":
    main()