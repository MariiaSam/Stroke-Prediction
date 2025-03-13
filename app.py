import streamlit as st
import joblib
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

        gender = st.selectbox('Gender', ["Male", "Female"], index=None, help='Patient`s gender', label_visibility="visible", placeholder="Choose patient`s gender")
        age = st.number_input('Age', min_value=0, value=None, help='Patient`s age', placeholder="What is the patient's age?")
        hypertension = st.selectbox('The presence of hypertension', [0, 1], index=None, help='(0 - no, 1 - yes)', label_visibility="visible", placeholder="Does the patient have hypertension?")
        heart_disease = st.selectbox('The presence of heart disease', [0, 1], index=None, help='(0 - no, 1 - yes)', label_visibility="visible", placeholder="Does the patient have heart disease?")
        ever_married = st.selectbox('Does the patient have a marriage?', ["No", "Yes"], index=None, help='"No", "Yes"', label_visibility="visible", placeholder="Does the patient have a marriage?")
        work_type = st.selectbox('Patient`s occupation', ["Children", "Govt_jov", "Never_worked", "Private", "Self-employed"], index=None, label_visibility="visible", placeholder="Patient's occupation" )
        Residence_type = st.selectbox('Patient`s place of residence', ["Rural", "Urban"], index=None, label_visibility="visible", placeholder="Patient's place of residence" )
        avg_glucose_level = st.number_input('avg_glucose_level',  min_value=0,  value=None, help='Patient`s averageblood glucose level', placeholder="What is the average blood glucose level?")
        bmi =  st.number_input('Body mass index', min_value=0,  value=None, help='Patient`s bmi', placeholder="What is the patient's bmi?")
        smoking_status = st.selectbox('Smoking status of the patient', ["Formerly smoked", "Never smoked", "Smokes", "Unknown"], index=None, label_visibility="visible", placeholder="Does the patient smoke?"
    )

        btn = st.form_submit_button("Check")

        if btn:
            
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
                st.error(f"A patient with the relevant features has a high probability of stroke({probability[0][1]*100:.2f}%).")
            else:
                st.success(f"Patient has a low probability of stroke({probability[0][1]*100:.2f}%).")

if __name__ == "__main__":
    main()