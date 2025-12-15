import pandas as pd
import joblib
import streamlit as st


model_path = "message_model.pkl"
vectorizer_path = "message_vectorizer.pkl"
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title("message emotion app")
message =st.text_input("write your message here: ")

if st.button("emotion"):
    sample_data = pd.DataFrame({
    "message": [message]
    })


    sample_data['message'] = sample_data['message'].str.lower()

    converted_sample_data = vectorizer.transform(sample_data)

    make_recommendation = model.predict(converted_sample_data)


    st.success(f"emotion: {make_recommendation[0]}")
