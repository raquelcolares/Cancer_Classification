import streamlit as st
import requests
import base64
import time

GENDERS = {0: "Male",
           1: "Female"}

RISK = {0: "Low",
        1: "Medium",
        2: "High"}


def main():
    st.title(f":blue[Cancer Prediction] 🩺")
    model_type = st.selectbox(label="Select your model", options=["Neural Network", "XGBoost"])
    st.write(f"The model **{model_type}** was selected.")
    
    with st.sidebar:
        st.write(f"**Insert your data for the prediction:**")
        age = st.slider("**Age**", value=51, min_value=20, max_value=80, step=1)
        gender = st.selectbox("**Gender**", options=GENDERS.keys(), format_func=lambda x: GENDERS[x])
        bmi = st.slider("**BMI**", value=27.5, min_value=15.0, max_value=40.0, step=0.5)
        smoking = int(st.toggle("**Smoking**"))
        genetic_risk = st.selectbox("**Genetic Risk**", options=RISK.keys(), format_func=lambda x: RISK[x])
        physical_activity = st.slider("**Physical Activity h/week**", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
        alcohol_intake = st.slider("**Alcohol units consumed per week**", value=0.0, min_value=0.0, max_value=5.0, step=0.1)
        cancer_history = int(st.toggle("**Cancer History**"))
        classify = st.button("**Classify**")


    if classify:
        url = f"http://127.0.0.1:8000/cancer?"
        url += f"age={age}&gender={gender}&bmi{bmi}&smoking{smoking}"
        url += f"&genetic_risk={genetic_risk}&physical_activity={physical_activity}"
        url += f"&alcohol_intake={alcohol_intake}&cancer_history={cancer_history}&model_type={model_type}"

        with st.spinner("Classifying, please wait..."):
            time.sleep(2)
            
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()

            img = result["image"]
            img_decoded = base64.b64decode(img)
            
            st.image(img_decoded)
        else:
            st.error("An error occured")
            st.error(response.json())  
            
            
if __name__ == "__main__":
    st.set_page_config(
        page_title="Cancer Prediction",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/raquelcolares',
            'Report a bug': "https://github.com/raquelcolares",
            'About': "# Cancer Prediction - MLPClassifier and XGBoost"
            }
    )
    st.sidebar.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #ced9f1;
        }
    </style>
    """, unsafe_allow_html=True)
    main()
    