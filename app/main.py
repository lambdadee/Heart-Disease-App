import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("HeartDPr/Heart_Disease_Prediction.csv")
    data["Heart Disease"] = data["Heart Disease"].map({"Absence" : 0, "Presence" : 1})
    print(data.info())
    return data



def get_sidebar():
    st.sidebar.header("Measurements")
    data = get_clean_data()

    

    slider_labels = [
        ("Patient Age", "Age"),
        ("Sex", "Sex"),
        ("Chest Pain", "Chest pain type"),
        ("Blood Pressure", "BP"),
        ("Cholesterol", "Cholesterol"),
        ("Fibrin Split Products", "FBS over 120"),
        ("Electrocardiogram", "EKG results"),
        ("Maximu Heart Rate", "Max HR"),
        ("Exercise Angina", "Exercise angina"),
        ("ST Depression", "ST depression"),
        ("Slope of ST", "Slope of ST"),
        ("Number of vessels fluro", "Number of vessels fluro"),
        ("Thallium", "Thallium")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_value(input_dict):
    data = get_clean_data()

    x = data.drop(["Heart Disease"], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()

        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict

    

def get_radar_chart(input_data):

    input_data = get_scaled_value(input_data)

    categories = [
        "Age", "Sex", "Chest pain type", "Blood pressure", "Cholesterol",
        "FBS", "EKG results", "Max HR", "Exercise angina", "ST depression",
        "Slope of ST", "Number of vessels fluro", "Thallium"
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["Age"], input_data["Sex"], input_data["Chest pain type"], input_data["BP"],
            input_data["Cholesterol"], input_data["FBS over 120"], input_data["EKG results"], input_data["Max HR"],
            input_data["Exercise angina"], input_data["ST depression"], input_data["Slope of ST"], input_data["Number of vessels fluro"],
            input_data["Thallium"]
        ],
        theta=categories,
        fill="toself",
        name= "Input Values"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Heart Disease Prediction")
    st.write("Heart disease is ")

    if prediction[0] == 0:
        st.write("<span class = 'diagnosis absent'>Absent</span>", unsafe_allow_html=True)
    else:
        st.write("<span class = 'diagnosis present'>Present</span>", unsafe_allow_html=True)
    st.write("Probability of being Absent: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Present: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist cardiopatholost in making a diagnosis. However, it should not be used as a subtitute for a professional diagnosis. ")

    




def main():
    st.set_page_config(
        page_title="Heart Disease Diagnosis",
        page_icon=":heart_icon:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = get_sidebar()


    with st.container():
        st.title("Heart Disease Diagnosis")
        st.markdown("<style>div.block-container{padding-top : 2rem;}</style>", unsafe_allow_html=True)
        st.write("Please connect this app to your cardiopathology lab to help diagnose Heart disease from your heart tissue or blood sample to identify heart diseases or mutations. This app predicts, using a machine learning model, whether there is presence of disease in the heart or not based on the measurements it receives from the cardiopathology Laboratory. You can also update the measurements by hand using the sliders in the sidebar. ")

        col1, col2 = st.columns((4,1))

        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)

        with col2:
            add_prediction(input_data)


if __name__ == "__main__":
    main()