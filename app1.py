"""
Created on Mon Jun 17 02:20:31 2020

@author: Sheetal Jain and Shamith Jain
"""

import numpy as np
import pickle
import pandas as pd


import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

pickle_in = open("D:\\EV_selection\\artifacts\\model.pkl", "rb")
classifier = pickle.load(pickle_in)


# @app.route('/')
def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predict_note_authentication(
    ModelYear, Make, Model, ElectricVehicleType, ElectricRange
):
    """Let's Select the EV Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: MODELYEAR
        in: query
        type: number
        required: true
      - name: MAKE
        in: query
        type: string
        required: true
      - name: Model
        in: query
        type: string
        required: true
      - name: ElectricVehicleType
        in: query
        type: string
        required: true
      - name: ElectricRange
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    # prediction = classifier.predict(
    #     [[MODELYEAR, MAKE, Model, ElectricVehicleType, ElectricRange]]
    # )
    data = CustomData(
        Model_Year=ModelYear,
        Make=Make,
        Model=Model,
        Electric_Vehicle_Type=ElectricVehicleType,
        Electric_range=float(ElectricRange),
    )
    pred_df = data.get_data_as_data_frame()
    prediction = PredictPipeline()
    results = prediction.predict(pred_df)
    if results < 1:
        A = "Eligibility unknown as battery range has not been researched"
    elif 1 < results < 2:
        A = "Clean Alternative Fuel Vehicle Eligible"
    else:
        A = "Not eligible due to low battery range"
    return A


def main():
    st.title("EV SELECTOR")
    html_temp = """
    <div style="background-color:Darkblue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit EV Selection ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    MODELYEAR = st.text_input("Model Year", placeholder="Type Here")
    MAKE = st.text_input("Make", placeholder="Type Here")
    Model = st.text_input("Model", placeholder="Type Here")
    ElectricVehicleType = st.text_input(
        "Electric Vehicle Type", placeholder="Type Here"
    )
    ElectricRange = st.text_input("Electric Range", placeholder="Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(
            MODELYEAR, MAKE, Model, ElectricVehicleType, ElectricRange
        )
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("LETS LEARN TO PREDICT CLASSES")
        st.text("Built with Streamlit")


if __name__ == "__main__":
    main()
