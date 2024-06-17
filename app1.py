# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 02:20:31 2020

@author: Sheetal Jain and Shamith Jain
"""

# # -*- coding: utf-8 -*-
# """
# Created on  May 15 12:50:04 2020

# @author: krish.naik
# """


import numpy as np
import pickle
import pandas as pd

import streamlit as st

from PIL import Image

# app=Flask(__name__)
# Swagger(app)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


# @app.route('/')
def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predict_note_authentication(
    MODELYEAR, MAKE, Model, ElectricVehicleType, ElectricRange
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

    prediction = classifier.predict(
        [[MODELYEAR, MAKE, Model, ElectricVehicleType, ElectricRange]]
    )
    print(prediction)
    return prediction


def main():
    st.title("EV SELECTOR")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    MODELYEAR = st.text_input("Model year", "Type Here")
    MAKE = st.text_input("Make", "Type Here")
    Model = st.text_input("Model", "Type Here")
    ElectricVehicleType = st.text_input("Electric Vehicle Type", "Type Here")
    ElectricRange = st.text_input("Electric Range", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(
            MODELYEAR, MAKE, Model, ElectricVehicleType, ElectricRange
        )
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("Lets LEARN TO PREDICT CLASSES")
        st.text("Built with Streamlit")


if __name__ == "__main__":
    main()
