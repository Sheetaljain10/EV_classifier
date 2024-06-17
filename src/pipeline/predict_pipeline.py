import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.Data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "D:\\EV_selection\\artifacts\\proprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Model_Year: int,
        Make: str,
        Model: str,
        Electric_Vehicle_Type: str,
        Electric_range: str,
    ):

        self.Model_year = Model_Year

        self.Make = Make

        self.Model = Model

        self.ElectricVehicleType = Electric_Vehicle_Type

        self.Electric_range = Electric_range

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Model Year": [self.Model_year],
                "Make": [self.Make],
                "Model": [self.Model],
                "Electric Vehicle Type": [self.ElectricVehicleType],
                "Electric Range": [self.Electric_range],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
