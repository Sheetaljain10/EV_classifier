import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation

        """
        try:
            numerical_columns = ["Model Year", "Electric Range"]
            categorical_columns = [
                "Electric Vehicle Type",
                "Make",
                "Model",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path).iloc[:, [5, 6, 7, 8, 9, 10]]
            test_df = pd.read_csv(test_path).iloc[:, [5, 6, 7, 8, 9, 10]]

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
            # numerical_columns = ["Model Year", "Electric Range", "Base MSRP"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = pd.DataFrame(train_df[target_column_name])

            # print(input_feature_train_df.shape, " ", target_feature_train_df.shape)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = pd.DataFrame(test_df[target_column_name])

            # print(input_feature_test_df, "\n\n ", target_feature_test_df)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Encode the target column for categorical value.
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(
                target_feature_train_df
            )
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            # Ensure the target feature is reshaped correctly
            # target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            # target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            input_feature_test_arr = input_feature_test_arr.toarray()

            print("\n\nROWS:", input_feature_train_arr.shape)
            print("\n\nROWS: ", input_feature_test_arr.shape)

            # target_feature_train_df = np.reshape(target_feature_train_df, (149503, 1))
            # target_feature_test_df = np.reshape(target_feature_test_df, (37376, 1))

            # print(np.array(target_feature_train_df).shape)
            # print(np.array(target_feature_test_df).shape)

            # train_arr = np.hstack((input_feature_train_arr, target_feature_train_df))
            # test_arr = np.hstack((input_feature_test_arr, target_feature_test_df))

            # print(train_arr)

            # ERROR
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df),
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df),
            ]

            # print(train_arr)
            # print(test_arr)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
