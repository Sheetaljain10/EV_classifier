import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            # train_array.reshape(-1, 1)
            # test_array.reshape(-1, 1)
            # print(train_array)
            # print(test_array.shape)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Linear Regression": LinearRegression(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm="SAMME"),
            }

            params = {
                "Decision Tree": {
                    "max_depth": [10, 10, 10, 10, 10, 10],
                    "min_samples_leaf": [5, 5, 5, 5, 5, 5],
                    "criterion": [
                        "gini",
                        "log_loss",
                        "entropy",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [10, 20, 30, 40, 50, 60],
                    "max_features": ["sqrt", "log2", "auto", "sqrt", "log2", "auto"],
                },
                "Linear Regression": {},
                "XGBClassifier": {
                    "max_depth": [6, 8, 10, 14, 16, 18, 20],
                    "early_stoping_rounds": [10, 15, 20, 25, 30, 35],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # print((model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
