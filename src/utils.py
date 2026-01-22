import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj: The Python object to be saved.
    """
    import pickle

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    logging.info("Evaluating models...")
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]

            # Perform Grid Search CV
            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)
            
            # Train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Calculate R2 score
            r2_square = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = r2_square

        return report
    except Exception as e:
        raise CustomException(e, sys)