from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

# import batch

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def prepare_data(df: pd.DataFrame, var: str):
    return df.sort_values(by=var)        

def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    result_df = prepare_data(df=df, var="dropOff_datetime")
    expected_df = pd.read_parquet("sample.parquet")
    print(expected_df)
    print(f"# rows: {len(expected_df)}")
    assert pd.DataFrame.equals(result_df, expected_df)

if __name__ == "__main__":
    test_prepare_data()
    

# def read_text(file):
#     test_directory = Path(__file__).parent

#     with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
#         return f_in.read().strip()


# def test_base64_decode():
#     base64_input = read_text('data.b64')

#     actual_result = model.base64_decode(base64_input)
#     expected_result = {
#         "ride": {
#             "PULocationID": 130,
#             "DOLocationID": 205,
#             "trip_distance": 3.66,
#         },
#         "ride_id": 256,
#     }

#     assert actual_result == expected_result


# def test_prepare_features():
#     model_service = model.ModelService(None)

#     ride = {
#         "PULocationID": 130,
#         "DOLocationID": 205,
#         "trip_distance": 3.66,
#     }

#     actual_features = model_service.prepare_features(ride)

#     expected_fetures = {
#         "PU_DO": "130_205",
#         "trip_distance": 3.66,
#     }

#     assert actual_features == expected_fetures


# class ModelMock:
#     def __init__(self, value):
#         self.value = value

#     def predict(self, X):
#         n = len(X)
#         return [self.value] * n


# def test_predict():
#     model_mock = ModelMock(10.0)
#     model_service = model.ModelService(model_mock)

#     features = {
#         "PU_DO": "130_205",
#         "trip_distance": 3.66,
#     }

#     actual_prediction = model_service.predict(features)
#     expected_prediction = 10.0

#     assert actual_prediction == expected_prediction


# def test_lambda_handler():
#     model_mock = ModelMock(10.0)
#     model_version = 'Test123'
#     model_service = model.ModelService(model_mock, model_version)

#     base64_input = read_text('data.b64')

#     event = {
#         "Records": [
#             {
#                 "kinesis": {
#                     "data": base64_input,
#                 },
#             }
#         ]
#     }

#     actual_predictions = model_service.lambda_handler(event)
#     expected_predictions = {
#         'predictions': [
#             {
#                 'model': 'ride_duration_prediction_model',
#                 'version': model_version,
#                 'prediction': {
#                     'ride_duration': 10.0,
#                     'ride_id': 256,
#                 },
#             }
#         ]
#     }

#     assert actual_predictions == expected_predictions
