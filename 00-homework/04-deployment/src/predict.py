import os
import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Prediction of FHV trips')
parser.add_argument('-y', '--year', type=str, help='Year')
parser.add_argument('-m', '--month', type=str, help='Month')
args = parser.parse_args()
year = args.year
month = args.month

PATH = "../src/"

with open(os.path.join(PATH, 'model.bin'), 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(os.path.join(PATH, f"data/fhv_tripdata_{year}-{month}.parquet"))

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print(y_pred.mean())