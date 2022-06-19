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

PATH = "."

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

year = df["pickup_datetime"].dt.year
month = df["pickup_datetime"].dt.strftime('%m')
df['ride_id'] = year.map(str) +"/"+month.map(str) +"_" + df.index.astype('str')
df['prediction'] = y_pred
df_result = df[['ride_id', 'prediction']]

output_file = os.path.join(PATH, "data", "results.parquet")
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)