from re import I
from xmlrpc.client import DateTime
import pandas as pd
import datetime
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect import get_run_logger

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        get_run_logger().info(f"The mean duration of training is {mean_duration}")
    else:
        get_run_logger().info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    get_run_logger().info(f"The shape of X_train is {X_train.shape}")
    get_run_logger().info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    get_run_logger().info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    get_run_logger().info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date: datetime.datetime = None):
    if date is None:
        date = datetime.datetime.now()
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    get_run_logger().info(f"The date is {date}")
    date_train = date - datetime.timedelta(days=60)
    year_train = date_train.year
    month_train = f"0{str(date_train.month)}" if date_train.month < 10 else str(date_train.month)

    date_val =  date - datetime.timedelta(days=30)
    year_val = date_val.year
    month_val = f"0{str(date_val.month)}" if date_val.month < 10 else str(date_val.month)


    train_path = f"./data/fhv_tripdata_{year_train}-{month_train}.parquet"
    val_path = f"./data/fhv_tripdata_{year_val}-{month_val}.parquet"
    get_run_logger().info(f"The path of training data is {train_path}")
    get_run_logger().info(f"The path of validation data is {val_path}")
    return train_path, val_path, str(date.date())


@flow#(task_runner=SequentialTaskRunner())
def main(date = "2021-08-15"):
    categorical = ['PUlocationID', 'DOlocationID']
    
    train_path, val_path, date = get_paths(date).result()
    
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    with open(f'model-{date}.pkl','wb') as f:
        pickle.dump(dv, f)

# Usual run
# main()

# To set an schedule
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="cron-schedule-deployment",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
