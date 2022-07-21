from datetime import datetime
from pathlib import Path

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
    