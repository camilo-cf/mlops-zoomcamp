from datetime import datetime
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

# import batch

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def prepare_data(df: pd.DataFrame, var: str):
    return df.sort_values(by=var)     

def get_output_path(year=2021, month=1):
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month) 

def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    S3_ENDPOINT_URL = "http://localhost:4566"
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    
    output = get_output_path()
    print(output)

    df.to_parquet(
                    output,
                    engine='pyarrow',
                    compression=None,
                    index=False,
                    storage_options=options
                )

if __name__ == "__main__":
    test_prepare_data()
    