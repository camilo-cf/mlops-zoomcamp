FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 
RUN pip install pipenv 

WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system

COPY [ "predict.py", "predict.py"]

CMD [ "python", "predict.py", "-y", "2021", "-m", "04" ]

