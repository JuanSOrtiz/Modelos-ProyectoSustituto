FROM python:3.9

ADD model.pkl .
ADD predict.py .
ADD runScripts.py .
ADD test_data_input.csv .
ADD test_data_target.csv .
ADD test.csv .
ADD train.csv .
ADD train.py .

RUN pip install scikit-learn
RUN pip install xgboost
RUN pip install opendatasets
RUN pip install pandas
RUN pip install numpy
RUN pip install loguru
RUN pip install argparse


CMD ["python", "./runScripts.py"]