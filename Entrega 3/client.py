# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:30:43 2024

@author: JUAN
"""

import requests

BASE_URL = 'http://localhost:5000'

def predict(data):
    endpoint = f'{BASE_URL}/predict'
    response = requests.post(endpoint, json=data)
    return response.json()

def train():
    endpoint = f'{BASE_URL}/train'
    response = requests.post(endpoint)
    return response.json()

if __name__ == '__main__':
    # Ejemplo de solicitud de predicción
    data = {"0": 1.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0, "5": 1.0, "6": 0.0, "7": 0.0, "8": 1.0, "9": 0.0, "10": 1.0, "11": 0.0, "12": 1.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 1.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 1.0, "22": 0.0, "23": 0.36870280969594504, "24": -0.5750395224864274, "25": 0.9228658473315263, "26": 0.0426903165460184, "27": 1.34335329497167233, "28": -1.210697730305341, "29": -2.0887694637879228, "30": 0.16024747976767847},
       
    prediction = predict(data)
    print("Prediction:", prediction)

    # Ejemplo de solicitud de entrenamiento
    train_result = train()
    print("Training Result:", train_result)
