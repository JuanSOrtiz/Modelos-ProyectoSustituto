# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:40:32 2024

@author: JUAN
"""

import sys
import pandas as pd
import numpy as np

train_data_file  = 'train.csv'
test_data_file   = 'test.csv'
test_input_file  = 'test_data_input.csv'
test_target_file = 'test_data_target.csv'
test_preds_file  = 'test_predictions.csv'
model_file       = 'model.pkl'

python = sys.executable
!$python train.py --model_file $model_file --train_data_file $train_data_file --test_data_file $test_data_file --overwrite_model
!$python predict.py --model_file $model_file --input_file $test_input_file  --predictions_file test_predictions.csv

preds   = pd.read_csv(test_preds_file).values
targets = pd.read_csv(test_target_file).values

acc = np.mean(preds==targets)
print (f"accuracy on test {acc:.3f}")