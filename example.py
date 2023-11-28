from denfis.denfis import DENFIS

import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

train_x = df[["data_1", "data_2", "data_3"]][:500]
train_y = df[["result"]][:500]

test_x = df[["data_1", "data_2", "data_3"]][500:]
test_y = df[["result"]][500:]

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

denfis = DENFIS(threshold_diameter=0.1, width_of_triangle=1.7)
denfis.train(train_x, train_y)

pred_y = denfis.predict(test_x)
print(pred_y[:10])
