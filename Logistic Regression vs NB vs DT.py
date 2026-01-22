import pandas as pd

import ReadCleanAndSplitData as readAndSplit

data_path = "Telco-Customer-Churn.csv"
read_split_obj = readAndSplit(data_path)

X_train_scaled, X_test_scaled, y_train, y_test = read_split_obj.read_data()

# Branched
