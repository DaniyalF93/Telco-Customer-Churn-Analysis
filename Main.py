import pandas as pd
from TrainData import TrainData

from ReadCleanAndSplitData import ReadCleanAndSplitData

data_path = "Telco-Customer-Churn.csv"
read_split_obj = ReadCleanAndSplitData(data_path)
training_obj = TrainData()

X_train_scaled, X_test_scaled, y_train, y_test = read_split_obj.read_data()


model, acc = training_obj.train_logistic_regression(
    X_train_scaled, X_test_scaled, y_train, y_test)

print(model)
print(acc)
