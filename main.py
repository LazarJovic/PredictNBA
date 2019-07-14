import pandas as pd
from dataPreparation import DataPreparation
from MLAlgorithms import MLAlgorithms

# Making pandas DataFrame from .csv file
data_frame = pd.read_csv("datasets\\NBA14-17.csv", parse_dates=["gmDate"])
data_frame.head()
data_frame.tail()
test_data_frame = pd.read_csv("datasets\\NBA17-18.csv", parse_dates=["gmDate"])

# Train and Test data frames preparation
data_prepare = DataPreparation()
data_prepare.prepare_data_frame(data_frame, True)
data_prepare.prepare_data_frame(test_data_frame, False)

# Algorithms
algorithms = MLAlgorithms()
algorithms.svm_func(data_frame, test_data_frame)
algorithms.linear_regression_func(data_frame, test_data_frame)
algorithms.random_forest_func(data_frame, test_data_frame)
algorithms.naive_bayes(data_frame, test_data_frame)


