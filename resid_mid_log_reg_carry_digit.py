import matplotlib.pyplot as plt
import matplotlib
#print(matplotlib.get_backend())
import pandas as pd
from config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('saved_runs/variable_digit_add_50/resid_mid_carry_log_reg_data.csv')

config = Config()
d_model = config.d_model
train_col_namews = [f"hidden_{i}" for i in range(d_model)]

# Split data into train and test
x_train = data[data["is_train"]][train_col_namews]
x_test = data[~data["is_train"]][train_col_namews]
y_train = data[data["is_train"]]["carry_1"]
y_test = data[~data["is_train"]]["carry_1"]

def get_accuracy(log_reg, x_test, y_test):
    y_pred = log_reg.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_number_of_non_zero_coefficients(log_reg):
    log_reg.coef_ = log_reg.coef_.reshape(-1)
    non_zero_indices = [i for i, x in enumerate(log_reg.coef_) if x != 0]
    return len(non_zero_indices)

def get_top_n_coef(log_reg, n):
    log_reg.coef_ = log_reg.coef_.reshape(-1)
    coef = log_reg.coef_
    # Sort the coefficients in descending order and get the top n
    top_n_indices = np.argsort(coef)[::-1][:n]
    return top_n_indices

# Plot the accuracy of the model for different values of C
c_values = np.linspace(0.0001, 0.01, 100)
accuracies = []
num_non_zero = []

for c in c_values:
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=c)
    log_reg.fit(x_train, y_train)
    # Calculate the accuracy
    accuracy = get_accuracy(log_reg, x_test, y_test)
    accuracies.append(accuracy*100)

    # Calculate the number of non-zero coefficients
    num_coef = get_number_of_non_zero_coefficients(log_reg)
    num_non_zero.append(num_coef)

top_3 = get_top_n_coef(log_reg, 3)
print(f"Top 3 coefficients: {top_3}")

# Simple plot to test
fig, ax = plt.subplots()
ax.plot(c_values, accuracies, label='Accuracy', color='b')
ax.set_xlabel('L1 penalty values')
ax.set_ylabel('Accuracy', color='b')
ax2 = ax.twinx()
ax2.plot(c_values, num_non_zero, label='Number of non-zero coefficients', color='r')
ax2.set_ylabel('Number of non-zero coefficients', color='r')
ax.legend()
plt.show()
