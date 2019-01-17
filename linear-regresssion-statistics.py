# Import Libraries
import pandas
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.model_selection import train_test_split

# Load dataset
url = "data/Swedish-insurance.xls"
names = ['total-no-of-claims', 'total-amount']
dataset = pandas.read_excel(url, names=names)

# Lets display the top 5 records in the Dataset
print(dataset.head())
# Checking data type of columns and if any NULL values are present
print(dataset.info())
# Displaying basic stats for the columns
print(dataset.describe())

#convert string columns to float
dataset[['total-no-of-claims','total-amount']] = dataset[['total-no-of-claims','total-amount']].astype(float)
#spit dataset into 2 parts - training and testing dataset
train, test =train_test_split(dataset,test_size=0.4)

#Simple Linear regression

#Estimate the Coefficients b0,b1 for linear model y=b0+b1*X
#convert dataframe to numpy array
x = dataset['total-no-of-claims'].values
y = dataset['total-amount'].values

#calculate mean
x_mean = sum(x) / float(len(x))
y_mean = sum(y) / float(len(y))

#b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
#Calculate variance
x_variance = sum([(value - x_mean)**2 for value in x])

#Calculate covariance - covariance(x, x_mean, y, y_mean)
covar = 0.0
for i in range(len(x)):
    covar += (x[i] - x_mean) * (y[i] - y_mean)


#Estimate the coefficients
b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
b0 = y_mean - b1 * x_mean
#Make the prediction
predictions = list()
actuals = list()
for row in test.values:
    yp = b0 + b1 * row[0]
    predictions.append(yp)
    actuals.append(row[1])
    
#Calculate RSME
sum_error = 0.0
for i in range(len(actuals)):
    prediction_error = predictions[i] - actuals[i]
    sum_error += (prediction_error ** 2)
mean_error = sum_error / float(len(actuals))
rmse = sqrt(mean_error)
print('RMSE: %.3f' % (rmse))
