
# CO2 Emission-Regression For Cars Using Python - Linear and Polynom Regression

# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading The Data 

data = pd.read_csv('CO2 Emissions_Canada.csv')
print(data.head(10))

# Regression

X = data[["Vehicle Class","Transmission","Fuel Type","Engine Size(L)","Cylinders","Fuel Consumption Comb (L/100 km)"]]
Y = data[["CO2 Emissions(g/km)"]]
#X.head(10)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols = ["Engine Size(L)","Cylinders","Fuel Consumption Comb (L/100 km)"]
X[cols] = sc.fit_transform(X[cols])

from sklearn.preprocessing import OrdinalEncoder
oc = OrdinalEncoder()
cols = ["Vehicle Class","Transmission","Fuel Type"]
X[cols] = oc.fit_transform(X[cols])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,Y)

print(reg.score(X,Y))
print(X)
print(reg.predict([[14.0,21.0,12,2.3,-0.65,-0.37]]))

