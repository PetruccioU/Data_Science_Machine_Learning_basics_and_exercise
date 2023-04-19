import numpy
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import sklearn
import pandas
from sklearn import linear_model

print('========================')
#Median
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.median(speed)
print('Median:')
print(x)

print('========================')
#Mode
x = stats.mode(speed)
print('Mode:')
print(x)

print('========================')
#Standard Deviation is often represented by the symbol Sigma: σ
speed = [32,111,138,28,59,77,97]
x = numpy.std(speed)
print('Standard Deviation:')
print(x)

print('========================')
#Variance is often represented by the symbol Sigma Squared: σ2
speed = [32,111,138,28,59,77,97]
x = numpy.var(speed)
print('Variance:')
print(x)

print('========================')
#Percentile
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 75)
print('Percentile:')
print(x)

print('========================')
print('Plot 1 Normal Data Distribution')
#Normal Data Distribution
x = numpy.random.normal(5.0, 1.0, 100000)
plt.hist(x, 100)
plt.show()

print('========================')
print('Plot 2 Scatter Plot')
#Scatter Plot
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show()

print('========================')
print('Plot 3 Random Data Distributions')
#Random Data Distributions
x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)
plt.scatter(x, y)
plt.show()

print('========================')
print('Plot 4 Linear Regression')
#Linear Regression
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
    return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
#Predict
speed = myfunc(10)
print('Predict speed:')
print(speed)

#Bad fit Linear Regression
print('Plot 5 Bad fit Linear Regression')
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope_bad, intercept_bad, r_bad, p_bad, std_err_bad = stats.linregress(x, y)
def myfunc_bad(x):
  return slope_bad * x + intercept_bad
mymodel_bad = list(map(myfunc_bad, x))
plt.scatter(x, y)
plt.plot(x, mymodel_bad)
plt.show()
#Relationship
print('Relationship for BAD Linear Regression:')
print(r_bad)

print('========================')
print('Plot 6 Polynomial Regression')
#Polynomial Regression
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
#R-squared, for relationship in Polynomial Regression
from sklearn.metrics import r2_score
print('R-squared, for relationship in Polynomial Regression')
print(r2_score(y, mymodel(x)))
#Predict Values
speed = mymodel(17)
print('Predicted Speed:')
print(speed)

#Bad Fit Polynomial Regression
print('Plot 7 Bad Fit Polynomial Regression')
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(2, 95, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
#Bad R_squared:
print('R-squared, for BAD relationship in Polynomial Regression:')
print(r2_score(y, mymodel(x)))

print('========================')
print('Multiple Regression:')
#Multiple Regression
df = pandas.read_csv("data.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
regr.fit(X, y)
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])
print('predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:')
print(predictedCO2)
#Coefficient
print('Weight: 0.00755095')
print('Volume: 0.00780526')
print(regr.coef_)


