# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:49:42 2019

@author: hp
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from scipy.stats import norm
from operator import itemgetter

#importing dataset
dfCollege = pd.read_csv('college.csv')
dfNBA = pd.read_csv('nba.csv')
dfDraftClass = pd.read_csv('draft.csv')

dfCombined = pd.read_csv('college-nba.csv')


#Basic correlation between college 3pt% and NBA 3pt%
plt.style.use('fivethirtyeight')
 
collegenba3pt, ax = plt.subplots()

ax.scatter(dfCollege['3P%'], dfNBA['3P%'], color = 'orange')
ax.axvline(x = np.mean(dfCollege['3P%']), color = 'black')
ax.axhline(y = np.mean(dfNBA['3P%']), label = "Average", color = 'black')
collegenba3pt.suptitle("Correlation between college and NBA 3PT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("College 3PT%")
ax.set_ylabel("NBA 3PT%")
 
ax.plot(np.unique(dfCollege['3P%']), 
        np.poly1d(np.polyfit(dfCollege['3P%'], dfNBA['3P%'], 1))(np.unique(dfCollege['3P%'])))

ax.legend(loc='best', prop={'size': 9, "family": "Rockwell"})
 
slope, intercept, r_value, p_value, std_err = linregress(dfCollege['3P%'], dfNBA['3P%'])
print("Wingspan and DWS: slope =", slope, ", intercept =", intercept, ", r_value =", r_value,
    ", p_value =", p_value, ", std_err =", std_err)
rsqaured = r_value ** 2
rpString = "r = " + str(round(r_value, 3)) + ", p = " + str(round(p_value, 3)) + ", rsquared = " + str(round(rsqaured, 3))
collegenba3pt.show()



#Basic correlation between college PPG and NBA PPG
plt.style.use('fivethirtyeight')
 
collegenbaPPG, ax = plt.subplots()

ax.scatter(dfCollege['PTS/G'], dfNBA['PTS/G'], color = 'orange')
ax.axvline(x = np.mean(dfCollege['PTS/G']), color = 'black')
ax.axhline(y = np.mean(dfNBA['PTS/G']), label = "Average", color = 'black')
collegenbaPPG.suptitle("Correlation between college and NBA PPG", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("College PPG")
ax.set_ylabel("NBA PPG")
 
ax.plot(np.unique(dfCollege['PTS/G']), np.poly1d(np.polyfit(dfCollege['PTS/G'], dfNBA['PTS/G'], 1))(np.unique(dfCollege['PTS/G'])))

ax.legend(loc='best', prop={'size': 9, "family": "Rockwell"})
 
slope, intercept, r_value, p_value, std_err = linregress(dfCollege['PTS/G'], dfNBA['PTS/G'])
print("Wingspan and DWS: slope =", slope, ", intercept =", intercept, ", r_value =", r_value,
    ", p_value =", p_value, ", std_err =", std_err)
rsqaured = r_value ** 2
rpString = "r = " + str(round(r_value, 3)) + ", p = " + str(round(p_value, 3)) + ", rsquared = " + str(round(rsqaured, 3))



#Histograms for seeing how common PPG is

plt.style.use('fivethirtyeight')
collegePpgHist, ax = plt.subplots()

ax.hist(dfCollege['PTS/G'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
collegePpgHist.suptitle("Histogram of sample's college PPG", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("PPG")
ax.set_ylabel("Frequency")

overall_mean = dfCollege['PTS/G'].mean()
overall_std = dfCollege['PTS/G'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#Histogram of NBA ppg

plt.style.use('fivethirtyeight')
nbaPpgHist, ax = plt.subplots()

ax.hist(dfNBA['PTS/G'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
nbaPpgHist.suptitle("Histogram of sample's NBA PPG", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("PPG")
ax.set_ylabel("Frequency")

overall_mean = dfNBA['PTS/G'].mean()
overall_std = dfNBA['PTS/G'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#Draft class's ppg histogram

plt.style.use('fivethirtyeight')
draftClassPpgHist, ax = plt.subplots()

ax.hist(dfDraftClass['PTS/G'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
draftClassPpgHist.suptitle("Histogram of 2018 draft class's college PPG", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("PPG")
ax.set_ylabel("Frequency")

overall_mean = dfDraftClass['PTS/G'].mean()
overall_std = dfDraftClass['PTS/G'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#College 3pt% histgram

plt.style.use('fivethirtyeight')
college3ptHist, ax = plt.subplots()

ax.hist(dfCollege['3P%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")

college3ptHist.suptitle("Histogram of sample's college 3PT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("3PT%")
ax.set_ylabel("Frequency")

overall_mean = dfCollege['3P%'].mean()
overall_std = dfCollege['3P%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#NBA 3pt% histogram

plt.style.use('fivethirtyeight')
nba3ptHist, ax = plt.subplots()

ax.hist(dfNBA['3P%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
nba3ptHist.suptitle("Histogram of sample's NBA 3PT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("3PT%")
ax.set_ylabel("Frequency")

overall_mean = dfNBA['3P%'].mean()
overall_std = dfNBA['3P%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#Draft class's 3pt% histogram

plt.style.use('fivethirtyeight')
draftClass3ptHist, ax = plt.subplots()

ax.hist(dfDraftClass['3P%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")

draftClass3ptHist.suptitle("Histogram of 2018 draft class's 3PT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("3PT%")
ax.set_ylabel("Frequency")

overall_mean = dfDraftClass['3P%'].mean()
overall_std = dfDraftClass['3P%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#College ft%

plt.style.use('fivethirtyeight')
collegeFtHist, ax = plt.subplots()

ax.hist(dfCollege['FT%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
collegeFtHist.suptitle("Histogram of sample's college FT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("FT%")
ax.set_ylabel("Frequency")

overall_mean = dfCollege['FT%'].mean()
overall_std = dfCollege['FT%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#NBA ft%

plt.style.use('fivethirtyeight')
nbaFtHist, ax = plt.subplots()

ax.hist(dfNBA['FT%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
nbaFtHist.suptitle("Histogram of sample's NBA FT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("FT%")
ax.set_ylabel("Frequency")

overall_mean = dfNBA['FT%'].mean()
overall_std = dfNBA['FT%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")


#Draft class's ft%

plt.style.use('fivethirtyeight')
draftClassFtHist, ax = plt.subplots()

ax.hist(dfDraftClass['FT%'], bins = 16, edgecolor = 'white', linewidth = 3, normed = True, label = "Actual distribution")
draftClassFtHist.suptitle("Histogram of draft class's college FT%", weight = 'bold', size = 18, y = 1.05)
ax.set_xlabel("FT%")
ax.set_ylabel("Frequency")

overall_mean = dfDraftClass['FT%'].mean()
overall_std = dfDraftClass['FT%'].std()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, overall_mean, overall_std)
ax.plot(x, p, 'k', linewidth=5, color='orange', label = "Normal distribution")






# Create the train and test set

train, test = train_test_split(dfCombined, test_size=0.2, random_state=99)
 
xtrain = train[['FG%-Col', '2P%-Col', '3P%-Col', 'FT%-Col', 'TS%-Col', 'eFG%-Col', 'PTS/G-Col']]
ytrain = train[['PTS/G-NBA']]
 
xtest = test[['FG%-Col', '2P%-Col', '3P%-Col', 'FT%-Col', 'TS%-Col', 'eFG%-Col', 'PTS/G-Col']]
ytest = test[['PTS/G-NBA']]


# Create a linear regression model and test its accuracy

linReg = linear_model.LinearRegression()
linReg.fit(xtrain, ytrain)

y_predLin = linReg.predict(xtest)

print(y_predLin - ytest)

print('Coefficients: \n', linReg.coef_)
print("Mean squared error: %.3f" % mean_squared_error(ytest, y_predLin))
print('Variance score: %.3f' % r2_score(ytest, y_predLin))


# Let's try a ridge regression

ridgeReg = linear_model.Ridge(alpha = .1)

ridgeReg.fit(xtrain, ytrain)

y_predRidge = ridgeReg.predict(xtest)

print(y_predRidge - ytest)

print('Coefficients: \n', ridgeReg.coef_)
print("Mean squared error: %.3f" % mean_squared_error(ytest, y_predRidge))
print('Variance score: %.3f' % r2_score(ytest, y_predRidge))



# Let's try a support vector regression instead

svr_rbf = SVR(kernel='rbf', gamma=1e-3, C=200, epsilon=0.1)
svr_rbf.fit(xtrain, ytrain.values.ravel())

y_rbf = svr_rbf.predict(xtest)

print(y_rbf - ytest['PTS/G-NBA'])

print("Mean squared error: %.3f" % mean_squared_error(ytest, y_rbf))
print('Variance score: %.3f' % r2_score(ytest, y_rbf))


# Let's graph the mean squared error of all three models

msePlot, ax = plt.subplots()

mseScores = [mean_squared_error(ytest, y_predLin), mean_squared_error(ytest, y_predRidge), mean_squared_error(ytest, y_rbf)]
x_pos = np.arange(len(mseScores))

ax.bar(x_pos, mseScores, edgecolor = 'white', linewidth = 3)

mseNames = ["Linear regression", "Ridge regression", "Support vector\nregression"]

labels = [i for i in mseNames]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .5
    ax.text(rect.get_x() + rect.get_width() / 1.8, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white', size = 16)

msePlot.suptitle("Mean squared error (MSE) of regressions", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel("MSE (lower is better)")


# Let's graph the variance score of all three models

r2plot, ax = plt.subplots()

r2scores = [r2_score(ytest, y_predLin), r2_score(ytest, y_predRidge), r2_score(ytest, y_rbf)]
x_pos = np.arange(len(r2scores))

ax.bar(x_pos, r2scores, edgecolor = 'white', linewidth = 3)

r2names = ["Linear regression", "Ridge regression", "Support vector\nregression"]

labels = [i for i in r2names]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .025
    ax.text(rect.get_x() + rect.get_width() / 1.8, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white', size = 16)

r2plot.suptitle("Variance score of regressions", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel(r"R$^{\rm 2}$ (higher is better)")
ax.xaxis.set_visible(False)



# Let's plot the coefficients

linearCoef = linReg.coef_
ridgeCoef = ridgeReg.coef_

linearCoefPlot = []
ridgeCoefPlot = []

for i in linearCoef[0]:
    linearCoefPlot.append(i)

for i in ridgeCoef[0]:
    ridgeCoefPlot.append(i)   

x_lin = np.arange(len(linearCoefPlot))
x_ridge = np.arange(len(ridgeCoefPlot))

coefPlot, ax = plt.subplots()

ax.scatter(x_lin, linearCoefPlot, label = "Linear regression")
ax.scatter(x_ridge, ridgeCoefPlot, color = 'orange', label = "Ridge regression")

ax.set_xticklabels(['FG%', '2P%', '3P%', 'FT%', 'TS%', 'eFG%', '3PAr', 'FTr', 'PTS/G'])
ax.set_xticks(x_lin)
ax.axhline(y = 0, color = 'black')

ax.set_ylabel("Correlation to NBA PPG")
ax.set_xlabel("College stat")
coefPlot.suptitle("Regression coefficients", weight = 'bold', size = 18, y = 1.005)
ax.legend(loc = 'best')



# Let's see what the linear model predicts for this draft class

draftClassTest = dfDraftClass.loc[:, 'FG%': 'PTS/G']

linear_draftClass = linReg.predict(draftClassTest)

names = dfDraftClass.iloc[:, 0]

for i, j in zip(linear_draftClass, names):
    print(i, j)
    
    
# Let's see what the ridge regression predicts for this draft class

ridge_draftClass = ridgeReg.predict(draftClassTest)

for i, j in zip(ridge_draftClass, names):
    print(i, j)



# Let's see what the support vector regression predicts for this draft class

svr_draftClass = svr_rbf.predict(draftClassTest)

for i, j in zip(svr_draftClass, names):
    print(i, j)




# Let's plot the linear regression predictions

plt.style.use('fivethirtyeight')
linRegPredictions, ax = plt.subplots()

linearPlot = []

for i in linear_draftClass:
    linearPlot.append(float(i))
    
combinedLinear = [[i, j] for i, j in zip(names, linearPlot)]

sortedLinear = sorted(combinedLinear, key = itemgetter(1), reverse = True)
print(sortedLinear)

sortedLinearData = [row[1] for row in sortedLinear]
x_pos = np.arange(len(sortedLinearData))

ax.bar(x_pos, sortedLinearData, edgecolor = 'white', linewidth = 3)

labels = [row[0] for row in sortedLinear]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .5
    ax.text(rect.get_x() + rect.get_width() / 1.75, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white')

linRegPredictions.suptitle("Linear regression predicted PPG", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel("Predicted PPG")




# Let's plot the ridge regression predictions

ridgeRegPredictions, ax = plt.subplots()

ridgePlot = []

for i in ridge_draftClass:
    ridgePlot.append(float(i))
    
combinedRidge = [[i, j] for i, j in zip(names, ridgePlot)]

sortedRidge = sorted(combinedRidge, key = itemgetter(1), reverse = True)
print(sortedRidge)

sortedRidgeData = [row[1] for row in sortedRidge]
x_pos = np.arange(len(sortedRidgeData))

ax.bar(x_pos, sortedRidgeData, edgecolor = 'white', linewidth = 3)

labels = [row[0] for row in sortedRidge]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .5
    ax.text(rect.get_x() + rect.get_width() / 1.75, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white')

ridgeRegPredictions.suptitle("Ridge regression predicted PPG", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel("Predicted PPG")






# Let's plot the ridge regression predictions

svrPredictions, ax = plt.subplots()

svrPlot = []

for i in svr_draftClass:
    svrPlot.append(float(i))
    
combinedSVR = [[i, j] for i, j in zip(names, svrPlot)]

sortedSVR = sorted(combinedSVR, key = itemgetter(1), reverse = True)
print(sortedSVR)

sortedSVRdata = [row[1] for row in sortedSVR]
x_pos = np.arange(len(sortedSVRdata))

ax.bar(x_pos, sortedSVRdata, edgecolor = 'white', linewidth = 3)

labels = [row[0] for row in sortedSVR]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .5
    ax.text(rect.get_x() + rect.get_width() / 1.75, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white')

svrPredictions.suptitle("Support vector regression predicted PPG", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel("Predicted PPG")





# Let's see who has the highest average ppg among the three models

avgPredictions, ax = plt.subplots()

averagePred = []

for i, j, h in zip(linear_draftClass, ridge_draftClass, svr_draftClass):
    averagePred.append(float((i + j + h) / 3))

combinedAvg = [[i, j] for i, j in zip(names, averagePred)]

sortedAvg = sorted(combinedAvg, key = itemgetter(1), reverse = True)
print(sortedAvg)

sortedAvgData = [row[1] for row in sortedAvg]
x_pos = np.arange(len(sortedAvgData))

ax.bar(x_pos, sortedAvgData, edgecolor = 'white', linewidth = 3)

labels = [row[0] for row in sortedAvg]

rects = ax.patches
for rect, label in zip(rects, labels):
    height = .5
    ax.text(rect.get_x() + rect.get_width() / 1.75, height, label,
            ha='center', va='bottom', rotation = 'vertical', color = 'white')

avgPredictions.suptitle("Average predicted PPG from 3 regressions", weight = 'bold', size = 18, y = 1.005)
ax.set_ylabel("Predicted PPG")
