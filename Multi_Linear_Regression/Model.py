import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#from MeLinearRegressionModel import linear_regression_model
import pickle
# Let's create a function to create adjusted R-Squared
def adj_r2(x,y):
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
data =pd.read_csv('Admission_Prediction.csv')
#data.head()
data.describe(include='all')
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())
data= data.drop(columns = ['Serial No.'])
data.head()


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
#         plt.ylabel('Chance of admit',fontsize=20)
    plotnumber+=1
plt.tight_layout()


y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])


plt.figure(figsize=(20,30), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=15 :
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(X[column],y)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Chance of Admit',fontsize=20)
    plotnumber+=1
plt.tight_layout()
scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns
print(vif)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=350)

regression = LinearRegression()

regression.fit(x_train,y_train)
# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))
# prediction using the saved model
loaded_model = pickle.load(open(filename, 'rb'))

#print(a)
a=regression.score(x_train,y_train)
b=regression.score(x_test,y_test)
print('The model score on training data is ', a ,', Model score on Testing data  is ', b)
print('The r_square on training data is ',adj_r2(x_train,y_train),' The r_squared on testing data is ', adj_r2(x_test,y_test))
#regression.score(x_train,y_train)
#regression.score(x_test,y_test)
# Gre=input('Enter GRE score : ')
# toefl=input('Enter the TOEFL score : ')
# sop=input('enter SOP rating : ')
# lor=input('Enter the LOR rating : ')
# un_rate=input('Enter Univercity ratings : ')
# cgpa=input('Enter CGPA : ')
# Research=input('Done research ?1/0 : ')
# prediction=loaded_model.predict(scaler.transform([[Gre,toefl,un_rate,sop,lor,cgpa,Research]]))
# print(prediction)
