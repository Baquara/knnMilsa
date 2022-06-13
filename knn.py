# numbers, stats, plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# sklearn support
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor


# machine learning algorithm of interest
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    
    # Load the data from this file
    data_file = ("./milsa.csv")
    
    # x data labels
    xnlabs = ['Funcionario']
    xqlabs = ['EstCivil','Inst','Filhos','Salario', 'Idade','Meses','Regiao']
    xlabs = xnlabs + xqlabs

    # y data labels
    ylabs = ['Salario']

    # Load data to dataframe
    df = pd.read_csv(data_file, header=None, names=xlabs)

    
    # Filter zero values of height/length/diameter
    df['Inst'] = df['Inst'].replace(['1o Grau'],'1')
    df['Inst'] = df['Inst'].replace(['2o Grau'],'2')
    df['Inst'] = df['Inst'].replace(['Superior'],'3')
    df['Inst'] = df['Inst'].astype(int)

    df['EstCivil'] = df['EstCivil'].replace(['casado'],'1')
    df['EstCivil'] = df['EstCivil'].replace(['solteiro'],'0')
    df['EstCivil'] = df['EstCivil'].astype(int)

    df['Filhos'] = df['Filhos'].fillna(0)
    df['Filhos'] = df['Filhos'].astype(int)

    df['Regiao'] = df['Regiao'].replace(['capital'],'0')
    df['Regiao'] = df['Regiao'].replace(['interior'],'1')
    df['Regiao'] = df['Regiao'].replace(['outro'],'2')
    df['Regiao'] = df['Regiao'].astype(int)

    correlation_matrix = df.corr()
    print(correlation_matrix["Salario"])



    df = df.drop("Funcionario", axis=1)
    
    
    
    return Bunch(data   = df,
                 target = df[ylabs],
                 feature_names = xqlabs,
                 target_names  = ylabs)

dataset = load_data()
x = dataset.data.drop("Salario", axis=1)
y = dataset.target

print (x)
print ("-"*20)
print (y.head())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)


knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)


d = {'EstCivil': [0], 'Inst': [2], 'Filhos': [1], 'Idade': [30], 'Meses': [10], 'Regiao': [0]}
df2 = pd.DataFrame(data=d)

prediction = knn_model.predict(df2)

print("e"*100)

print("DataFrame de entrada customizada:")
print(df2)
print("O salário previsto para essa entrada é: ")
print(prediction)



test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)


print("a"*100)
print(X_test)
print(X_test.iloc[:, 1])
print(type(X_test))

print("b"*100)




cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 4], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)
plt.show()


cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 4], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 1], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)
plt.show()


cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 1], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()





'''
url = ("./milsa.csv")
milsa = pd.read_csv(url, header=None) 



milsa.columns = [
    "Funcionario",	"EstCivil",	"Inst",	"Filhos",	"Salario",	"Idade",	"Meses",	"Regiao"

]

milsa = milsa.drop("Funcionario", axis=1)


milsa['Inst'] = milsa['Inst'].replace(['1o Grau'],'1')
milsa['Inst'] = milsa['Inst'].replace(['2o Grau'],'2')
milsa['Inst'] = milsa['Inst'].replace(['Superior'],'3')
milsa['Inst'] = milsa['Inst'].astype(int)

milsa['EstCivil'] = milsa['EstCivil'].replace(['casado'],'1')
milsa['EstCivil'] = milsa['EstCivil'].replace(['solteiro'],'0')
milsa['EstCivil'] = milsa['EstCivil'].astype(int)

milsa['Filhos'] = milsa['Filhos'].fillna(0)
milsa['Filhos'] = milsa['Filhos'].astype(int)

milsa['Regiao'] = milsa['Regiao'].replace(['capital'],'0')
milsa['Regiao'] = milsa['Regiao'].replace(['interior'],'1')
milsa['Regiao'] = milsa['Regiao'].replace(['outro'],'2')
milsa['Regiao'] = milsa['Regiao'].astype(int)

print(milsa.head())

#milsa["Salario"].hist(bins=15)
#plt.show()

correlation_matrix = milsa.corr()

print(correlation_matrix["Salario"])


X = milsa.drop("Salario", axis=1)
X = X.values
y = milsa["Salario"]
y = y.values



new_data_point = np.array([
    0,
    2,
    1,
    30,
    10,
    0,
])

distances = np.linalg.norm(X - new_data_point, axis=1)

k = 3
nearest_neighbor_ids = distances.argsort()[:k]
nearest_neighbor_salary = y[nearest_neighbor_ids]
prediction = nearest_neighbor_salary.mean()

print("Salário estimado:")

print(prediction)

'''
