import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

numeros = skdata.load_digits()
target = numeros['target'] == 1
imagenes = numeros['images'] 
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

x_train = x_train @ vectores.T
x_test = x_test @ vectores.T

clf = LinearDiscriminantAnalysis()
score_train_true = np.zeros(41)
score_test_true = np.zeros(41)
score_train_false = np.zeros(41)
score_test_false = np.zeros(41)

for i in range(3, 41):
    clf.fit(x_train[:,:i+1], y_train)
    ypred_train = clf.predict(x_train[:,:i+1])
    ypred_test = clf.predict(x_test[:,:i+1])
    score_train_true[i] = f1_score(y_train, ypred_train)
    score_test_true[i] = f1_score(y_test, ypred_test)
    score_train_false[i] = f1_score(y_train, ypred_train, pos_label = False)
    score_test_false[i] = f1_score(y_test, ypred_test, pos_label = False)

plt.figure(figsize=(7,7))

plt.subplot(1,2,1)
plt.scatter(range(0,41), score_train_true, label='train 50%')
plt.scatter(range(0,41), score_test_true, label='test 50%')
plt.title('Clasificación UNO')
plt.xlabel('Número de componentes PCA')
plt.ylabel('F1 score')
plt.xlim(2,41)
plt.legend()
#plt.show()

plt.subplot(1,2,2)
plt.scatter(range(0,41), score_train_false, label='train 50%')
plt.scatter(range(0,41), score_test_false, label='test 50%')
plt.title('Clasificación OTROS')
plt.xlabel('Número de componentes PCA')
plt.ylabel('F1 score')
plt.xlim(2,41)
plt.ylim(0.9,1)
#plt.legend()
plt.savefig('F1_score_LinearDiscriminantAnalysis.png')
plt.show()