import perceptron as pp
import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv', header=0)

# setosa and versicolor
y = df.iloc[0:100, 5].values
y = np.where(y == 'setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [1,3]].values


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

ppn = pp.Perceptron(epochs=50, eta=0.1)

ppn.train(X, y)
ppn.predict(df.iloc[0:100, [1,3]].values)
print('Weights: %s' % ppn.w_)
plot_decision_regions(X, y, clf=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()
