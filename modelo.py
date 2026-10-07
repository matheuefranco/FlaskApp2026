
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd

csv_url = 'https://docs.google.com/spreadsheets/d/1mZZMHEtyg8aiQT2fEBO_QcGhIgpKifqvz8wkEz3-kng/export?format=csv'
df = pd.read_csv(csv_url)
#print(df)

variedade_replace = {'Setosa': 0, 'Versicolor': 1, 'Virginica':2}
df['especie_cod'] = df['especie'].map(variedade_replace)
iris_list = ['Setosa', 'Versicolor', 'Virginica']
#print(iris_list[0])
#print(df)

X = df.drop(columns=['especie', 'especie_cod'])
y =  df['especie_cod']
#print(X)
#print(y)

#split de dados
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=50)

# Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

import pickle
pickle.dump(clf, open('model.pkl','wb')) # salvando o modelo
pickle.dump(iris_list,open('names.pkl','wb')) # salvando os identificadores de saida

# Load the model and the names list
loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_names = pickle.load(open('names.pkl', 'rb'))


test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(test_input)
predicted_class_name = loaded_names[prediction[0]]
print("Predicted class:", predicted_class_name)