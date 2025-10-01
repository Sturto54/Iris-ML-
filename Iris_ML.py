import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#load dataset

iris_dataset = load_iris()

#convert to pandas dataframe
df = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names)
#add in column 'target'
df['target'] = iris_dataset.target

x = df[iris_dataset.feature_names]
y = df['target']
#split an test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

#train a model
model = RandomForestClassifier(random_state = 42)
model.fit(x_train,y_train)

#make prediction
y_pred = model.predict(x_test)

#print the accuracy and stuff
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test,y_pred, target_names= iris_dataset.target_names))