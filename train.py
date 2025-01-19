import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data_dict=pickle.load(open("./data.pickle", "rb"))

data=np.asarray(data_dict["data"])
labels=np.asarray(data_dict["labels"])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model_rfc=RandomForestClassifier()
model_rfc.fit(x_train, y_train)
y_pred_rfc=model_rfc.predict(x_test)
score_rfc=accuracy_score(y_pred_rfc, y_test)
print("Accuracy score of Random Forest Classifier:{}%".format(score_rfc*100))

model_knn=KNeighborsClassifier(n_neighbors=4)
model_knn.fit(x_train, y_train)
y_pred_knn=model_knn.predict(x_test)
score_knn=accuracy_score(y_pred_knn, y_test)
print("Accuracy score of K Nearest Neighbors:{}%".format(score_knn*100))

model_dc=DecisionTreeClassifier(random_state=1)
model_dc.fit(x_train, y_train)
y_pred_dc=model_dc.predict(x_test)
score_dc=accuracy_score(y_pred_dc, y_test)
print("Accuracy score of Decision Tree Classifier:{}%".format(score_dc*100))

f=open("model.p", "wb")
pickle.dump({"model":model_rfc},f)
f.close()