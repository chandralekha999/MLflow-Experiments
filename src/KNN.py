import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot  as plt
import seaborn as sns

#Loading wine dataset
wine = load_wine()
x = wine.data
y = wine.target

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=42)
mlflow.set_experiment("KNN")
with mlflow.start_run():
    #Apply KNN
    n_neighbors = 5
    KNN= KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN.fit(x_train,y_train)
    #Pred
    y_pred = KNN.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion-matrix2.png')
    #mlflow metrics
    mlflow.log_param("n_neighbours are", n_neighbors)
    mlflow.log_metric("accuracy is", accuracy)
    mlflow.log_artifact("confusion-matrix2.png")
    mlflow.log_artifact(__file__)


    print(accuracy)





