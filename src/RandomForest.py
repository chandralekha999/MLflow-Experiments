import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#Loading wine dataset
wine = load_wine()
x = wine.data
y = wine.target

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
# Random Forest 
max_depth = 5
n_estimators = 10
print(n_estimators)

mlflow.set_experiment("Random_Forest")
with mlflow.start_run():
    #Training
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)
    #Prediction
    y_pred = rf.predict(x_test)

    #Evaluation
    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion-matrix1.png')

    #mlflow metrics
    mlflow.log_metric('accuracy is ', accuracy)
    mlflow.log_param('parms max_depth is', max_depth)
    mlflow.log_param('param n_estimators is',n_estimators)
    mlflow.log_artifact("confusion-matrix1.png")
    mlflow.log_artifact(__file__)


    print(accuracy)
