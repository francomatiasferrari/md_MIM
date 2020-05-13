#Cleaner
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class Modeler:
# Funciones de entrenamiento y tunning
    def tunner_grid(self, candidate, param_grid, train_features, train_labels, n_folds):
        # Create a based model
        model = candidate
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = model, # Model to fit
                                    param_grid = param_grid, # The hyperparameters to tune 
                                    cv = n_folds, # Number of folds to the cross-validation training
                                    verbose=2,
                                    random_state=42,
                                    n_jobs = -1) # Number of cores to use. (-1 indicates all of them)
        # Fit the grid search to the data
        grid_search.fit(train_features, train_labels) 
        return grid_search

    def tunner_random(self, candidate, random_grid, train_features, train_labels, n_folds, n_iter):
        model = candidate

        rf_random = RandomizedSearchCV(estimator = model, 
                                        param_distributions = random_grid, 
                                        n_iter = n_iter, 
                                        cv = n_folds, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
        
        rf_random.fit(X = train_features, y= train_labels)

        return rf_random

# Funciones de metricas y resultados

    def plot_roc(self,model,X,y):
        # Arma el grafico de la curva ROC de una sola linea
        probs = model.predict_proba(X)

        fpr, tpr, threshold = metrics.roc_curve(y,probs[:,1])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Model Performance')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    def plot_roc_2(self,model,X_train,y_train,X_test,y_test):
        # Arma el grafico de la curva ROC de dos lineas
        # Sirve para comparar test y train en un solo grafico
        probs = model.predict_proba(X_train)
        probs_test = model.predict_proba(X_test)

        fpr, tpr, threshold = metrics.roc_curve(y_train,probs[:,1])
        fpr_test, tpr_test, threshold = metrics.roc_curve(y_test,probs_test[:,1])

        roc_auc = metrics.auc(fpr, tpr)
        roc_auc_test = metrics.auc(fpr_test, tpr_test)

        plt.title('Model Performance')
        plt.plot(fpr, tpr, 'b', label = 'AUC train = %0.5f' % roc_auc)
        plt.plot(fpr_test, tpr_test, 'g', label = 'AUC test = %0.5f' % roc_auc_test)

        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_feature_importance(self,model,X):
        # Funcion de sklearn que nos da la importancia de cada variable
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])

        #Ploteamos un grafico de barras que indica en orden la importancia de las variables.
        plt.figure(figsize=(15, 25))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('Features importance')
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self,model, X, y): # Function to evaluate the Confusion Matrix
        predictions = model.predict(X) # Prediction of the model
        confMat = metrics.confusion_matrix(y, predictions) # Calculate the metrics comparing real vs predict
        print(confMat)

# Funciones de guardado de resultados 
    def guardar_modelo(self,model,nombreModel):
        # Guarda el modelo en un pickle
        with open('../pickles/'+final_model+'.pkl', 'wb') as handle:
            pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)

    def abrir_modelo(self,nombreModel):
        # Abre el modelo de un pickle
        with open('../pickles/'+final_model+'.pkl', 'rb') as handle:
            model = pickle.load(handle)
        return model

    def guardar_submit(self,probs,nombreArchivo):
        # Guarda los resultados de scorear validation en un archivo para el submit de la competencia.
        p = pd.DataFrame(probs[:,1:])
        p = p.rename(columns={0: "Label"})
        p.insert(0, 'id', p.index)
        p.to_csv(r'../resultados/'+nombreArchivo+'.csv', index = False)


