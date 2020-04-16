#Cleaner
import pandas as pd
import numpy as np


class Transformer:
    def __init__(self):
        #Load the dataset with the inputs correlated values
        self.dataset_values = pd.read_csv('../datos/train_sample_train.csv')
    
    def check_dummies(self,df,df_train):
        # Busco las columnas diferentes entre datasets
        l1 = list(set(df_train.columns) - set(df.columns))

        if("Label" in l1): # Esto me da la posibilidad de pasar por parametro df o X_train
            l1.remove("Label")

        # Busco columnas que se hayan creado de mas en validacion
        l2 = list(set(df.columns) - set(df_train.columns))

        # Completo columnas faltantes con 0s
        for i in l1:
            df[i] = 0

        # Elimino columnas de mas
        df.drop(l2, axis=1, inplace=True)

        return df

    def crear_dummies(self,df,df_train):
        # Crea dummies de todas las variables categoricas o booleanas
        variables = df.select_dtypes(include = ['O','bool']).columns
        df= pd.get_dummies(df, columns=variables, drop_first=True) #Creo dummies, elimino una de las categorias para no redundar
        df= df.loc[:,~df.columns.isin(variables)] #Elimino del dataset las variables a las que les cree las dummies
        
        # Chequea que sean las mismas que con las que se entrenó el modelo y sino corrige
        if (df_train is not None):
            df= self.check_dummies(df,df_train)

        return df

    def transform_all(self, df, df_train = None):
        df = self.crear_dummies(df,df_train)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
