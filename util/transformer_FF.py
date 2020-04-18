#Cleaner
import pandas as pd
import numpy as np


class Transformer:
    def __init__(self):
        #Load the dataset with the inputs correlated values
        self.dataset_values = pd.read_csv('../datos/train_sample.csv')
    
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

    def crea_dia_semana(self,df):
        # Arma una tabla base con muchas semanas para 2000 dias
        d = {'dia': [], 'dia_semana': []}
        df_week_table = pd.DataFrame(data=d)

        dia_sem = 0
        for i in range(0,2000):
            if dia_sem==7:
                dia_sem=0
            dia_sem += 1
            df_week_table = df_week_table.append({'dia': i, 'dia_semana': dia_sem}, ignore_index=True)
        
        # Joinea con install_date para determinar que dia se intalo el juego
        df = df.merge(df_week_table, right_on="dia", left_on="install_date", how='left')  
        df.drop(['dia'], axis=1, inplace = True) 
        return df

    def crea_mes(self,df):
        # Arma una tabla base con muchos meses para 2000 dias
        d = {'dia': [], 'mes': []}
        df_mes_table = pd.DataFrame(data=d)

        mes = 1
        dias_mes = 0
        for i in range(0,2000):
            if mes==12:
                mes=1
            if dias_mes==30:
                dias_mes=0
                mes+=1
            dias_mes += 1
            df_mes_table = df_mes_table.append({'dia': i, 'mes':mes}, ignore_index=True)

        # Joinea con install_date para determinar que mes se intalo el juego
        df = df.merge(df_mes_table, right_on="dia", left_on="install_date", how='left')  
        df.drop(['dia'], axis=1, inplace = True) 
        return df

    def dummies_for_dsi(self,df):
        names_dsiX = [col for col in df.columns if '_dsi3' in col] # Se lo aplicamos solo a las dsi3
        for i in names_dsiX:
            q1_0s = self.dataset_values[self.dataset_values.Label==0][i].quantile(0.25)
            df[i+"_dum"] = 0
            df.loc[(df[i] <= q1_0s),i+"_dum"]=1
            #df.drop(i, axis=1, inplace=True)
        return df

    def transform_all(self, df, df_train = None):
        # Crea, transforma y selecciona features
        df = self.crea_dia_semana(df)
        df = self.crea_mes(df)
        df = self.dummies_for_dsi(df)

        # Elimina columnas reciduales
        df.drop(['install_date'], axis=1, inplace = True) 
        
        # Esto siempre tiene que ser el ultimo
        df = self.crear_dummies(df,df_train)


        # Ordena las columnas (muchos modelos lo exigen)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
