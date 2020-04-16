#Common libraries for work with data
import os.path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Plot libraries
import matplotlib.pyplot as plt
import seaborn as sns


#Carga datos iniciales (crea el label)
def crear_label(df):
    #Creo la columna Label que contiene el target de si un user hizo(1) o no(0) churn
    df.loc[(df['Label_max_played_dsi'] == 3), 'Label'] = 1
    df.loc[(df['Label_max_played_dsi'] != 3), 'Label'] = 0
    return df

def eliminar_datos_noConfiables(df):
    indexNames = df[(df['Label'] == 1) & (df['install_date'] >= 383)].index
    df.drop(indexNames , inplace=True)
    df = df.reset_index(drop=True)
    return df

def verificar_balance(df,df_aux):
    dif = np.absolute(df['Label'].mean() - df_aux['Label'].mean())
    b=True
    if dif > 0.05:
        b=False
    return b
    
def extraer_muestra(s,df):
    df_aux = df.sample(frac=s, random_state=42)
    if verificar_balance(df,df_aux) == False:
        print('Desbalance')
    else: print('Balanceado')
    return df_aux

def cargar_datos(sample=1):
    #Si existe el archivo de muestra lo levanta
    if (os.path.isfile('../datos/train_sample.csv')) & (sample<1):
        dataset = pd.read_csv('../datos/train_sample.csv')
    #Si existe el archivo total
    elif (os.path.isfile('../datos/train_entero.csv')) & (sample==1):
        dataset = pd.read_csv('../datos/train_entero.csv')
    #Si no existe levanta a todos y crea una muestra
    else:
        dataset = pd.read_csv('../datos/train_1.csv')
        for i in range(2,6):
            df = pd.read_csv('../datos/train_'+str(i)+'.csv')
            dataset = pd.concat([dataset,df],ignore_index=True).drop_duplicates()
        dataset = dataset.reset_index(drop=True)
        
        #No hace falta cambiar los blanks ("") por Nan porque pandas ya lo interpreta como Nan por defecto
        dataset = crear_label(dataset) #Crea el label de churn
        dataset = eliminar_datos_noConfiables(dataset) # Elimina los datos no confiables (383)
        if sample < 1:
            dataset = extraer_muestra(sample,dataset) #Extrae una muestra
            dataset = dataset.reindex(sorted(dataset.columns), axis=1).reset_index(drop=True) #Reinicia los index
        
            #Crea el sample
            dataset.to_csv(r'../datos/train_sample.csv', index = False)
        else:
            dataset.to_csv(r'../datos/train_entero.csv', index = False)

    # Pregunta por test y train
    """
    if (os.path.isfile('../datos/train_sample_train.csv')) & (os.path.isfile('../datos/train_sample_test.csv')):
        dataset_train = pd.read_csv('../datos/train_sample_train.csv')   
        dataset_test = pd.read_csv('../datos/train_sample_test.csv')   
    else:
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.3, random_state=42)
        dataset_train.to_csv(r'../datos/train_sample_train.csv', index = False)
        dataset_test.to_csv(r'../datos/train_sample_test.csv', index = False)
    """

    dataset['Label'] = dataset['Label'].astype('int64')
    dataset.drop(['Label_max_played_dsi'], axis=1, inplace = True)
    return dataset



