#Cleaner
import pandas as pd
import numpy as np


class Cleaner:
    def __init__(self):
        #Load the dataset with the inputs correlated values
        self.dataset_values = pd.read_csv('../datos/train_sample.csv')

    def calculate_corr_values_corr(self, col_null, col_corr, n):
        nearest_n = self.dataset_values[col_corr].iloc[(self.dataset_values[col_corr]-n).abs().argsort()[:1]].values[0]
        a = self.dataset_values[self.dataset_values[col_corr] == nearest_n][col_null].mean() # Mean of a similar high correlated value
        try:
            a = int(a) # Transform it to an integer
        except:
            a = self.dataset_values[col_null].mean() # If not a value abailable take gral mean
            a = int(a)
        return a

    def fill_nan_values_corr_2(self, df, col_null, col_corr):
        # Lambda functions thats apply only if the condition es fulfilled
        df['outlier'] = df[df[col_null].isnull()].apply(lambda x: self.calculate_corr_values_corr(col_null, col_corr, x[col_corr]), axis=1)

        df[col_null].fillna(df['outlier'], inplace=True)
        df.drop(['outlier'], axis=1, inplace=True)
        df[col_null] = df[col_null].astype('int64')
        return df
    
    def fill_nan_values_corr_1(self, df, col_null, col_corr):
        # Crea una tabla con los valores mas cercanos
        df_aux = df.copy()
        df_aux.dropna(subset=[col_null], inplace=True)
        tabla_nulos_temp = df_aux[[col_null,col_corr]].groupby(col_corr, as_index = False).mean().astype(int)
        
        # Los junta con el dataset original 
        df = pd.merge(df, tabla_nulos_temp, how='left', on=[col_corr], suffixes=('', '_Nulls'))
        
        # Completa los nulls de la columna original con la auxiliar y luego la elimina
        df[col_null].fillna(df[col_null+'_Nulls'], inplace=True)
        df.drop([col_null+'_Nulls'], axis=1, inplace=True)
        return df
    
    def tratar_nulos_categoricos(self,df):
        # Crea etiquetas nuevas para las variables categoricas a las que vale la pena hacerlo
        values = {'categorical_7': 'sin_cat7', 'country': 'sin_country', 'site': 'sin_public'}
        df = df.fillna(value=values)

        # Elimina nulls considerados outliers
        df.dropna(subset=['device_model'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        #Eliminar columnas no necesarias
        df.drop(['user_id'], axis=1, inplace = True)
        return df
    
    def tratar_nulos_numericos(self,df):
        df = self.fill_nan_values_corr_1(df, df.StartBattle_sum_dsi1.name, df.LoseBattle_sum_dsi1.name) # Tabla con valores mas cercanos a corr
        df = self.fill_nan_values_corr_1(df, df.OpenChest_sum_dsi2.name, df.EnterDeck_sum_dsi2.name) # Tabla con valores mas cercanos a corr
        df = self.fill_nan_values_corr_2(df, df.StartBattle_sum_dsi1.name, df.LoseBattle_sum_dsi1.name) # Tarda 0.02 segs por row
        df = self.fill_nan_values_corr_2(df, df.OpenChest_sum_dsi2.name, df.EnterDeck_sum_dsi2.name) # Tarda 0.02 segs por row
        df['ChangeArena_sum_dsi3'].fillna(0, inplace = True)
        df['ChangeArena_sum_dsi3'] = df['ChangeArena_sum_dsi3'].astype('int64')
        df['Label'] = df['Label'].astype('int64')
        df.drop(['age'], axis=1, inplace = True)

        #Eliminar columnas no necesarias
        df.drop(['id'], axis=1, inplace = True)
        df.drop(['Label_max_played_dsi'], axis=1, inplace = True)
        return df
    
    def clean_all(self,df):
        df = self.tratar_nulos_numericos(df)
        df = self.tratar_nulos_categoricos(df)
        return df
