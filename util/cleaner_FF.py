#Cleaner
import pandas as pd
import numpy as np


class Cleaner:
    def __init__(self):
        #Load the dataset with the inputs correlated values
        self.dataset_values = pd.read_csv('../datos/train_sample_train.csv')

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
    
    def cat_x_otros(self,df,column,threshold):
    
	    c = self.dataset_values.groupby(column).user_id.count().sort_values(ascending=False)
	    c_prop = c / c.sum() #Identifico la concentracion por cagegoria
	    cat_princ = c_prop[c_prop > threshold].index.tolist() #Me quedo con las categorias principales

	    df.loc[~df[column].isin(cat_princ),column]  = "otros" #Reemplazo las categorias no principales por etiqueta "otros"
	        
	    return df

    def tratar_nulos_categoricos(self,df):
        # Crea etiquetas nuevas para las variables categoricas a las que vale la pena hacerlo
        values = {'categorical_7': 'sin_cat7', 'site': 'sin_public'}
        df = df.fillna(value=values)

        # Trata los valores nulos del campo device_model en relacion al campo platform
        df.loc[(df.platform == "iOS") & (df.device_model.isnull()),"device_model"] = 'iPhone7,2' # Modelo mas recurrente para iOS
        df.loc[(df.platform == "Android") & (df.device_model.isnull()),"device_model"] = 'SM-T560' # Modelo mas recurrente para Android

        return df
    
    def tratar_nulos_numericos(self,df):
        df = self.fill_nan_values_corr_1(df, df.StartBattle_sum_dsi1.name, df.LoseBattle_sum_dsi1.name) # Tabla con valores mas cercanos a corr
        df = self.fill_nan_values_corr_1(df, df.OpenChest_sum_dsi2.name, df.EnterDeck_sum_dsi2.name) # Tabla con valores mas cercanos a corr
        df = self.fill_nan_values_corr_2(df, df.StartBattle_sum_dsi1.name, df.LoseBattle_sum_dsi1.name) # Tarda 0.02 segs por row
        df = self.fill_nan_values_corr_2(df, df.OpenChest_sum_dsi2.name, df.EnterDeck_sum_dsi2.name) # Tarda 0.02 segs por row
        df['ChangeArena_sum_dsi3'].fillna(0, inplace = True)
        df['ChangeArena_sum_dsi3'] = df['ChangeArena_sum_dsi3'].astype('int64')
        #df['Label'] = df['Label'].astype('int64')
        

        #Eliminar columnas no necesarias
        df.drop(['age'], axis=1, inplace = True) # Por la cantidad de nulos que tiene y porque no aporta info a Label
        df.drop(['id'], axis=1, inplace = True) # Se elimina porque no aporta ninguna informacion por la naturaleza de la variable
        #df.drop(['Label_max_played_dsi'], axis=1, inplace = True) # Se elimina porque es el target expresado de otra manera
        df.drop(['traffic_type'], axis=1, inplace = True) # Se elimina porque tiene todos los valores iguales (2)
        
        return df

    def cat_a_bucket(self,df,columna):

        c = self.dataset_values.groupby(columna).user_id.count().sort_values(ascending=False)
        c_prop = c / c.sum()
        c_prop = c_prop.cumsum()
        c_prop = c_prop.rename("cumsum")

        df = pd.merge(df, c_prop, left_on=columna, right_index=True, how='left')

        new_name = columna + "_agg"

        df.loc[(df['cumsum'] < 0.2), new_name] = 1
        df.loc[((df['cumsum'] >= 0.2) & (df['cumsum'] < 0.4)), new_name] = 2
        df.loc[((df['cumsum'] >= 0.4) & (df['cumsum'] < 0.6)), new_name] = 3
        df.loc[((df['cumsum'] >= 0.6) & (df['cumsum'] < 0.8)), new_name] = 4
        df.loc[((df['cumsum'] >= 0.8)), new_name] = 5
        df[new_name].fillna(6, inplace=True)
        
        df[new_name] = df[new_name].astype('int64')
        df.drop(['cumsum'], axis=1, inplace = True)
        df.drop([columna], axis=1, inplace = True)
        
        return df

    def tratar_outliers_categoricos(self,df):
        # Cambia las categorias unicas o poco frecuentes por una unica categoria en base a un threshold de concentracion seteado por jucio experto
        df = self.cat_x_otros(df,"categorical_1",0.02)# 4 Categorias en total
        df = self.cat_x_otros(df,"categorical_2",0.01)# 4 Categorias en total
        df = self.cat_x_otros(df,"categorical_3",0.01)# 3 Categorias en total
        df = self.cat_x_otros(df,"categorical_4",0.01)# 2 Categorias en total
        df = self.cat_x_otros(df,"categorical_5",0.005)# 6 Categorias en total
        df = self.cat_x_otros(df,"categorical_6",0.01)# 5 Categorias en total
        df = self.cat_x_otros(df,"categorical_7",0.05)# 10 Categorias en total
        df = self.cat_x_otros(df,"site",0.01)# 2 Categorias en total (tuvo o no tuvo public)

        # buckets de categoricas
        df = self.cat_a_bucket(df,"country")
        df = self.cat_a_bucket(df,"device_model")
        
        #Eliminar columnas no necesarias
        df.drop(['user_id'], axis=1, inplace = True) # Por la cantidad de valores nulos tomados como outliers se elimina la columna
        return df

    def tratar_negativos(self,df,column_neg, column_pos):
        df.loc[df[column_neg] < 0, column_neg] = df.loc[df[column_neg] < 0, column_pos] 
        return df

    def tratar_outliers_numericos(self,df):
    	df = self.tratar_negativos(df,"BuyCard_sum_dsi1", "BuyCard_sum_dsi0")
    	df = self.tratar_negativos(df,"OpenChest_sum_dsi3", "OpenChest_sum_dsi2")
    	return df

    def clean_all(self,df):
        df = self.tratar_nulos_numericos(df)
        df = self.tratar_nulos_categoricos(df)
        df = self.tratar_outliers_categoricos(df)
        df = self.tratar_outliers_numericos(df)

        df = df.reindex(sorted(df.columns), axis=1)
        return df
