#Cleaner
import pandas as pd
import numpy as np

class Transformer:
    def __init__(self):
        #Load the dataset with the inputs correlated values
        self.dataset_values = pd.read_csv('../datos/train_entero.csv')
    
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

    def device_to_brand(self,df):
        # modelos android
        df.loc[(
        (df['device_model'].str.contains('SM-', regex=False)==True)    |
        (df['device_model'].str.contains('Redmi', regex=False)==True)  |
        (df['device_model'].str.contains('Moto', regex=False)==True)   |
        (df['device_model'].str.contains('HUAWEI', regex=False)==True) |
        (df['device_model'].str.contains('Lenovo', regex=False)==True) |
        (df['device_model'].str.contains('LG', regex=False)==True)
        ) &
        (df['platform'] == 'Android'), 'cat_device'] = 'Android_marcas'
        df.loc[(df['device_model'].str.contains('SM-', regex=False)==False) &
        (df['device_model'].str.contains('Redmi', regex=False)==False) &
        (df['device_model'].str.contains('Moto', regex=False)==False) &
        (df['device_model'].str.contains('HUAWEI', regex=False)==False) &
        (df['device_model'].str.contains('Lenovo', regex=False)==False) &
        (df['device_model'].str.contains('LG', regex=False)==False) & 
        (df['platform'] == 'Android'), 'cat_device'] = 'Android_resto'

        #ipad vs iphone
        df.loc[(df['device_model'].str.contains('iPad', regex=False)==True) 
                & (df['platform']=='iOS'), 'cat_device'] = 'iPad'

        df.loc[(df['device_model'].str.contains('iPad', regex=False)==False) 
                & (df['platform']=='iOS'), 'cat_device'] = 'iPhone'

        #peores categorias de android segun churn
        self.dataset_values.loc[(self.dataset_values['device_model'].str.contains('SM-', regex=False)==False) &
        (self.dataset_values['device_model'].str.contains('Redmi', regex=False)==False) &
        (self.dataset_values['device_model'].str.contains('Moto', regex=False)==False) &
        (self.dataset_values['device_model'].str.contains('HUAWEI', regex=False)==False) &
        (self.dataset_values['device_model'].str.contains('Lenovo', regex=False)==False) &
        (self.dataset_values['device_model'].str.contains('LG', regex=False)==False) & 
        (self.dataset_values['platform'] == 'Android'), 'cat_device'] = 'Android_resto'

        c = self.dataset_values.loc[self.dataset_values.cat_device == 'Android_resto', :].groupby('device_model')
        c2_prop = pd.DataFrame(c.user_id.count()/ c.user_id.count().sum())#Identifico la concentracion por cagegoria
        c2_prop['churn'] = c[['Label']].mean()
        c2_prop = c2_prop.reset_index()
        android_mejores = c2_prop.loc[c2_prop.churn <= 0.21,'device_model'].tolist()
        android_peores = c2_prop.loc[c2_prop.churn >=  0.21,'device_model'].tolist()

        # asigna la etiqueta al dataset que esta siendo transformado
        df.loc[df.device_model.isin(android_mejores),'cat_device'] = 'Android_mejores'
        df.loc[df.device_model.isin(android_peores),'cat_device'] = 'Android_peores'

        return df

    def device_by_churn(self,df):
        #obtenemos churn de cada modelo
        c = self.dataset_values.groupby(['platform','device_model'])
        c_prop = pd.DataFrame(c.user_id.count()/ c.user_id.count().sum())#Identifico la concentracion por cagegoria
        c_prop['churn'] = c[['Label']].mean()
        c_prop = c_prop.reset_index()

        #identificamos los models de cada categoria
        bucket1 = c_prop.loc[(c_prop.churn >= 0.054) & (c_prop.churn < 0.16) &(c_prop.user_id>0.0002),'device_model'].tolist()
        bucket2 = c_prop.loc[(c_prop.churn >= 0.16)  & (c_prop.churn < 0.19) &(c_prop.user_id>0.0002),'device_model'].tolist()
        bucket3 = c_prop.loc[(c_prop.churn >= 0.19)  & (c_prop.churn < 0.21) &(c_prop.user_id>0.0002),'device_model'].tolist()
        bucket4 = c_prop.loc[(c_prop.churn >= 0.21)  & (c_prop.user_id>0.0002),'device_model'].tolist()

        #cramos categorias de bucket
        df.loc[df.device_model.isin(bucket1),'cat_device2'] = 'bucket1'
        df.loc[df.device_model.isin(bucket2),'cat_device2'] = 'bucket2'
        df.loc[df.device_model.isin(bucket3),'cat_device2'] = 'bucket3'
        df.loc[df.device_model.isin(bucket4),'cat_device2'] = 'bucket4'
        df.loc[df.cat_device2.isna(),'cat_device2'] = 'bucket5'
        return df

    def country_new_bucket(self,df):
        #obtenemos churn de cada pais
        c = self.dataset_values.groupby(['country'])
        c_prop = pd.DataFrame(c.user_id.count()/ c.user_id.count().sum())#Identifico la concentracion por cagegoria
        c_prop['churn'] = c[['Label']].mean()
        c_prop = c_prop.reset_index().sort_values(by='churn', ascending = False)

        #identificamos los models de cada grupo de churn
        bucket1 = c_prop.loc[(c_prop['churn'] >= 0.11) & (c_prop['churn'] < 0.16) &(c_prop['user_id']>0.0002),'country'].tolist()
        bucket2 = c_prop.loc[(c_prop['churn'] >= 0.16)  & (c_prop['churn'] < 0.19) &(c_prop['user_id']>0.0002),'country'].tolist()
        bucket3 = c_prop.loc[(c_prop['churn'] >= 0.19)  & (c_prop['churn'] < 0.21) &(c_prop['user_id']>0.0002),'country'].tolist()
        bucket4 = c_prop.loc[(c_prop['churn'] >= 0.21)  & (c_prop['user_id']>0.0002),'country'].tolist()

        #cramos categorias de bucket
        df.loc[df['country'].isin(bucket1),'cat_country'] = 'bucket1'
        df.loc[df['country'].isin(bucket2),'cat_country'] = 'bucket2'
        df.loc[df['country'].isin(bucket3),'cat_country'] = 'bucket3'
        df.loc[df['country'].isin(bucket4),'cat_country'] = 'bucket4'
        df.loc[df['cat_country'].isna(),'cat_country'] = 'bucket5'
        return df

    def transform_all(self, df, df_train = None):
        # Crea, transforma y selecciona features
        df = self.crea_dia_semana(df)
        #df = self.crea_mes(df)        # Descubrimos que el mes resta en las metricas de roc porque los modelos le dan mucha importancia por lo que la eliminamos.
        df = self.dummies_for_dsi(df)
        df = self.device_to_brand(df)
        df = self.device_by_churn(df)
        df = self.country_new_bucket(df)

        # Elimina columnas reciduales
        df.drop(['install_date'], axis=1, inplace = True) 
        df.drop(['user_id'], axis=1, inplace = True)
        df.drop(['country'], axis=1, inplace = True)
        df.drop(['device_model'], axis=1, inplace = True)

        # Esto siempre tiene que ser el ultimo
        df = self.crear_dummies(df,df_train)

        # Ordena las columnas (muchos modelos lo exigen)
        df = df.reindex(sorted(df.columns), axis=1)
        return df
