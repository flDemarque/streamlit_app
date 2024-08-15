from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib


default_args = {'owner': 'airflow'}

path = "/home/aluno/airflow/dags"  #Linux

path_db_producao = path + "/StudentsPerformance.db"

path_temp_csv = path + "/StudentsPerformance.csv"


dag = DAG(dag_id='data_pipeline',  default_args=default_args, schedule_interval='@daily', start_date=days_ago(2),)


def verificar_registros_e_exportar_csv():
    selectDb = """SELECT * FROM students"""
    
    # Conectando ao banco de dados
    conexao = sqlite3.connect(path_db_producao)
    
    # Lendo os dados do banco de dados
    df = pd.read_sql(selectDb, con=conexao)
    
    # Obtendo a quantidade de registros
    record_count = len(df)
    
    # Verificando se a quantidade de registros é um múltiplo de 1000
    if record_count % 1000 == 0 and record_count != 0:
        # Convertendo para CSV se a condição for atendida
        df.to_csv(path+"/StudentsPerformance.csv",  index=False)
    
    # Fechar a conexão com o banco de dados
    conexao.close()

def treinar_modelo():
    df = pd.read_csv(path_temp_csv)
    
    # Exibindo informações sobre o DataFrame
    df.info()
    
    # Exibir valores únicos de algumas colunas
    print(df['gender'].unique())
    print(df['race_ethnicity'].unique())
    print(df['parental_level_of_education'].unique())
    print(df['lunch'].unique())
    print(df['test_preparation_course'].unique())
    
    # Eliminar outliers
    def eliminar_outlier_por_quartil(df, colname):
        q1 = df[colname].quantile(q=0.25)
        q3 = df[colname].quantile(q=0.75)
        FIQ = q3 - q1
        Faixa_q1 = q1 - 1.5 * FIQ
        Faixa_q3 = q3 + 1.5 * FIQ
        list_index_outlier =  []
        for i in range(len(df)):
            if ((df.iloc[i][colname] < Faixa_q1) or (df.iloc[i][colname] > Faixa_q3)):
                list_index_outlier.append(i)
        df = df.drop(df.index[list_index_outlier])
        print('Index of outliers: ', list_index_outlier)
        return df

    for col in df.select_dtypes(include=["number"]):
        df = eliminar_outlier_por_quartil(df, col)
    
    # Tratar dados categóricos com get_dummies
    for col in df.columns:
        if df[col].dtype == 'object' and len(list(df[col].unique())) > 3:
            df = pd.get_dummies(df, columns=[col], dtype=np.int64)
    
    # Tratar dados categóricos com cat.codes
    for col in df.columns:
        if df[col].dtype == 'object' and len(list(df[col].unique())) <= 2:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    
    # Exibir informações do DataFrame após tratamento
    df.info()
    
    # Separar as features e a variável alvo
    X = df.drop(['math_score'], axis = 1)
    y = df[['math_score']]
    
    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    # Treinar modelo de Regressão Linear
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Fazer previsões e calcular métricas de desempenho
    predicoes_teste = modelo.predict(X_test)
    r2 = r2_score(y_test, predicoes_teste)
    rmse = np.sqrt(mean_squared_error(y_test, predicoes_teste))
    
    print("R²:", r2)
    print("RMSE:", rmse)
    
    # Salvar o modelo treinado
    joblib.dump(modelo, path + '/modelo_ML.pkl')

#nao vao funcionar estes comandos por causa do airflow que nao funciona no windows

verificar_registros_e_exportar_csv = PythonOperator(task_id="verificar_registros_e_exportar_csv", python_callable=verificar_registros_e_exportar_csv, dag=dag)

treinar_modelo = PythonOperator(task_id="treinar_modelo", python_callable=treinar_modelo, dag=dag)


#ETL 
verificar_registros_e_exportar_csv >> treinar_modelo