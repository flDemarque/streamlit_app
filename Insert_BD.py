import sqlite3
import random
from datetime import datetime, date
import os
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
import sqlite3
import random
from datetime import date
import os


default_args = {'owner': 'airflow'}

dag = DAG(dag_id='insert_bd',  default_args=default_args, schedule_interval='@daily', start_date=days_ago(2))

def executar_script():
    # Configurações
    db_path = '/home/aluno/airflow/dags/StudentsPerformance.db'
    log_file = 'last_run.log'
    num_records = 10

    # Função para verificar se o script já foi executado hoje
    def already_ran_today(log_file):
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                last_run = f.read().strip()
            if last_run == str(date.today()):
                return True
        return False

    # Função para registrar a data de execução do script
    def update_log(log_file):
        with open(log_file, 'w') as f:
            f.write(str(date.today()))

    # Se o script já foi executado hoje, termina o programa
    if already_ran_today(log_file):
        print("O script já foi executado hoje. Saindo.")
        return

    # Conectar ao banco de dados
    conexao = sqlite3.connect(db_path)
    cursor = conexao.cursor()

    # Dados possíveis para as colunas
    genders = ["Male", "Female"]
    races = ["group_A", "group_B", "group_C", "group_D", "group_E"]
    parental_education = ["bachelors_degree", "some_college", "masters_degree", "associates_degree", "high_school", "some_high_school"]
    lunch_options = ["Standard", "Free/Reduced"]
    prep_courses = ["Completed", "None"]

    # Gerar 10 registros aleatórios
    for _ in range(num_records):
        gender = random.choice(genders)
        race_ethnicity = random.choice(races)
        parental_level_of_education = random.choice(parental_education)
        lunch = random.choice(lunch_options)
        test_preparation_course = random.choice(prep_courses)
        math_score = random.randint(0, 100)
        reading_score = random.randint(0, 100)
        writing_score = random.randint(0, 100)

        # Inserir no banco de dados
        cursor.execute("""
            INSERT INTO students (gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score))

    # Salvar as mudanças e fechar a conexão
    conexao.commit()
    conexao.close()

    # Atualizar o log
    update_log(log_file)


exe = PythonOperator(task_id="sdsd", python_callable=executar_script, dag=dag)