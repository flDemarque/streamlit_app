import pandas as pd
import streamlit as st
import joblib

# Carregando o modelo treinado
model = joblib.load('modelo_ML.pkl')

# Carregando uma amostra dos dados
dataset = pd.read_csv('StudentsPerformance.csv')

# Defina as colunas esperadas pelo modelo
expected_columns = [
    'gender', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score',
    'race_ethnicity_group_A', 'race_ethnicity_group_B', 'race_ethnicity_group_C', 
    'race_ethnicity_group_D', 'race_ethnicity_group_E', 
    'parental_level_of_education_associates_degree', 'parental_level_of_education_bachelors_degree',
    'parental_level_of_education_high_school', 'parental_level_of_education_masters_degree',
    'parental_level_of_education_some_college', 'parental_level_of_education_some_high_school'
]

# Título
st.title("Data App - Students Performance in Exams")

# Subtítulo
st.markdown("This is a data application used to display the machine learning solution to the problem of students' math grade performance.")

st.sidebar.subheader("Define student attributes for performance prediction:")

# Mapeando dados do usuário para cada atributo
gender = st.sidebar.selectbox("Gender?", ("Male", "Female"))
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity?", ("group_A", "group_B", "group_C", "group_D", "group_E"))
parental_level_of_education = st.sidebar.selectbox("Parental level of education?", ("bachelors_degree", "some_college", "masters_degree", "associates_degree", "high_school", "some_high_school"))
lunch = st.sidebar.selectbox("Lunch?", ("Standard", "Free/Reduced"))
test_preparation_course = st.sidebar.selectbox("Test preparation course?", ("Completed", "None"))
reading_score = st.sidebar.number_input("Reading score", value=dataset["reading_score"].mean())
writing_score = st.sidebar.number_input("Writing score", value=dataset["writing_score"].mean())

# Transformando o dado de entrada em valor binário
gender = 1 if gender == "Male" else 0
lunch = 1 if lunch == "Standard" else 0
test_preparation_course = 1 if test_preparation_course == "Completed" else 0

# Preparando variáveis categóricas
race_ethnicity_dummies = pd.get_dummies([race_ethnicity], prefix='race_ethnicity')
parental_level_of_education_dummies = pd.get_dummies([parental_level_of_education], prefix='parental_level_of_education')

# Inserindo um botão na tela
btn_predict = st.sidebar.button("Make Prediction")

# Verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame({
        "gender": [gender],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    })

    # Adicionando as variáveis categóricas
    data_teste = pd.concat([data_teste, race_ethnicity_dummies, parental_level_of_education_dummies], axis=1)

    # Adicionando colunas ausentes com valor 0
    for col in expected_columns:
        if col not in data_teste.columns:
            data_teste[col] = 0

    # Reordenando as colunas para garantir a correspondência com o modelo
    data_teste = data_teste[expected_columns]

    # Imprimindo os dados de teste
    st.write(data_teste)

    # Realizando a predição
    result = model.predict(data_teste)

    st.subheader("The student's performance in the exam is:")
    result = str(round(float(result[0]), 2))

    st.write(result)
