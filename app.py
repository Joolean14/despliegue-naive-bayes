import pandas as pd
import pickle
import streamlit as st

# Cargar el modelo entrenado
filename = 'naive_bayes_model.pkl'
nb, le, variables = pickle.load(open(filename, 'rb'))

# Interfaz Streamlit
st.title('Predicción de pago oportuno en 45 días')

tipo_cliente = st.selectbox('Tipo cliente', ["COMBUSTIBLE","TRANSPORTE", "MATERIALES"])
vencimiento_dia_habil = st.selectbox('Vencimiento día hábil', ["SI", "NO"])

# DataFrame con los inputs
datos = [[tipo_cliente, vencimiento_dia_habil]]
data = pd.DataFrame(datos, columns=['Tipo_cliente', 'Vencimiento_dia_habil'])

# Preparación: dummies
data_preparada = pd.get_dummies(
    data,
    columns=['Tipo_cliente', 'Vencimiento_dia_habil'],
    drop_first=False,
    dtype=int
)

# Reindexar para que tenga las mismas columnas que en entrenamiento
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# Predicción
Y_pred = nb.predict(data_preparada)

# Agregar predicción al DataFrame original
data['Prediccion'] = le.inverse_transform(Y_pred)
st.write(data)
