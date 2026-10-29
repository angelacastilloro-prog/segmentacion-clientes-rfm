import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentación de clientes RFM", page_icon="🛒", layout="centered")


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_rfm.pkl")


def clasificar(modelo, recencia, frecuencia, monetario):
    """Aplica el mismo preprocesamiento del notebook: log1p + StandardScaler + K-Means."""
    entrada = pd.DataFrame([[recencia, frecuencia, monetario]], columns=modelo["columnas"])
    escalada = modelo["scaler"].transform(np.log1p(entrada))
    cluster = int(modelo["kmeans"].predict(escalada)[0])
    return modelo["mapa_nombres"][cluster]


RECOMENDACIONES = {
    "Campeones": "Compran seguido y recientemente. Ofrecer programa de recompensas y acceso anticipado a productos.",
    "Clientes leales": "Compran con buena frecuencia. Incentivar ventas cruzadas y aumentar el valor de cada pedido.",
    "Nuevos / recientes": "Compraron hace poco, pero pocas veces. Enviar ofertas de segunda compra para convertirlos en clientes leales.",
    "Perdidos / inactivos": "Poca actividad y mucho tiempo sin comprar. Campañas de bajo costo o dejar de invertir en ellos.",
}

st.title("🛒 Segmentación de clientes (RFM)")
st.write(
    "Modelo de Machine Learning clásico (K-Means) entrenado con el dataset **UCI Online Retail**. "
    "Ingresa el comportamiento de un cliente y la app indica a qué segmento pertenece."
)

try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró `modelo_rfm.pkl`. Ponlo en la misma carpeta que `app.py`.")
    st.stop()

tab1, tab2 = st.tabs(["🔎 Clasificar un cliente", "📊 Resumen de segmentos"])

with tab1:
    c1, c2, c3 = st.columns(3)
    recencia = c1.number_input("Recencia (días desde la última compra)", min_value=0, max_value=1000, value=30)
    frecuencia = c2.number_input("Frecuencia (n.º de compras)", min_value=1, max_value=1000, value=5)
    monetario = c3.number_input("Gasto total (£)", min_value=0.0, max_value=1_000_000.0, value=1000.0, step=50.0)

    if st.button("Clasificar cliente", type="primary"):
        segmento = clasificar(modelo, recencia, frecuencia, monetario)
        st.success(f"Segmento: **{segmento}**")
        st.info(RECOMENDACIONES.get(segmento, "Segmento sin recomendación definida."))

with tab2:
    resumen = pd.DataFrame(modelo["resumen"]).set_index("Segmento")
    st.dataframe(resumen)
    st.bar_chart(resumen["Gasto_medio"])
    st.caption("Valores medios calculados con los datos de entrenamiento.")
