import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="ImageClassify",
    page_icon="🖼️",
    layout="wide"
)

MODEL_PATH = "modelo_cifar11.keras"
CLASSES_PATH = "class_names_11.json"

IMAGE_SIZE = (32, 32)


# ==========================================================
# INFORMACIÓN DE LAS CLASES
# ==========================================================

CLASS_INFO = {
    "Avión": "✈️",
    "Automóvil": "🚗",
    "Pájaro": "🐦",
    "Gato": "🐱",
    "Ciervo": "🦌",
    "Perro": "🐕",
    "Rana": "🐸",
    "Caballo": "🐎",
    "Barco": "🚢",
    "Camión": "🚚",
    "Persona": "👤"
}


# ==========================================================
# CARGAR MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "No se encontró el archivo modelo_cifar11.keras"
        )

        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ==========================================================
# CARGAR CLASES
# ==========================================================

@st.cache_data
def cargar_clases():

    if not os.path.exists(CLASSES_PATH):

        st.error(
            "No se encontró el archivo class_names_11.json"
        )

        st.stop()

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as archivo:

        return json.load(archivo)


# ==========================================================
# CARGAR RECURSOS
# ==========================================================

modelo = cargar_modelo()
class_names = cargar_clases()


# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown(
    """
    <div style="text-align:center">

    <h1>🖼️ ImageClassify</h1>

    <h3>
    Clasificación de imágenes mediante Machine Learning
    </h3>

    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    Aplicación web desarrollada con Machine Learning capaz
    de clasificar imágenes en diferentes categorías utilizando
    una Red Neuronal Convolucional.
    """
)


# ==========================================================
# AUTOR
# ==========================================================

st.sidebar.title("👤 Información del proyecto")

st.sidebar.write(
    "**Autor:** Ruth Palacios"
)

st.sidebar.write(
    "**Modelo:** CNN"
)

st.sidebar.write(
    "**Dataset:** CIFAR-10 + clase Persona"
)

st.sidebar.write(
    "**Categorías:** 11"
)


# ==========================================================
# SELECCIÓN DE IMAGEN
# ==========================================================

st.header("📷 Seleccionar imagen")

st.write(
    "Puedes cargar una imagen o utilizar la cámara."
)

col1, col2 = st.columns(2)


with col1:

    archivo = st.file_uploader(
        "📁 Subir una imagen",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


with col2:

    foto = st.camera_input(
        "📷 Tomar una fotografía"
    )


# ==========================================================
# SELECCIONAR FUENTE
# ==========================================================

imagen_archivo = None

if archivo is not None:

    imagen_archivo = archivo

elif foto is not None:

    imagen_archivo = foto


# ==========================================================
# PROCESAR IMAGEN
# ==========================================================

if imagen_archivo is not None:

    imagen = Image.open(
        imagen_archivo
    ).convert("RGB")

    st.divider()

    col_imagen, col_resultado = st.columns(
        [1, 1]
    )


    # ======================================================
    # MOSTRAR IMAGEN
    # ======================================================

    with col_imagen:

        st.subheader(
            "🖼️ Imagen seleccionada"
        )

        st.image(
            imagen,
            caption="Imagen proporcionada por el usuario",
            use_container_width=True
        )


    # ======================================================
    # PREPROCESAMIENTO
    # ======================================================

    imagen_procesada = imagen.resize(
        IMAGE_SIZE
    )

    imagen_array = np.array(
        imagen_procesada
    )

    imagen_array = (
        imagen_array.astype("float32")
        / 255.0
    )

    imagen_array = np.expand_dims(
        imagen_array,
        axis=0
    )


    # ======================================================
    # PREDICCIÓN
    # ======================================================

    with st.spinner(
        "🤖 Analizando imagen..."
    ):

        predicciones = modelo.predict(
            imagen_array,
            verbose=0
        )[0]


    indice = int(
        np.argmax(predicciones)
    )

    clase = class_names[indice]

    confianza = float(
        predicciones[indice]
    )


    # ======================================================
    # RESULTADO
    # ======================================================

    with col_resultado:

        st.subheader(
            "🤖 Resultado"
        )

        icono = CLASS_INFO.get(
            clase,
            "🔎"
        )

        st.success(
            f"{icono} Objeto identificado: {clase}"
        )

        st.metric(
            "Porcentaje de confianza",
            f"{confianza * 100:.2f}%"
        )

        st.progress(
            confianza
        )


    # ======================================================
    # PROBABILIDADES
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Probabilidad por categoría"
    )

    resultados = []

    for i, nombre in enumerate(class_names):

        probabilidad = float(
            predicciones[i]
        )

        resultados.append(
            (
                nombre,
                probabilidad
            )
        )


    # Ordenar de mayor a menor
    resultados.sort(
        key=lambda x: x[1],
        reverse=True
    )


    for nombre, probabilidad in resultados:

        icono = CLASS_INFO.get(
            nombre,
            "🔎"
        )

        st.write(
            f"{icono} **{nombre}** — "
            f"{probabilidad * 100:.2f}%"
        )

        st.progress(
            probabilidad
        )


# ==========================================================
# INFORMACIÓN
# ==========================================================

st.divider()

st.subheader(
    "ℹ️ Sobre la aplicación"
)

st.write(
    """
    ImageClassify utiliza una Red Neuronal Convolucional
    entrenada para clasificar imágenes.

    El sistema procesa la imagen proporcionada por el usuario,
    la adapta al tamaño requerido por el modelo y genera una
    predicción junto con su porcentaje de confianza.
    """
)


# ==========================================================
# PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    "ImageClassify | Proyecto académico de Machine Learning"
)
