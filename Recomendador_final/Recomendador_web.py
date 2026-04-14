import streamlit as st
import pandas as pd
import time
import Recomendador as motor

# Configuración de página
st.set_page_config(page_title="Recomendador Renzo", layout="wide")

st.title("Sistema de Recomendación de Películas")
st.markdown("---")

# ==============================
# CACHE DE DATOS
# ==============================
@st.cache_data
def cargar_datos_cacheados():
    return motor.cargar_datos()

# ==============================
# SESSION STATE
# ==============================
if "cargado" not in st.session_state:
    st.session_state.cargado = False

if "tiempo_carga_mostrado" not in st.session_state:
    st.session_state.tiempo_carga_mostrado = False

if "peliculas_nuevo_usuario" not in st.session_state:
    st.session_state.peliculas_nuevo_usuario = []

# ==============================
# CARGA CONTROLADA
# ==============================
inicio_carga = time.perf_counter()

if not st.session_state.cargado:
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        status_text.text(f"Cargando dataset... {i+1}%")

    matriz, movies = cargar_datos_cacheados()

    progress_bar.empty()
    status_text.empty()

    st.session_state.cargado = True
else:
    matriz, movies = cargar_datos_cacheados()

fin_carga = time.perf_counter()
tiempo_carga = fin_carga - inicio_carga

# ==============================
# MENSAJE TEMPORAL DE CARGA
# ==============================
mensaje_placeholder = st.empty()

if not st.session_state.tiempo_carga_mostrado:
    mensaje_placeholder.success("Dataset cargado correctamente")
    mensaje_placeholder.info(f"Tiempo total de carga: {tiempo_carga:.4f} segundos")

    # Espera 3 segundos y desaparece
    time.sleep(3)
    mensaje_placeholder.empty()

    st.session_state.tiempo_carga_mostrado = True

# ==============================
# CONTENIDO PRINCIPAL
# ==============================
if matriz is not None:

    st.subheader("Información del Dataset")
    st.write(f"Usuarios: {matriz.shape[0]}")
    st.write(f"Películas: {matriz.shape[1]}")
    st.write(f"Calificaciones totales: {matriz.count().sum()}")

    # ==============================
    # SIDEBAR ORGANIZADO
    # ==============================
    st.sidebar.header(" Recomendaciones")
    uid = st.sidebar.number_input("ID de Usuario", 1, int(matriz.index.max()), 2)
    k_vecinos = st.sidebar.slider("Vecinos (K)", 5, 300, 50)
    soporte = st.sidebar.slider("Soporte mínimo (Votos)", 1, 50, 5)

    st.sidebar.header("Vecinos (Manhattan)")
    min_comunes = st.sidebar.slider("Películas mínimas en común", 1, 20, 3)

    col1, col2 = st.columns([2, 1])

    # ==============================
    # FUNCIONES PRINCIPALES
    # ==============================
    with col1:
        st.subheader("Opciones del sistema")

        # RECOMENDACIONES
        if st.button("Generar Recomendaciones"):
            inicio = time.perf_counter()

            recos, media = motor.obtener_recomendaciones(
                matriz, movies, uid, k_vecinos, "pearson", soporte
            )

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.success(f"Promedio del usuario {uid}: **{round(media, 2)}**")
            st.info(f"Tiempo de ejecución: **{tiempo:.6f} segundos**")

            if recos:
                df_recos = pd.DataFrame(
                    recos,
                    columns=["Título", "Score", "Votos Vecinos"]
                )
                st.subheader("Top Recomendaciones para ti")
                st.dataframe(df_recos.head(20), use_container_width=True)
            else:
                st.warning("No hay películas que superen los filtros establecidos.")

        # VECINOS
        if st.button("Ver vecinos más cercanos (Manhattan)"):
            inicio = time.perf_counter()

            vecinos = motor.obtener_vecinos_cercanos_manhattan(
                matriz, uid, k_vecinos, min_comunes
            )

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.info(f"Tiempo de ejecución: **{tiempo:.6f} segundos**")

            if vecinos:
                df_vecinos = pd.DataFrame(
                    vecinos,
                    columns=["ID Usuario Vecino", "Distancia Manhattan", "Películas en común"]
                )
                st.subheader(f"Top {k_vecinos} vecinos más cercanos del usuario {uid}")
                st.dataframe(df_vecinos, use_container_width=True)
            else:
                st.warning("No se encontraron vecinos con películas en común.")

        # ==============================
        # AGREGAR USUARIO
        # ==============================
        st.markdown("---")
        st.subheader("Agregar Usuario")

        texto_busqueda = st.text_input("Buscar película por nombre")

        if texto_busqueda.strip() != "":
            resultados = motor.buscar_peliculas_por_nombre(movies, texto_busqueda)

            if not resultados.empty:
                pelicula_seleccionada = st.selectbox(
                    "Seleccione una película",
                    resultados["title"].tolist()
                )

                rating_nuevo = st.selectbox(
                    "Calificación",
                    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                )

                if st.button("Agregar película"):
                    fila = resultados[resultados["title"] == pelicula_seleccionada].iloc[0]
                    movie_id = int(fila["movieId"])

                    st.session_state.peliculas_nuevo_usuario.append(
                        (movie_id, pelicula_seleccionada, rating_nuevo)
                    )

                    st.success("Película agregada.")

            else:
                st.warning("No se encontraron resultados.")

        # LISTA TEMPORAL
        if st.session_state.peliculas_nuevo_usuario:
            df_temp = pd.DataFrame(
                st.session_state.peliculas_nuevo_usuario,
                columns=["movieId", "Título", "Rating"]
            )

            st.dataframe(df_temp[["Título", "Rating"]], use_container_width=True)

            if st.button("Guardar nuevo usuario"):
                inicio = time.perf_counter()

                calificaciones = [
                    (movie_id, rating)
                    for movie_id, _, rating in st.session_state.peliculas_nuevo_usuario
                ]

                nuevo_uid, mensaje = motor.agregar_usuario_con_calificaciones(calificaciones)

                fin = time.perf_counter()
                tiempo = fin - inicio

                if nuevo_uid is not None:
                    st.success(f"{mensaje} ID: {nuevo_uid}")
                    st.info(f"Tiempo de ejecución: **{tiempo:.6f} segundos**")

                    # 🔥 RESET PARA RECARGAR DATASET
                    st.session_state.peliculas_nuevo_usuario = []
                    st.cache_data.clear()
                    st.session_state.cargado = False
                    st.session_state.tiempo_carga_mostrado = False
                    st.rerun()
                else:
                    st.error(mensaje)

    # ==============================
    # GRAFICAS
    # ==============================
    with col2:
        st.subheader("Análisis del Dataset")

        if st.button("Ver Distribución de Datos"):
            inicio = time.perf_counter()

            fig = motor.graficar_distribucion(matriz)

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.pyplot(fig)
            st.info(f"Tiempo de ejecución: **{tiempo:.6f} segundos**")

else:
    st.error("No se pudo cargar el dataset.")