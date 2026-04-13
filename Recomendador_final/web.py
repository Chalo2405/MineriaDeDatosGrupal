import streamlit as st
import pandas as pd
import time
import johan as motor

# Configuraci贸n de p谩gina
st.set_page_config(page_title="Recomendador Renzo", layout="wide")

st.title("Sistema de Recomendacion de Peliculas")
st.markdown("---")

# ==============================
#  BARRA DE CARGA + TIEMPO
# ==============================
progress_bar = st.progress(0)
status_text = st.empty()

inicio_carga = time.perf_counter()

# Simulaci贸n de carga progresiva
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)
    status_text.text(f"Cargando dataset... {i+1}%")

# Carga real (SIN modificar tu l贸gica)
matriz, movies = motor.cargar_datos()

fin_carga = time.perf_counter()
tiempo_carga = fin_carga - inicio_carga

progress_bar.empty()
status_text.empty()

# ==============================
# RESULTADO DE CARGA
# ==============================
if matriz is not None:
    st.success(" Dataset cargado correctamente")
    st.info(f" Tiempo total de carga: {tiempo_carga:.4f} segundos")

    st.subheader(" Informacion del Dataset")
    st.write(f"Usuarios: {matriz.shape[0]}")
    st.write(f"Peliculas: {matriz.shape[1]}")
    st.write(f"Calificaciones totales: {matriz.count().sum()}")

    # ==============================
    # SIDEBAR
    # ==============================
    st.sidebar.header(" Parametros del Motor")
    uid = st.sidebar.number_input("ID de Usuario", 1, 610, 2)
    k_vecinos = st.sidebar.slider("Vecinos (K)", 5, 300, 50)
    soporte = st.sidebar.slider("Soporte Minimo (Votos)", 1, 50, 5)

    col1, col2 = st.columns([2, 1])

    # ==============================
    # FUNCIONES PRINCIPALES
    # ==============================
    with col1:
        st.subheader(" Opciones del sistema")

        if st.button(" Generar Recomendaciones"):
            inicio = time.perf_counter()

            recos, media = motor.obtener_recomendaciones(
                matriz, movies, uid, k_vecinos, "pearson", soporte
            )

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.success(f"Promedio del usuario {uid}: **{round(media, 2)}**")
            st.info(f" Tiempo de ejecucion: **{tiempo:.6f} segundos**")

            if recos:
                st.subheader("Top Recomendaciones para ti:")
                df_recos = pd.DataFrame(recos, columns=["Titulo", "Score", "Votos Vecinos"])
                st.dataframe(df_recos.head(20), use_container_width=True)
            else:
                st.warning("No hay peliculas que superen los filtros establecidos.")

        if st.button("Ver vecinos mas cercanos (Manhattan)"):
            inicio = time.perf_counter()

            vecinos = motor.obtener_vecinos_cercanos_manhattan(matriz, uid, k_vecinos)

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.info(f" Tiempo de ejecucion: **{tiempo:.6f} segundos**")

            if vecinos:
                df_vecinos = pd.DataFrame(
                    vecinos,
                    columns=["ID Usuario Vecino", "Distancia Manhattan", "Peliculas en comunes"]
                )
                st.subheader(f"Top {k_vecinos} vecinos mas cercanos del usuario {uid}")
                st.dataframe(df_vecinos, use_container_width=True)
            else:
                st.warning("No se encontraron vecinos con peliculas en comunes.")

    # ==============================
    # GRAFICAS
    # ==============================
    with col2:
        st.subheader(" Analisis del Dataset")

        if st.button("Ver Distribucion de Datos"):
            inicio = time.perf_counter()

            fig = motor.graficar_distribucion(matriz)

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.pyplot(fig)
            st.info("Nota: Se ha hecho un zoom al rango 0-60 para observar mejor la mayoria de las peliculas.")
            st.info(f" Tiempo de ejecucion: **{tiempo:.6f} segundos**")

else:
    st.error("No se pudo cargar el dataset. Verifica la carpeta 'ml-latest-small'.")