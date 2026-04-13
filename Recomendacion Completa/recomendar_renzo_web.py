import streamlit as st
import pandas as pd
import time
import recomendador_renzo as motor

# Configuración de página
st.set_page_config(page_title="Recomendador Renzo", layout="wide")

st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("---")

# Carga de datos
with st.spinner("Cargando base de datos..."):
    matriz, movies = motor.cargar_datos()

if matriz is not None:
    # Sidebar: Configuración
    st.sidebar.header("⚙️ Parámetros del Motor")
    uid = st.sidebar.number_input("ID de Usuario", 1, 610, 2)
    k_vecinos = st.sidebar.slider("Vecinos (K)", 5, 300, 50)
    soporte = st.sidebar.slider("Soporte Mínimo (Votos)", 1, 50, 5)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Opciones del sistema")

        if st.button("🚀 Generar Recomendaciones"):
            inicio = time.perf_counter()

            recos, media = motor.obtener_recomendaciones(
                matriz, movies, uid, k_vecinos, "pearson", soporte
            )

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.success(f"Promedio del usuario {uid}: **{round(media, 2)}**")
            st.info(f"⏱️ Tiempo de ejecución: **{tiempo:.6f} segundos**")

            if recos:
                st.subheader("Top Recomendaciones para ti:")
                df_recos = pd.DataFrame(recos, columns=["Título", "Score", "Votos Vecinos"])
                st.dataframe(df_recos.head(20), use_container_width=True)
            else:
                st.warning("No hay películas que superen los filtros establecidos.")

        if st.button("👥 Ver vecinos más cercanos (Manhattan)"):
            inicio = time.perf_counter()

            vecinos = motor.obtener_vecinos_cercanos_manhattan(matriz, uid, k_vecinos)

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.info(f"⏱️ Tiempo de ejecución: **{tiempo:.6f} segundos**")

            if vecinos:
                df_vecinos = pd.DataFrame(
                    vecinos,
                    columns=["ID Usuario Vecino", "Distancia Manhattan", "Películas en común"]
                )
                st.subheader(f"Top {k_vecinos} vecinos más cercanos del usuario {uid}")
                st.dataframe(df_vecinos, use_container_width=True)
            else:
                st.warning("No se encontraron vecinos con películas en común.")

    with col2:
        st.subheader("📊 Análisis del Dataset")

        if st.button("Ver Distribución de Datos"):
            inicio = time.perf_counter()

            fig = motor.graficar_distribucion(matriz)

            fin = time.perf_counter()
            tiempo = fin - inicio

            st.pyplot(fig)
            st.info("Nota: Se ha hecho un zoom al rango 0-60 para observar mejor la mayoría de las películas.")
            st.info(f"⏱️ Tiempo de ejecución: **{tiempo:.6f} segundos**")
else:
    st.error("No se pudo cargar el dataset. Verifica la carpeta 'ml-latest-small'.")