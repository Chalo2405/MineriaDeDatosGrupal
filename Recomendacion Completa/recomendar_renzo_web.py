import streamlit as st
import recomendador_renzo as motor # Importamos tu lógica

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
        if st.button("🚀 Generar Recomendaciones"):
            recos, media = motor.obtener_recomendaciones(matriz, movies, uid, k_vecinos, "pearson", soporte)
            
            st.success(f"Promedio del usuario {uid}: **{round(media, 2)}**")
            
            if recos:
                st.subheader(f"Top Recomendaciones para ti:")
                # Mostrar en tabla
                import pandas as pd
                df_recos = pd.DataFrame(recos, columns=["Título", "Score", "Votos Vecinos"])
                st.dataframe(df_recos.head(20), use_container_width=True)
            else:
                st.warning("No hay películas que superen los filtros establecidos.")

    with col2:
        st.subheader("📊 Análisis del Dataset")
        if st.button("Ver Distribución de Datos"):
            # Ahora 'fig' es el objeto que devuelve la función
            fig = motor.graficar_distribucion(matriz)
            # Pasamos 'fig' directamente a st.pyplot
            st.pyplot(fig) 
            st.info("Nota: Se ha hecho un zoom al rango 0-60 para observar mejor la mayoría de las películas.")
else:
    st.error("No se pudo cargar el dataset. Verifica la carpeta 'ml-latest-small'.")