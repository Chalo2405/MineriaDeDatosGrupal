import streamlit as st
import pandas as pd
import time
import Recomendador as motor
import psutil
import os

def obtener_memoria_mb():
    proceso = psutil.Process(os.getpid())
    memoria_bytes = proceso.memory_info().rss
    return memoria_bytes / (1024 * 1024)

# Configuración de página
st.set_page_config(page_title="Recomendador Renzo", layout="wide", page_icon=None)

st.title("Sistema de Recomendación de Películas")
st.markdown("---")

# ==============================
# CACHE DE DATOS INTEGRADO
# ==============================
@st.cache_data
def cargar_todos_los_datos():
    matriz, movies = motor.cargar_datos()
    
    df_ratings_full = pd.read_csv("ml-latest-small/ratings.csv")
    df_movies_full = pd.read_csv("ml-latest-small/movies.csv")
    
    dicc_generos = dict(zip(df_movies_full['movieId'], df_movies_full['genres'].str.split('|')))
    
    scores_objetivos, promedio_global = motor.calcular_scores_objetivos_manual(df_ratings_full)
    
    return matriz, movies, df_ratings_full, df_movies_full, dicc_generos, scores_objetivos, promedio_global

# ==============================
# INICIALIZACIÓN
# ==============================
if "peliculas_nuevo_usuario" not in st.session_state:
    st.session_state.peliculas_nuevo_usuario = []

with st.spinner("Cargando y procesando motores de recomendación (Esto puede tardar unos segundos)..."):
    try:
        matriz, movies, df_ratings_full, df_movies_full, dicc_generos, scores_obj, prom_global = cargar_todos_los_datos()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()

# ==============================
# BARRA LATERAL (MENÚ Y GLOBALES)
# ==============================
st.sidebar.title("Navegación")
opcion_menu = st.sidebar.radio(
    "Selecciona un módulo:",
    [
        "Motor Colaborativo (Vecinos)",
        "Motor Híbrido (Influencers/Géneros)",
        "Agregar Nuevo Usuario",
        "Análisis del Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Usuario Activo")
uid = st.sidebar.number_input("ID de Usuario Objetivo", 1, int(matriz.index.max()), 2)


# ==============================================================
# VISTA 1: MOTOR COLABORATIVO (PEARSON / MANHATTAN)
# ==============================================================
if opcion_menu == "Motor Colaborativo (Vecinos)":
    st.header("Motor Colaborativo (Basado en Usuarios)")
    st.write("Encuentra recomendaciones basándose en la correlación y distancia con otros usuarios de la red.")
    
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        k_vecinos = st.slider("Cantidad de Vecinos (K)", 5, 300, 50)
        soporte = st.slider("Soporte mínimo (Votos en común)", 1, 50, 5)
    with col_params2:
        min_comunes = st.slider("Películas mínimas para calcular distancia (Manhattan)", 1, 20, 3)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generar Recomendaciones (Pearson)", use_container_width=True):
            memoria_antes = obtener_memoria_mb()
            inicio = time.perf_counter()

            recos, media = motor.obtener_recomendaciones(matriz, movies, uid, k_vecinos, "pearson", soporte)

            tiempo = time.perf_counter() - inicio
            memoria_usada = obtener_memoria_mb() - memoria_antes

            st.success(f"Promedio histórico del usuario {uid}: {round(media, 2)}")
            if recos:
                st.subheader("Top Recomendaciones")
                st.dataframe(pd.DataFrame(recos, columns=["Título", "Score", "Votos Vecinos"]).head(10), use_container_width=True)
            else:
                st.warning("No hay recomendaciones con estos filtros.")
            st.info(f"Tiempo: {tiempo:.4f}s | RAM usada: {memoria_usada:.4f} MB")

    with col2:
        if st.button("Ver Vecinos Cercanos (Manhattan)", use_container_width=True):
            memoria_antes = obtener_memoria_mb()
            inicio = time.perf_counter()

            vecinos = motor.obtener_vecinos_cercanos_manhattan(matriz, uid, k_vecinos, min_comunes)

            tiempo = time.perf_counter() - inicio
            memoria_usada = obtener_memoria_mb() - memoria_antes

            if vecinos:
                st.subheader(f"Top {k_vecinos} Vecinos Físicos")
                st.dataframe(pd.DataFrame(vecinos, columns=["ID Vecino", "Distancia", "Pelis Común"]), use_container_width=True)
            else:
                st.warning("No se encontraron vecinos.")
            st.info(f"Tiempo: {tiempo:.4f}s | RAM usada: {memoria_usada:.4f} MB")


# ==============================================================
# VISTA 2: MOTOR HÍBRIDO (INFLUENCERS - TAREA 3)
# ==============================================================
elif opcion_menu == "Motor Híbrido (Influencers/Géneros)":
    st.header("Motor Híbrido: Perfilado de Usuario e Influencers")
    st.write("Este motor analiza matemáticamente tus afinidades por género y extrae las películas mejor valoradas objetivamente.")
    
    umbral_score = st.slider("Umbral mínimo de calidad (Score Objetivo)", 1.0, 5.0, 3.5, step=0.1)
    
    if st.button("Analizar Perfil y Recomendar", use_container_width=True):
        memoria_antes = obtener_memoria_mb()
        inicio = time.perf_counter()
        
        historial_usuario = df_ratings_full[df_ratings_full['userId'] == uid][['movieId', 'rating']].values.tolist()
        
        if len(historial_usuario) == 0:
            st.error("Este usuario no tiene un historial de películas calificadas.")
        else:
            afinidades = motor.calcular_afinidad_generos_manual(historial_usuario, dicc_generos)
            
            if not afinidades:
                st.error("No se pudo determinar la afinidad.")
            else:
                generos_ordenados = sorted(afinidades.items(), key=lambda x: x[1], reverse=True)
                genero_top = generos_ordenados[0][0]
                afinidad_top = generos_ordenados[0][1]
                
                st.success(f"Análisis completado. El género favorito del Usuario {uid} es: {genero_top} (Afinidad: {afinidad_top:.2f})")
                
                with st.expander("Ver desglose completo de afinidades por género"):
                    df_afinidad = pd.DataFrame(generos_ordenados, columns=["Género", "Score de Afinidad"])
                    st.dataframe(df_afinidad, use_container_width=True)
                
                peliculas_vistas = set([x[0] for x in historial_usuario])
                recomendaciones_influencer = []
                
                for m_id, score in scores_obj.items():
                    if score >= umbral_score and m_id not in peliculas_vistas:
                        generos_de_esta_peli = dicc_generos.get(m_id, [])
                        if genero_top in generos_de_esta_peli:
                            filtro_titulo = df_movies_full[df_movies_full['movieId'] == m_id]
                            if not filtro_titulo.empty:
                                titulo = filtro_titulo['title'].values[0]
                                recomendaciones_influencer.append((titulo, score))
                
                recomendaciones_influencer.sort(key=lambda x: x[1], reverse=True)
                
                st.subheader(f"Recomendaciones del Influencer de '{genero_top}'")
                if recomendaciones_influencer:
                    df_influencer = pd.DataFrame(recomendaciones_influencer, columns=["Película", "Score Objetivo (0-5)"])
                    st.dataframe(df_influencer.head(15), use_container_width=True)
                else:
                    st.warning("Ya has visto todas las películas buenas de este género o ninguna supera el umbral establecido.")
                    
        tiempo = time.perf_counter() - inicio
        memoria_usada = obtener_memoria_mb() - memoria_antes

        st.info(f"Tiempo: {tiempo:.4f}s | RAM usada: {memoria_usada:.4f} MB")


# ==============================================================
# VISTA 3: AGREGAR NUEVO USUARIO
# ==============================================================
elif opcion_menu == "Agregar Nuevo Usuario":
    st.header("Añadir Usuario a la Base de Datos")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        texto_busqueda = st.text_input("Buscar película por nombre")

        if texto_busqueda.strip() != "":
            resultados = motor.buscar_peliculas_por_nombre(movies, texto_busqueda)

            if not resultados.empty:
                pelicula_seleccionada = st.selectbox("Seleccione una película", resultados["title"].tolist())
                rating_nuevo = st.selectbox("Calificación", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

                if st.button("Agregar al carrito de calificaciones", use_container_width=True):
                    fila = resultados[resultados["title"] == pelicula_seleccionada].iloc[0]
                    movie_id = int(fila["movieId"])

                    ya_existe = False
                    for m_id, _, _ in st.session_state.peliculas_nuevo_usuario:
                        if m_id == movie_id:
                            ya_existe = True
                            break

                    if ya_existe:
                        st.warning("Esa película ya fue agregada a la lista temporal.")
                    else:
                        st.session_state.peliculas_nuevo_usuario.append((movie_id, pelicula_seleccionada, rating_nuevo))
                        st.success("Película agregada a la lista temporal.")
            else:
                st.warning("No se encontraron resultados.")

    with col2:
        st.subheader("Lista de Valoraciones")
        if st.session_state.peliculas_nuevo_usuario:
            df_temp = pd.DataFrame(st.session_state.peliculas_nuevo_usuario, columns=["movieId", "Título", "Rating"])
            st.dataframe(df_temp[["Título", "Rating"]], use_container_width=True)

            if st.button("Guardar Nuevo Usuario en Base de Datos", use_container_width=True):
                calificaciones = [(m, r) for m, _, r in st.session_state.peliculas_nuevo_usuario]
                nuevo_uid, mensaje = motor.agregar_usuario_con_calificaciones(calificaciones)

                if nuevo_uid is not None:
                    st.success(f"Éxito. {mensaje} Su nuevo ID es: {nuevo_uid}")
                    st.session_state.peliculas_nuevo_usuario = []
                    st.cache_data.clear()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(mensaje)
        else:
            st.info("Aún no has agregado ninguna calificación.")


# ==============================================================
# VISTA 4: ANÁLISIS DE DATASET
# ==============================================================
elif opcion_menu == "Análisis del Dataset":
    st.header("Estadísticas del Sistema")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Usuarios", f"{matriz.shape[0]:,}")
    col2.metric("Total Películas", f"{matriz.shape[1]:,}")
    col3.metric("Valoraciones Totales", f"{matriz.count().sum():,}")
    
    st.markdown("---")
    
    if st.button("Generar Gráfico de Distribución (Long Tail)"):
        inicio = time.perf_counter()
        fig = motor.graficar_distribucion(matriz)
        st.pyplot(fig)
        st.info(f"Gráfico renderizado en {time.perf_counter() - inicio:.4f}s")