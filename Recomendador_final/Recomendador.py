import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


# ==============================
# 1. CARGA DE DATOS
# ==============================
def cargar_datos():
    try:
        df_ratings = pd.read_csv(
            "ml-latest-small/ratings.csv",
            usecols=["userId", "movieId", "rating"]
        )
        df_movies = pd.read_csv(
            "ml-latest-small/movies.csv",
            usecols=["movieId", "title"]
        )

        matriz = df_ratings.pivot(index="userId", columns="movieId", values="rating")
        return matriz, df_movies

    except FileNotFoundError:
        print("Error: No se encontraron los archivos en la carpeta 'ml-latest-small/'.")
        return None, None


# ==============================
# 2. FUNCIONES MATEMÁTICAS MANUALES
# ==============================
def manhattan_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) == 0:
        return float('inf')

    distancia = sum(abs(u1[i] - u2[i]) for i in comunes)
    return distancia


def euclidiana_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) == 0:
        return float('inf')

    suma_sq = sum((u1[i] - u2[i]) ** 2 for i in comunes)
    return suma_sq ** 0.5


def coseno_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) == 0:
        return 0

    v1 = u1[comunes].values
    v2 = u2[comunes].values

    num = np.sum(v1 * v2)
    den = (np.sum(v1 ** 2) ** 0.5) * (np.sum(v2 ** 2) ** 0.5)

    if den != 0:
        return num / den
    return 0


def pearson_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) < 2:
        return 0

    v1 = u1[comunes].values
    v2 = u2[comunes].values

    mu1 = np.mean(v1)
    mu2 = np.mean(v2)

    num = np.sum((v1 - mu1) * (v2 - mu2))
    den = (np.sum((v1 - mu1) ** 2) ** 0.5) * (np.sum((v2 - mu2) ** 2) ** 0.5)

    if den != 0:
        return num / den
    return 0


# ==============================
# 3. VECINOS CERCANOS MANHATTAN
# ==============================
def obtener_vecinos_cercanos_manhattan(matriz, user_id, k=10, min_comunes=3):
    if user_id not in matriz.index:
        return []

    u_obj = matriz.loc[user_id]
    vecinos = []

    for otro_id in matriz.index:
        if otro_id == user_id:
            continue

        u_otro = matriz.loc[otro_id]
        comunes = u_obj.dropna().index.intersection(u_otro.dropna().index)

        if len(comunes) >= min_comunes:
            distancia = sum(abs(u_obj[i] - u_otro[i]) for i in comunes)
            vecinos.append((otro_id, distancia, len(comunes)))

    vecinos.sort(key=lambda x: (x[1], -x[2]))
    return vecinos[:k]


# ==============================
# 4. MOTOR DE RECOMENDACIÓN
# ==============================
def obtener_recomendaciones(matriz, df_movies, user_id, k=10, metrica="pearson", min_soporte=3):
    if user_id not in matriz.index:
        return [], 0

    u_obj = matriz.loc[user_id]
    media_u = u_obj.mean()
    similitudes = []

    for otro_id in matriz.index:
        if otro_id == user_id:
            continue

        u_otro = matriz.loc[otro_id]

        if metrica == "pearson":
            sim = pearson_manual(u_obj, u_otro)
        elif metrica == "coseno":
            sim = coseno_manual(u_obj, u_otro)
        elif metrica == "manhattan":
            sim = 1 / (1 + manhattan_manual(u_obj, u_otro))
        else:
            sim = 1 / (1 + euclidiana_manual(u_obj, u_otro))

        if sim > 0:
            similitudes.append((otro_id, sim))

    similitudes.sort(key=lambda x: x[1], reverse=True)
    vecinos = similitudes[:k]

    recomendaciones = []
    no_vistas = matriz.columns.difference(u_obj.dropna().index)

    for m_id in no_vistas:
        num = 0
        den = 0
        conteo = 0

        for v_id, sim in vecinos:
            nota = matriz.loc[v_id, m_id]
            if not pd.isna(nota):
                num += nota * sim
                den += sim
                conteo += 1

        if den > 0 and conteo >= min_soporte:
            score = num / den

            if score >= media_u:
                fila_pelicula = df_movies[df_movies["movieId"] == m_id]

                if not fila_pelicula.empty:
                    titulo = fila_pelicula["title"].values[0]
                    recomendaciones.append((titulo, score, conteo))

    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    return recomendaciones, media_u


# ==============================
# 5. BUSCAR PELÍCULAS
# ==============================
def buscar_peliculas_por_nombre(df_movies, texto):
    if texto.strip() == "":
        return pd.DataFrame(columns=df_movies.columns)

    resultados = df_movies[df_movies["title"].str.contains(texto, case=False, na=False)]
    return resultados.head(30)


# ==============================
# 6. AGREGAR USUARIO PERSISTENTE
# ==============================
def agregar_usuario_con_calificaciones(calificaciones, ruta_ratings="ml-latest-small/ratings.csv"):
    if len(calificaciones) == 0:
        return None, "No se ingresaron calificaciones."

    try:
        df_ratings = pd.read_csv(ruta_ratings)

        if "userId" not in df_ratings.columns or "movieId" not in df_ratings.columns or "rating" not in df_ratings.columns:
            return None, "El archivo ratings.csv no tiene las columnas necesarias."

        nuevo_user_id = int(df_ratings["userId"].max()) + 1
        timestamp_actual = int(time.time())

        nuevas_filas = []

        for movie_id, rating in calificaciones:
            fila = {
                "userId": nuevo_user_id,
                "movieId": int(movie_id),
                "rating": float(rating)
            }

            if "timestamp" in df_ratings.columns:
                fila["timestamp"] = timestamp_actual

            nuevas_filas.append(fila)

        df_nuevo = pd.DataFrame(nuevas_filas)

        for col in df_ratings.columns:
            if col not in df_nuevo.columns:
                df_nuevo[col] = np.nan

        df_nuevo = df_nuevo[df_ratings.columns]

        df_ratings = pd.concat([df_ratings, df_nuevo], ignore_index=True)
        df_ratings.to_csv(ruta_ratings, index=False)

        return nuevo_user_id, "Usuario agregado correctamente."

    except Exception as e:
        return None, f"Error al guardar usuario: {e}"
# ========================        
# ESTADISTICAS DATAFRAME
# ========================
def graficar_distribucion(matriz):
    conteo_votos = matriz.count()
    
    # Creamos la figura y el eje de forma explícita (evita el error de Streamlit)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Filtramos los datos para el histograma para que se vea mejor el "zoom"
    # Solo mostramos películas con hasta 100 votos para no aplastar la gráfica
    ax.hist(conteo_votos, bins=50, range=(0, 100), color='skyblue', edgecolor='black')
    
    ax.set_title("Distribución de Popularidad (Zoom en películas con pocos votos)")
    ax.set_xlabel("Número de Calificaciones recibidas")
    ax.set_ylabel("Cantidad de Películas")
    
    # Ajustamos el límite del eje X para que se vea claro el rango 0-50
    ax.set_xlim(0, 60) 
    ax.grid(axis='y', alpha=0.3)
    
    return fig # Devolvemos el objeto figura

# ==============================================================
# 7. TAREA 3: MOTOR HÍBRIDO Y MATEMÁTICA MANUAL (INFLUENCERS)
# ==============================================================

def calcular_scores_objetivos_manual(df_ratings):
    """
    Calcula el score real de cada película eliminando el sesgo del usuario.
    Toda la matemática se hace manualmente con diccionarios y listas.
    """
    print("Calculando promedios ajustados manualmente...")
    
    # 1. Extraemos los datos a una lista pura de Python para iterar rápido
    # Formato esperado por cada fila: [userId, movieId, rating]
    lista_ratings = df_ratings[['userId', 'movieId', 'rating']].values.tolist()
    
    suma_global = 0
    conteo_global = 0
    
    sumas_usuarios = {}
    conteos_usuarios = {}
    
    # --- PASO 1: Calcular sumatorias totales y por usuario ---
    for fila in lista_ratings:
        u_id = int(fila[0])
        m_id = int(fila[1])
        nota = float(fila[2])
        
        # Para el promedio global
        suma_global += nota
        conteo_global += 1
        
        # Para el promedio de cada usuario
        if u_id not in sumas_usuarios:
            sumas_usuarios[u_id] = 0.0
            conteos_usuarios[u_id] = 0
            
        sumas_usuarios[u_id] += nota
        conteos_usuarios[u_id] += 1
        
    # --- PASO 2: Calcular promedios (Divisiones manuales) ---
    promedio_global = suma_global / conteo_global if conteo_global > 0 else 0
    
    promedios_usuarios = {}
    for u_id in sumas_usuarios:
        promedios_usuarios[u_id] = sumas_usuarios[u_id] / conteos_usuarios[u_id]
        
    # --- PASO 3: Calcular el Score Ajustado de las películas ---
    sumas_ajustadas_pelis = {}
    conteos_pelis = {}
    
    for fila in lista_ratings:
        u_id = int(fila[0])
        m_id = int(fila[1])
        nota = float(fila[2])
        
        # Matemática: Nota original - Promedio de ese usuario
        nota_ajustada = nota - promedios_usuarios[u_id]
        
        if m_id not in sumas_ajustadas_pelis:
            sumas_ajustadas_pelis[m_id] = 0.0
            conteos_pelis[m_id] = 0
            
        sumas_ajustadas_pelis[m_id] += nota_ajustada
        conteos_pelis[m_id] += 1

    # --- PASO 4: Generar el diccionario final de Scores Objetivos ---
    scores_finales_peliculas = {}
    for m_id in sumas_ajustadas_pelis:
        promedio_ajustado = sumas_ajustadas_pelis[m_id] / conteos_pelis[m_id]
        
        # Score final = Promedio Global + Promedio Ajustado de la peli
        score_final = promedio_global + promedio_ajustado
        
        # Control manual de límites (0 a 5)
        if score_final > 5.0: 
            score_final = 5.0
        elif score_final < 0.0: 
            score_final = 0.0
            
        scores_finales_peliculas[m_id] = score_final
        
    return scores_finales_peliculas, promedio_global


def calcular_afinidad_generos_manual(historial_usuario_lista, dicc_peliculas_generos):
    """
    Calcula qué tanto le gusta un género a un usuario combinando cantidad y nota.
    - historial_usuario_lista: lista de listas [[movieId, rating], [movieId, rating]...]
    - dicc_peliculas_generos: diccionario {movieId: ["Action", "Sci-Fi", ...]}
    """
    vistas_por_genero = {}
    sumas_por_genero = {}
    vistas_totales = 0
    
    # --- PASO 1: Recorrer el historial del usuario ---
    for fila in historial_usuario_lista:
        m_id = int(fila[0])
        nota = float(fila[1])
        vistas_totales += 1
        
        # Obtener los géneros de esta película (Si no existe, devuelve lista vacía)
        generos_peli = dicc_peliculas_generos.get(m_id, [])
        
        for genero in generos_peli:
            # Descartamos si la película no tiene género válido
            if genero == "(no genres listed)": 
                continue
                
            if genero not in vistas_por_genero:
                vistas_por_genero[genero] = 0
                sumas_por_genero[genero] = 0.0
                
            vistas_por_genero[genero] += 1
            sumas_por_genero[genero] += nota

    # --- PASO 2: Aplicar la fórmula de afinidad ---
    afinidad_final = {}
    
    if vistas_totales == 0:
        return afinidad_final
    
    for genero in vistas_por_genero:
        # Promedio de notas para ese género
        promedio_nota_genero = sumas_por_genero[genero] / vistas_por_genero[genero]
        
        # Frecuencia (qué porcentaje de sus películas son de este género)
        frecuencia = vistas_por_genero[genero] / vistas_totales
        
        # Afinidad = Frecuencia * Promedio de Nota
        afinidad_final[genero] = frecuencia * promedio_nota_genero
        
    return afinidad_final


if __name__ == "__main__":
    menu()