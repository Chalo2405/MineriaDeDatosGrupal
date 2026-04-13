import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. CARGA DE DATOS
# ==============================
def cargar_datos():
    try:
        # Cargamos solo lo necesario para optimizar memoria
        df_ratings = pd.read_csv("ml-latest-small/ratings.csv", usecols=["userId", "movieId", "rating"])
        df_movies = pd.read_csv("ml-latest-small/movies.csv", usecols=["movieId", "title"])
        # Matriz pivot: Filas=Usuarios, Columnas=Películas. No llenamos con 0.
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
    if len(comunes) == 0: return float('inf')
    distancia = sum(abs(u1[i] - u2[i]) for i in comunes)
    return distancia

def euclidiana_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) == 0: return float('inf')
    suma_sq = sum((u1[i] - u2[i])**2 for i in comunes)
    return suma_sq**0.5

def coseno_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) == 0: return 0
    v1, v2 = u1[comunes].values, u2[comunes].values
    num = np.sum(v1 * v2)
    den = (np.sum(v1**2)**0.5) * (np.sum(v2**2)**0.5)
    return num / den if den != 0 else 0

def pearson_manual(u1, u2):
    comunes = u1.dropna().index.intersection(u2.dropna().index)
    if len(comunes) < 2: return 0
    v1, v2 = u1[comunes].values, u2[comunes].values
    mu1, mu2 = np.mean(v1), np.mean(v2)
    num = np.sum((v1 - mu1) * (v2 - mu2))
    den = (np.sum((v1 - mu1)**2)**0.5) * (np.sum((v2 - mu2)**2)**0.5)
    return num / den if den != 0 else 0

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
# 3. MOTOR DE RECOMENDACIÓN INTELIGENTE
# ==============================

def obtener_recomendaciones(matriz, df_movies, user_id, k=10, metrica="pearson", min_soporte=3):
    u_obj = matriz.loc[user_id]
    media_u = u_obj.mean() # Umbral dinámico base
    similitudes = []

    # Búsqueda de Vecinos
    for otro_id in matriz.index:
        if otro_id == user_id: continue
        u_otro = matriz.loc[otro_id]
        
        if metrica == "pearson": sim = pearson_manual(u_obj, u_otro)
        elif metrica == "coseno": sim = coseno_manual(u_obj, u_otro)
        elif metrica == "manhattan": sim = 1 / (1 + manhattan_manual(u_obj, u_otro))
        else: sim = 1 / (1 + euclidiana_manual(u_obj, u_otro))
        
        if sim > 0: similitudes.append((otro_id, sim))

    similitudes.sort(key=lambda x: x[1], reverse=True)
    vecinos = similitudes[:k]

    # Predicción con Filtros
    recomendaciones = []
    no_vistas = matriz.columns.difference(u_obj.dropna().index)

    for m_id in no_vistas:
        num, den, conteo = 0, 0, 0
        for v_id, sim in vecinos:
            nota = matriz.loc[v_id, m_id]
            if not pd.isna(nota):
                num += nota * sim
                den += sim
                conteo += 1
        
        if den > 0 and conteo >= min_soporte: # Filtro de soporte (Popularidad)
            score = num / den
            # Filtro de Calidad (Umbral dinámico: solo si es mejor que su promedio)
            if score >= media_u:
                titulo = df_movies[df_movies['movieId'] == m_id]['title'].values[0]
                recomendaciones.append((titulo, score, conteo))

    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    return recomendaciones, media_u

# ==============================
# 4. INTERFAZ DE USUARIO (MENÚ)
# ==============================

def menu():
    print("Cargando base de datos MovieLens...")
    matriz, movies = cargar_datos()
    if matriz is None: return

    while True:
        print("\n========================================")
        print("   SISTEMA DE RECOMENDACIÓN FILTRADO")
        print("========================================")
        print(f"Usuarios en sistema: {len(matriz.index)}")
        print("1. Generar Recomendaciones")
        print("2. Salir")
        
        op = input("\nSeleccione opción: ")
        
        if op == "1":
            try:
                uid = int(input("ID de Usuario (1-610): "))
                if uid not in matriz.index:
                    print("Usuario no existe.")
                    continue
                
                print("\nMétricas: pearson | coseno | manhattan | euclidiana")
                met = input("Elija métrica: ").lower().strip()
                k = int(input("Número de vecinos (K): "))
                soporte = int(input("Mínimo de vecinos que deben haber visto la película (Soporte): "))
                
                recos, media = obtener_recomendaciones(matriz, movies, uid, k, met, soporte)
                
                print(f"\n--- RESULTADOS PARA USUARIO {uid} ---")
                print(f"Su promedio de nota actual es: {round(media, 2)}")
                print(f"Mostrando top 10 que superan su promedio y tienen soporte >= {soporte}:")
                print("-" * 60)
                
                if not recos:
                    print("No se encontraron películas que cumplan los filtros.")
                else:
                    for i, (tit, score, n) in enumerate(recos[:10], 1):
                        print(f"{i}. {tit[:40]:40} | Score: {round(score, 2)} | (Visto por {n} vecinos)")
            except ValueError:
                print("Error: Ingrese valores numéricos válidos.")
        
        elif op == "2":
            print("Saliendo...")
            break

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

if __name__ == "__main__":
    menu()