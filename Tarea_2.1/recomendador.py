import pandas as pd
import numpy as np

RUTA_RATINGS = "ml-latest-small/ratings.csv"
RUTA_MOVIES = "ml-latest-small/movies.csv"


def cargar_datos():
    ratings = pd.read_csv(RUTA_RATINGS)
    movies = pd.read_csv(RUTA_MOVIES)
    matriz = ratings.pivot(index="userId", columns="movieId", values="rating")
    return ratings, movies, matriz


def similitud_pearson(matriz, usuario1, usuario2):
    r1 = matriz.loc[usuario1]
    r2 = matriz.loc[usuario2]

    comunes = pd.concat([r1, r2], axis=1).dropna()

    if len(comunes) < 2:
        return 0

    u1 = comunes.iloc[:, 0]
    u2 = comunes.iloc[:, 1]

    media1 = u1.mean()
    media2 = u2.mean()

    numerador = ((u1 - media1) * (u2 - media2)).sum()
    denominador = np.sqrt(((u1 - media1) ** 2).sum()) * np.sqrt(((u2 - media2) ** 2).sum())

    if denominador == 0:
        return 0

    return numerador / denominador


def obtener_similitud(tupla):
    return tupla[1]


def obtener_prediccion(tupla):
    return tupla[2]


def knn_usuarios(matriz, usuario_objetivo, k=5):
    vecinos = []

    for usuario in matriz.index:
        if usuario != usuario_objetivo:
            sim = similitud_pearson(matriz, usuario_objetivo, usuario)
            vecinos.append((usuario, sim))

    vecinos.sort(key=obtener_similitud, reverse=True)
    return vecinos[:k]


def recomendar_peliculas(matriz, movies, usuario_objetivo, k=5, umbral=3, min_vecinos=4):
    vecinos = knn_usuarios(matriz, usuario_objetivo, k)
    vistos_por_u = matriz.loc[usuario_objetivo]
    peliculas_no_vistas = vistos_por_u[vistos_por_u.isna()].index

    recomendaciones = []
    media_usuario = matriz.loc[usuario_objetivo].mean()

    for pelicula in peliculas_no_vistas:
        numerador = 0
        denominador = 0
        contador_vecinos = 0

        for vecino_id, similitud in vecinos:
            rating_vecino = matriz.loc[vecino_id, pelicula]

            if not pd.isna(rating_vecino) and similitud > 0:
                media_vecino = matriz.loc[vecino_id].mean()
                numerador += similitud * (rating_vecino - media_vecino)
                denominador += abs(similitud)
                contador_vecinos += 1

        if denominador > 0 and contador_vecinos >= min_vecinos:
            prediccion = media_usuario + (numerador / denominador)

            if prediccion > 5:
                prediccion = 5
            elif prediccion < 0.5:
                prediccion = 0.5

            if prediccion > umbral:
                fila = movies[movies["movieId"] == pelicula]
                if not fila.empty:
                    titulo = fila.iloc[0]["title"]
                    recomendaciones.append((pelicula, titulo, prediccion, contador_vecinos))

    recomendaciones.sort(key=obtener_prediccion, reverse=True)
    return recomendaciones


def pedir_usuario_valido(matriz):
    usuarios = list(matriz.index)
    minimo = min(usuarios)
    maximo = max(usuarios)

    print("\nUsuarios disponibles en el dataset:")
    print("Desde", minimo, "hasta", maximo)

    while True:
        entrada = input("Ingrese el ID del usuario: ").strip()

        if not entrada.isdigit():
            print("Ingrese un número válido.")
            continue

        usuario = int(entrada)

        if usuario not in matriz.index:
            print("Ese usuario no existe en el dataset.")
            continue

        return usuario


def pedir_k(maximo_permitido):
    while True:
        entrada = input("Ingrese el valor de k: ").strip()

        if not entrada.isdigit():
            print("Ingrese un número entero válido.")
            continue

        k = int(entrada)

        if k <= 0:
            print("k debe ser mayor que 0.")
            continue

        if k > maximo_permitido:
            print("k no puede ser mayor que", maximo_permitido)
            continue

        return k


def pedir_umbral():
    while True:
        entrada = input("Ingrese el umbral de recomendación: ").strip()

        try:
            umbral = float(entrada)

            if umbral < 0.5 or umbral > 5:
                print("El umbral debe estar entre 0.5 y 5.")
                continue

            return umbral
        except ValueError:
            print("Ingrese un valor numérico válido.")


def pedir_cantidad_resultados():
    while True:
        entrada = input("¿Cuántos resultados desea mostrar?: ").strip()

        if not entrada.isdigit():
            print("Ingrese un número entero válido.")
            continue

        n = int(entrada)

        if n <= 0:
            print("Debe ser mayor que 0.")
            continue

        return n


def ver_vecinos(matriz):
    usuario = pedir_usuario_valido(matriz)
    k = pedir_k(len(matriz.index) - 1)

    vecinos = knn_usuarios(matriz, usuario, k)

    print("\n========== T1: VECINOS MÁS CERCANOS ==========")
    print("Usuario objetivo:", usuario)
    print("Cantidad de vecinos (k):", k)
    print()

    for i, vecino in enumerate(vecinos, start=1):
        print(
            str(i) + ". Usuario:",
            vecino[0],
            "| Similitud Pearson:",
            round(vecino[1], 4)
        )


def ver_recomendaciones(matriz, movies):
    usuario = pedir_usuario_valido(matriz)
    k = pedir_k(len(matriz.index) - 1)
    umbral = pedir_umbral()
    cantidad = pedir_cantidad_resultados()

    recomendaciones = recomendar_peliculas(matriz, movies, usuario, k, umbral)

    print("\n========== T2: RECOMENDACIONES ==========")
    print("Usuario objetivo:", usuario)
    print("Cantidad de vecinos (k):", k)
    print("Umbral:", umbral)
    print()

    if len(recomendaciones) == 0:
        print("No se encontraron recomendaciones que superen el umbral.")
        return

    print("Total de recomendaciones encontradas:", len(recomendaciones))
    print()

    limite = min(cantidad, len(recomendaciones))

    for i in range(limite):
        pelicula_id, titulo, prediccion, vecinos_aportaron = recomendaciones[i]
        print(str(i + 1) + ".")
        print("   MovieId:", pelicula_id)
        print("   Título:", titulo)
        print("   Predicción:", round(prediccion, 4))
        print("   Vecinos que aportaron:", vecinos_aportaron)
        print()


def menu():
    try:
        ratings, movies, matriz = cargar_datos()
    except FileNotFoundError:
        print("No se encontraron los archivos.")
        print("Verifica que exista la carpeta 'ml-latest-small' con ratings.csv y movies.csv.")
        return

    total_usuarios = len(matriz.index)
    minimo_usuario = min(matriz.index)
    maximo_usuario = max(matriz.index)

    while True:
        print("\n================ MENÚ PRINCIPAL ================")
        print("Usuarios cargados:", total_usuarios)
        print("Rango de usuarios:", minimo_usuario, "a", maximo_usuario)
        print("1. Ver rango de usuarios")
        print("2. Ejecutar T1 - Vecinos más cercanos (K-NN)")
        print("3. Ejecutar T2 - Recomendación de películas")
        print("4. Salir")

        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            print("\nUsuarios cargados en el dataset:", total_usuarios)
            print("Rango válido de userId:", minimo_usuario, "a", maximo_usuario)
        elif opcion == "2":
            ver_vecinos(matriz)
        elif opcion == "3":
            ver_recomendaciones(matriz, movies)
        elif opcion == "4":
            print("Programa finalizado.")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    menu()