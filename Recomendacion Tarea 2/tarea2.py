import csv

# ==============================
# 1. CARGAR DATOS DESDE CSV
# ==============================
def cargar_datos(nombre_archivo):
    datos = {}
    try:
        with open(nombre_archivo, newline='', encoding='utf-8') as archivo:
            lector = csv.DictReader(archivo)
            for fila in lector:
                usuario = fila['Usuario'].strip()
                datos[usuario] = {}

                for item in fila:
                    if item != 'Usuario':
                        valor = fila[item].strip()
                        if valor != "":
                            try:
                                datos[usuario][item] = float(valor)
                            except ValueError:
                                continue
        return datos
    except FileNotFoundError:
        return None

# ==============================
# 2. DISTANCIA MANHATTAN
# ==============================
def manhattan(usuario1, usuario2):
    comunes = [item for item in usuario1 if item in usuario2]

    if not comunes:
        return float('inf')

    distancia = 0
    for item in comunes:
        distancia += abs(usuario1[item] - usuario2[item])

    return distancia

# ==============================
# 3. SIMILITUD PEARSON
# ==============================
def pearson(usuario1, usuario2):
    comunes = [item for item in usuario1 if item in usuario2]
    n = len(comunes)

    if n == 0:
        return 0

    sum1 = sum(usuario1[item] for item in comunes)
    sum2 = sum(usuario2[item] for item in comunes)
    sum1_sq = sum(usuario1[item] ** 2 for item in comunes)
    sum2_sq = sum(usuario2[item] ** 2 for item in comunes)
    sum_prod = sum(usuario1[item] * usuario2[item] for item in comunes)

    numerador = sum_prod - (sum1 * sum2 / n)
    denominador = ((sum1_sq - (sum1 ** 2 / n)) * (sum2_sq - (sum2 ** 2 / n))) ** 0.5

    if denominador == 0:
        return 0

    return numerador / denominador

# ==============================
# 4. OBTENER VECINOS POSITIVOS
# ==============================
def obtener_vecinos(datos, usuario_objetivo, k=3):
    vecinos = []

    for otro_usuario in datos:
        if otro_usuario != usuario_objetivo:
            similitud = pearson(datos[usuario_objetivo], datos[otro_usuario])
            if similitud > 0:
                vecinos.append((otro_usuario, similitud))

    vecinos.sort(key=lambda x: x[1], reverse=True)
    return vecinos[:k]

# ==============================
# 5. GENERAR RECOMENDACIONES
# ==============================
def recomendar(datos, usuario_objetivo, k=3):
    vecinos = obtener_vecinos(datos, usuario_objetivo, k)
    totales = {}
    similitudes = {}

    for vecino, similitud in vecinos:
        for item in datos[vecino]:
            if item not in datos[usuario_objetivo]:
                if item not in totales:
                    totales[item] = 0
                    similitudes[item] = 0

                totales[item] += datos[vecino][item] * similitud
                similitudes[item] += similitud

    ranking = []
    for item in totales:
        if similitudes[item] != 0:
            score = totales[item] / similitudes[item]
            ranking.append((item, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

# ==============================
# 6. MOSTRAR USUARIOS
# ==============================
def mostrar_usuarios(datos):
    print("\nUsuarios disponibles:")
    for usuario in datos:
        print(f" - {usuario}")

# ==============================
# 7. MOSTRAR DISTANCIAS MANHATTAN
# ==============================
def mostrar_distancias_manhattan(datos, usuario):
    print(f"\nDistancias Manhattan para {usuario}:")
    lista_distancias = []

    for otro in datos:
        if otro != usuario:
            distancia = manhattan(datos[usuario], datos[otro])
            lista_distancias.append((otro, distancia))

    lista_distancias.sort(key=lambda x: x[1])

    for nombre, distancia in lista_distancias:
        print(f" > {nombre:10} | Distancia: {round(distancia, 2)}")

# ==============================
# 8. MOSTRAR SIMILITUDES PEARSON
# ==============================
def mostrar_similitudes_pearson(datos, usuario):
    print(f"\nSimilitudes de Pearson para {usuario}:")
    lista = []

    for otro in datos:
        if otro != usuario:
            similitud = pearson(datos[usuario], datos[otro])
            lista.append((otro, similitud))

    lista.sort(key=lambda x: x[1], reverse=True)

    for nombre, similitud in lista:
        print(f" > {nombre:10} | Similitud: {round(similitud, 4)}")

# ==============================
# 9. MOSTRAR RECOMENDACIONES CON UMBRAL
# ==============================
def mostrar_recomendaciones_con_umbral(datos, usuario, k=3, umbral=3):
    print(f"\nRecomendaciones para {usuario}")
    print(f"Umbral de recomendación: {umbral}\n")

    recomendaciones = recomendar(datos, usuario, k)

    if not recomendaciones:
        print("No hay recomendaciones para mostrar.")
        return

    print("Evaluación de recomendaciones:")
    for item, score in recomendaciones:
        estado = "ACEPTADA" if score > umbral else "RECHAZADA"
        print(f" > {item:20} | Score estimado: {round(score, 2)} | {estado}")

    print("\nRecomendaciones finales:")
    hay_validas = False

    for item, score in recomendaciones:
        if score > umbral:
            print(f" * {item:20} -> {round(score, 2)}")
            hay_validas = True

    if not hay_validas:
        print("Ninguna recomendación supera el umbral.")

# ==============================
# 10. MENÚ
# ==============================
def mostrar_menu():
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Mostrar usuarios")
    print("2. Ver distancias Manhattan")
    print("3. Ver similitudes de Pearson con todos los usuarios")
    print("4. Ver recomendaciones con umbral")
    print("5. Salir")

# ==============================
# 11. MAIN
# ==============================
def main():
    nombre_archivo = "dataset.csv"
    datos = cargar_datos(nombre_archivo)

    if datos is None:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return

    while True:
        mostrar_menu()
        opcion = input("\nSeleccione una opción: ").strip()

        if opcion == "1":
            mostrar_usuarios(datos)

        elif opcion == "2":
            usuario = input("Ingrese el nombre del usuario: ").strip()
            if usuario in datos:
                mostrar_distancias_manhattan(datos, usuario)
            else:
                print("Error: El usuario no existe.")

        elif opcion == "3":
            usuario = input("Ingrese el nombre del usuario: ").strip()
            if usuario in datos:
                mostrar_similitudes_pearson(datos, usuario)
            else:
                print("Error: El usuario no existe.")

        elif opcion == "4":
            usuario = input("Ingrese el nombre del usuario: ").strip()
            if usuario in datos:
                try:
                    k = int(input("Ingrese el valor de k: "))
                    umbral = float(input("Ingrese el umbral de recomendación: "))
                    mostrar_recomendaciones_con_umbral(datos, usuario, k, umbral)
                except ValueError:
                    print("Error: k debe ser entero y el umbral debe ser numérico.")
            else:
                print("Error: El usuario no existe.")

        elif opcion == "5":
            print("Saliendo del programa...")
            break

        else:
            print("Opción inválida. Intente nuevamente.")

if __name__ == "__main__":
    main()