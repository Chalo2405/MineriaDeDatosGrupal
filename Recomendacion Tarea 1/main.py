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
                usuario = fila['Usuario']
                datos[usuario] = {}
                for item in fila:
                    if item != 'Usuario' and fila[item].strip() != "":
                        try:
                            datos[usuario][item] = float(fila[item])
                        except ValueError:
                            continue 
        return datos
    except FileNotFoundError:
        return None

# ==============================
# 2. CÁLCULO DE MANHATTAN
# ==============================
def manhattan(usuario1, usuario2):
    comunes = [item for item in usuario1 if item in usuario2]
    if not comunes:
        return float('inf')
    distancia = sum(abs(usuario1[item] - usuario2[item]) for item in comunes)
    return distancia

# ==============================
# 3. CÁLCULO DE PEARSON
# ==============================
def pearson(usuario1, usuario2):
    comunes = [item for item in usuario1 if item in usuario2]
    n = len(comunes)
    if n == 0: return 0

    sum1 = sum(usuario1[item] for item in comunes)
    sum2 = sum(usuario2[item] for item in comunes)
    sum1_sq = sum(usuario1[item]**2 for item in comunes)
    sum2_sq = sum(usuario2[item]**2 for item in comunes)
    sum_prod = sum(usuario1[item] * usuario2[item] for item in comunes)

    numerador = sum_prod - (sum1 * sum2 / n)
    denominador = ((sum1_sq - (sum1**2 / n)) * (sum2_sq - (sum2**2 / n))) ** 0.5

    if denominador == 0: return 0
    return numerador / denominador

# ==============================
# 4. KNN (LOS 3 VECINOS MÁS CERCANOS)
# ==============================
def obtener_vecinos(datos, usuario_objetivo, k=3):
    lista_similitudes = []
    for otro_usuario in datos:
        if otro_usuario != usuario_objetivo:
            sim = pearson(datos[usuario_objetivo], datos[otro_usuario])
            lista_similitudes.append((otro_usuario, sim))
    
    # Ordenamos por similitud Pearson (los más altos primero)
    lista_similitudes.sort(key=lambda x: x[1], reverse=True)
    return lista_similitudes[:k]

# 5. GENERAR RECOMENDACIONES
# ==============================
def recomendar(datos, usuario_objetivo, k=3):
    vecinos_cercanos = obtener_vecinos(datos, usuario_objetivo, k)
    totales = {}
    similitudes = {}

    for vecino, similitud in vecinos_cercanos:
        if similitud <= 0: continue
        for item in datos[vecino]:
            if item not in datos[usuario_objetivo]:
                totales.setdefault(item, 0)
                similitudes.setdefault(item, 0)
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
# 6. BLOQUE PRINCIPAL (MAIN)
# ==============================
def main():
    print("--- RecoMusic KNN (Pearson + Manhattan) ---\n")
    
    nombre_archivo = "dataset.csv" 
    datos = cargar_datos(nombre_archivo)

    if datos is None:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return

    usuario = input("Ingrese el nombre del usuario: ")
    if usuario not in datos:
        print("Error: El usuario no existe.")
        return

    # A. DISTANCIAS CON MANHATTAN (TABLA COMPLETA ORDENADA)
    print(f"\n1. DISTANCIAS MANHATTAN (De menor a mayor distancia):")
    lista_m = []
    for otro in datos:
        if otro != usuario:
            dist = manhattan(datos[usuario], datos[otro])
            lista_m.append((otro, dist))
    
    lista_m.sort(key=lambda x: x[1]) # El 0 es el más cercano
    for nombre, d in lista_m:
        print(f" > {nombre:10} | Distancia: {round(d, 2)}")

    # B. LOS 3 VECINOS MÁS CERCANOS (KNN - PEARSON)
    print(f"\n2. LOS 3 VECINOS MÁS CERCANOS (Seleccionados por Pearson):")
    mis_vecinos = obtener_vecinos(datos, usuario, k=3)
    for v, sim in mis_vecinos:
        print(f" >> Vecino: {v:10} | Similitud: {round(sim, 4)}")

    # C. RECOMENDACIONES
    print("\n3. RECOMENDACIONES FINALES:")
    recomendaciones = recomendar(datos, usuario, k=3)
    
    if not recomendaciones:
        print("No hay recomendaciones nuevas para mostrar.")
    else:
        for item, score in recomendaciones:
            print(f" * {item:15} -> Puntuación estimada: {round(score, 2)}")

if __name__ == "__main__":
    main()