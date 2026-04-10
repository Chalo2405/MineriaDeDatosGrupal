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

