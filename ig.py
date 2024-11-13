import matplotlib.pyplot as plt #importación para el grafico :)
import numpy as num
import pandas as pd
import etl #importación del etl para correr el main

# NORMALIZAR CON SIGMOIDAL
def normalizacion_sigmoidal(datos):

    epsilon = 1e-10
    media = num.mean(datos)

    #CALCULA DES ESTANDAR
    desviacion_estandar = num.std(datos)
    
    #SE NORMALIZA
    datos_normalizados = 1 / (1 + num.exp(-(datos - media) / (desviacion_estandar + epsilon)))
    
    return datos_normalizados



#----------------------------------------------------------------------------------------------------------------



# VECTORES EMBEDDING
def generar_vectores_embedding(datos, m, tau):

    total_datos = len(datos)
    cantidad_embedding = total_datos - (m - 1) * tau
    vectores_embedding = []


    for i in range(cantidad_embedding):
        vector = [datos[i + j * tau] for j in range(m)]
        vectores_embedding.append(vector)
    return num.array(vectores_embedding)



#----------------------------------------------------------------------------------------------------------------



#EMBEDDING A DISCRETO
def mapear_a_simbolos(vectores_embedding, c):

    vectores_simbolos = []

    for vector in vectores_embedding:
        simbolos = num.round(c * vector + 0.5).astype(int)
        vectores_simbolos.append(simbolos)


    return num.array(vectores_simbolos)


#----------------------------------------------------------------------------------------------------------------


#ENTROPIA NORMALIZADA
def calcular_entropia(probabilidades, c, m):

    patrones_maximos = c ** m
    entropia = -num.sum(probabilidades * num.log2(probabilidades + 1e-10))
    entropia_normalizada = entropia / num.log2(patrones_maximos)

    return entropia_normalizada



#----------------------------------------------------------------------------------------------------------------



#ENTROPIA DISPERCION PRINCIPAL
def entropia_dispersion(datos, c, m, tau):

    #NORMALIZACIÓN
    datos = normalizacion_sigmoidal(datos)

    #EMBEDDING VECTORES
    vectores_embedding = generar_vectores_embedding(datos, m, tau)

    #SIMBOLOS
    simbolos = mapear_a_simbolos(vectores_embedding, c)


    #SIMBOLOS A K
    valores_k = convertir_simbolos_a_k(simbolos, c, m)
    valores_k_a_planos = valores_k.reshape(-1)

    #FRECUENCIA
    patrones_unicos, cuentas = num.unique(valores_k_a_planos, return_counts=True)
    total_patrones = len(valores_k_a_planos)
    probabilidades = cuentas / total_patrones

    #ENTROPIA PERO NORMALIZADA
    valor_entropia = calcular_entropia(probabilidades, c, m)


    return valor_entropia



#----------------------------------------------------------------------------------------------------------------



# CONVERSION DE SIMBOLOS, PA QUE NO MUERA LA ED
def convertir_simbolos_a_k(simbolos, c, m):
    valores_k = []
    simbolos_menos_1 = simbolos - 1
    potencia_c = [c ** i for i in range(m)]


    for simbolo in simbolos_menos_1:
        valor = num.dot(simbolo, potencia_c)
        valores_k.append(1 + valor)

        
    return valores_k



#----------------------------------------------------------------------------------------------------------------



#ENTROPIA CONDICIONAL
def entropia_condicional(X, Y, c, m):
    N = len(X)

    #NORMALIZACIÓN
    X = normalizacion_sigmoidal(X)

    #BINS
    num_bins = int(num.sqrt(N))
    bins = num.linspace(num.min(X), num.max(X), num_bins + 1)
    datos_binned = num.digitize(X, bins) - 1

    bins_unicos = num.unique(datos_binned)
    valor_entropia_condicional = 0


    for bin_value in bins_unicos:
        mascara_bin = (datos_binned == bin_value)
        if num.sum(mascara_bin) == 0:
            continue
        frecuencias_clases = Y[mascara_bin].value_counts(normalize=True).values
        valor_entropia_condicional += (num.sum(mascara_bin) / N) * calcular_entropia(frecuencias_clases, c, m)



    return valor_entropia_condicional



#----------------------------------------------------------------------------------------------------------------

# CALCULA EL IG Y MUESTRA LOS GRAFICOS
def calcular_ganancia_informacion(datos, config):
    
    m = int(config[0][0])
    tau = int(config[0][1])
    c = int(config[0][2])
    top_K = int(config[0][3])

    
    X = datos.iloc[:, :-1]
    Y = datos.iloc[:, -1]


    #ENTROPIA DISPERSION
    probabilidades_clase = Y.value_counts(normalize=True).values
    entropia_clase = calcular_entropia(probabilidades_clase, c, m)

    #GI POR CARACTERISTICA
    valores_ig = []
    
    for i in range(X.shape[1]):
        entropia_caracteristica = entropia_condicional(X.iloc[:, i], Y, c, m)
        ig = entropia_clase - entropia_caracteristica
        valores_ig.append(ig)


    #GRAFICO IG POR CARACT.
    plt.figure()
    plt.stem(range(len(valores_ig)), valores_ig, basefmt=" ", linefmt="g", markerfmt="go")
    plt.xlabel('Number of Variable')
    plt.ylabel('Inform. Gain')
    plt.title('Information Gain')
    plt.show()

    #GRAFICO DE CARACTERISTICAS RELEVANTES
    indices_top_ig = num.argsort(valores_ig)[-top_K:][::-1]
    valores_top_ig = num.array(valores_ig)[indices_top_ig]
    plt.figure()
    plt.stem(range(len(valores_top_ig)), valores_top_ig, basefmt=" ", linefmt="g", markerfmt="go")
    plt.xlabel('Number of Variable')
    plt.ylabel('Inform. Gain')
    plt.title('Information Gain')
    plt.show()

    #GUARDAR ARCHIVOS DE SALIDA
    indices_top_ig = num.argsort(valores_ig)[-top_K:][::-1]
    indices_variables = pd.DataFrame(indices_top_ig, columns=["Idx"])
    indices_variables.to_csv("archivos_nuevos/Idx_variable.csv", index=False)

    datos_relevantes = X.iloc[:, indices_top_ig]
    datos_relevantes.to_csv("archivos_nuevos/DataIG.csv", index=False)


#----------------------------------------------------------------------------------------------------------------


#CARGA DATOS Y EJECUCIÓN ETL
def correrCodigo():
    config = etl.config()
    datos = etl.correrETL()
    calcular_ganancia_informacion(datos, config)
    print("IG ejecutado correctamente")

if __name__ == '__main__':
    correrCodigo()
