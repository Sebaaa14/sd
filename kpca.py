import numpy as np
import pandas as pd
import etl 

def kernel_gauss(x, z, sigma):
    distance = np.linalg.norm(x - z) ** 2 
    return np.exp(-distance / (2 * sigma ** 2))

def kpca_gauss(datos, sigma, k):
    # calculamos la matriz del kernel
    n = datos.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_gauss(datos[i], datos[j], sigma)
    
    uno_n = np.ones((n, n)) / n
    K_centrada = K - uno_n @ K - K @ uno_n + uno_n @ K @ uno_n  # centramos la matriz del kernel

    # valores y vectores propios
    eigvals, eigvecs = np.linalg.eigh(K_centrada)
    indices_ordenados = np.argsort(-eigvals)
    eigvals = eigvals[indices_ordenados]
    eigvecs = eigvecs[:, indices_ordenados]

    # k componentes principales y colocarlos
    eigvecs = eigvecs[:, :k]
    datos_transformados = K_centrada @ eigvecs

    return datos_transformados


def lectura():
    df = pd.read_csv("archivos_nuevos/DataIG.csv", header=None)
    return df

def seleccionar_muestras():
    M = 3000
    df = lectura()  # Cargar el archivo
    muestras = df.iloc[:M]  # Seleccionar las primeras 3000 filas
    return muestras

def guardar_muestras():
    muestras = seleccionar_muestras()
    muestras.to_csv("archivos_nuevos/Data.csv", header=None, index=False)

def aplicar_kpca(config):
    data = pd.read_csv("archivos_nuevos/Data.csv", header=None).values
    sigma = int(config[0][4])
    k = int(config[0][5])
    data_transformada = kpca_gauss(data, sigma, k)
    pd.DataFrame(data_transformada).to_csv("archivos_nuevos/DataKpca.csv", header=None, index=False)
    return data_transformada

def load_data():
    guardar_muestras()
    config = etl.config()
    data_kpca = aplicar_kpca(config)
    return data_kpca

# Beginning ...
def main():			
    load_data()

if __name__ == '__main__':   
    main()

#corazoncito <3