# Kernel-PCA by use Gaussian function

import numpy as np
import pandas as pd
#import utility_kpca as ut

# Gaussian Kernel
#def kernel_gauss():
 #   ...
  #  return(
#Kernel-PCA
#def kpca_gauss():
 #   ...
#    return()
# 

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

def load_data():
    guardar_muestras()  # Guardar las primeras 3000 muestras en Data.csv
    data_kpca = aplicar_kpca()  # Aplicar el algoritmo KPCA
    return(x,y)


# Beginning ...
def main():			
    load_data()
		

if __name__ == '__main__':   
	 main()

