#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import pandas as pd
import numpy as num
#import utility_etl  as ut

# Load parameters from config.csv
def config(): 
    config = pd.read_csv("config/config.csv", header=None)
    return config

def lectura():
  df = pd.read_csv("data/KDDTrain.txt", sep=",", header=None)
  return df

def preprocess_data(df):
    # Convertir las columnas 2, 3 y 4 (índices 1, 2 y 3) a enteros
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).astype(int)  # Columna 2
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce').fillna(0).astype(int)  # Columna 3
    df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce').fillna(0).astype(int)  # Columna 4
    return df

#Creamos una copia para no tener que correr el df read siempre (esto porque si no se corre el df cada vez, se bugea, con la copia no)

def definicion(df):
  dfKDD = df.copy()
  #HAY QUE AGREGAR UN INDICE A CADA FILA ANTES DE DIVIDIRLOS


  print(f"filas y columnas en dfKDD: ({dfKDD.shape[0]}, {dfKDD.shape[1]})")

  #elimina la ultima columna
  dfKDD = dfKDD.drop(dfKDD.columns[-1], axis=1)



  print(f"filas y columnas en dfKDD dsps de quitar la 1ra: ({dfKDD.shape[0]}, {dfKDD.shape[1]})")

  #--------------------------------------------------------------------------------------------------------------------


  #ASIGNANDO VALOR NUMERICO SEGUN EL TIPO DE ATAQUE

  #Clase 1
  normal_attack= ['normal']
  #Clase 2
  dos_attacks = ['neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 'apache2', 'processtable', 'mailbomb', 'udpstorm']
  #Clase 3
  probe_attacks = ['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan']

  #Definimos funcion para clasificar ataques (esto se puede ir al utils_etl)
  def clasificar_attack(attack):
      if attack in normal_attack:
          return 1
      elif attack in dos_attacks:
          return 2
      elif attack in probe_attacks:
          return 3
      else:
          return 0

  #Aplicamos la funcion a la penultima col del dfKDD
  dfKDD.iloc[:, -1] = dfKDD.iloc[:, -1].apply(clasificar_attack)


  print(dfKDD)

  return dfKDD

#Primero separamos las clases según el tipo de ataque
def separacion(dfKDD):
    class1 = dfKDD[dfKDD.iloc[:, -1] == 1]
    class2 = dfKDD[dfKDD.iloc[:, -1] == 2]
    class3 = dfKDD[dfKDD.iloc[:, -1] == 3]

    #se elimina la penultima columna que correspondia al tipo de ataque para dividir los archivos
    class1 = class1.drop(class1.columns[-1], axis=1)
    class2 = class2.drop(class2.columns[-1], axis=1)
    class3 = class3.drop(class3.columns[-1], axis=1)


    #Guardamos los nuevos archivos csv
    class1.to_csv('archivos_nuevos/class1.csv', index=False, header=False)
    class2.to_csv('archivos_nuevos/class2.csv', index=False, header=False)
    class3.to_csv('archivos_nuevos/class3.csv', index=False, header=False)

    #--------------------------------------------------------------------------

    #comprueba el numero de filas y columnas
    class1_df = pd.read_csv('archivos_nuevos/class1.csv', header=None)
    class2_df = pd.read_csv('archivos_nuevos/class2.csv', header=None)
    class3_df = pd.read_csv('archivos_nuevos/class3.csv', header=None)
    print(f"filas y columnas en class1.csv: ({class1_df.shape[0]}, {class1_df.shape[1]})")
    print(f"filas y columnas en class2.csv: ({class2_df.shape[0]}, {class2_df.shape[1]})")
    print(f"filas y columnas en class3.csv: ({class3_df.shape[0]}, {class3_df.shape[1]})")

    return class1_df, class2_df, class3_df

# Cargar los archivos de índices
def lectura_idx():
    idx1 = pd.read_csv('data/idx_class1.csv', header=None)[0].tolist()
    idx2 = pd.read_csv('data/idx_class2.csv', header=None)[0].tolist()
    idx3 = pd.read_csv('data/idx_class3.csv', header=None)[0].tolist()

    # Verificar si algún índice está fuera de rango (mayor que el tamaño esperado del archivo completo)
    max_idx = 25192  # Número total de filas en KDDTrain.txt
    idx1 = [idx for idx in idx1 if idx < max_idx]
    idx2 = [idx for idx in idx2 if idx < max_idx]
    idx3 = [idx for idx in idx3 if idx < max_idx]

    print("largo idx1", len(idx1))
    print("largo idx2", len(idx2))
    print("largo idx3", len(idx3))

    return idx1, idx2, idx3

# Ejemplo de uso con tus archivos de clase e índice
# seleccionMuestra(class1_df, class2_df, class3_df, idx1, idx2, idx3)

#----------------------------------------------------------------------------------------------------------------
def seleccionMuestra2(class1_df, class2_df, class3_df, idx1, idx2, idx3):
    # Unir los tres archivos idx en uno solo (idx_nuevo)
    idx_nuevo = idx1 + idx2 + idx3
    idx_unicos = list(set(idx_nuevo))
    print(f"Total de índices únicos en idx_nuevo: {len(idx_unicos)}")
    
    # Crear columna de clase en cada DataFrame de clase
    class1_df[len(class1_df.columns)] = 1
    class2_df[len(class2_df.columns)] = 2
    class3_df[len(class3_df.columns)] = 3
    
    # Concatenar los DataFrames de clases en uno solo
    class_nuevo = pd.concat([class1_df, class2_df, class3_df], ignore_index=True)
    max_filas_class_nuevo = class_nuevo.shape[0]
    print(f"Total de filas esperado: {class1_df.shape[0] + class2_df.shape[0] + class3_df.shape[0]}")
    print(f"Total de filas en class_nuevo después de concatenar: {class_nuevo.shape[0]}")

    # Filtrar índices válidos y fuera de rango
    valid_idx_nuevo = [idx for idx in idx_unicos if idx < max_filas_class_nuevo]
    out_of_range_idx = [idx for idx in idx_unicos if idx >= max_filas_class_nuevo]
    
    # Identificar la fuente de los índices fuera de rango
    out_of_range_sources = {
        'idx_class1': [idx for idx in idx1 if idx in out_of_range_idx],
        'idx_class2': [idx for idx in idx2 if idx in out_of_range_idx],
        'idx_class3': [idx for idx in idx3 if idx in out_of_range_idx],
    }
    print("Índices fuera de rango en idx_nuevo:", out_of_range_idx)
    print("Fuente de índices fuera de rango:", out_of_range_sources)
    print(f"Total de índices válidos en idx_nuevo: {len(valid_idx_nuevo)}")

    # Seleccionar las filas correspondientes de class_nuevo usando los índices válidos
    selected_rows_class_nuevo = class_nuevo.iloc[valid_idx_nuevo]
    print(f"Filas seleccionadas de class_nuevo: ({selected_rows_class_nuevo.shape[0]}, {selected_rows_class_nuevo.shape[1]})")

    # Guardar el archivo final
    selected_rows_class_nuevo.to_csv('archivos_nuevos/dataClass.csv', index=False, header=False)
    return selected_rows_class_nuevo



#--------------------------------------------------------------------------------------------------------


def seleccionMuestra(class1_df, class2_df, class3_df, idx1, idx2, idx3):

     # Verificar la cantidad de filas en class1 y los índices (LAMENTABLEMENTE EL IDX TIENE NUMEROS DE INDICES QUE NO EXISTEN, ASI QUE HAY QUE REVISAR)
     print(f"Total de filas en class1: {class1_df.shape[0]}")
     print(f"Total de índices en idx_class1: {len(idx1)}")

     # Asegúrate de que los índices sean válidos (es decir, estén dentro del rango de filas de class1_df)
     valid_idx_class1 = [idx for idx in idx1 if idx < len(class1_df)]

     #ESTA GUARDANDO BIEN LAS QUE GUARDA SOLO QUE TOMA 1 DE MAS EJEMPLO: LA 7907 EN REALIDAD ES LA 7908
     # Seleccionar las filas correspondientes en class1 usando los índices válidos
     selected_rows_class1 = class1_df.iloc[valid_idx_class1]

     # Guardar las filas seleccionadas en un nuevo archivo CSV sin los índices
     selected_rows_class1.to_csv('archivos_nuevos/class1_selected.csv', index=False, header=False)

     print(selected_rows_class1)

     # Verificar el archivo resultante (GUARDA MENOS Q HAY INDICES INEXISTENTES)
     print(f"Filas y columnas en class1_selected.csv: ({selected_rows_class1.shape[0]}, {selected_rows_class1.shape[1]})")

     #--------------------------------------------------------------------------------------------------------------------

     print(f"Total de filas en class2: {class2_df.shape[0]}")
     print(f"Total de índices en idx_class2: {len(idx2)}")

     valid_idx_class2 = [idx for idx in idx2 if idx < len(class2_df)]

     selected_rows_class2 = class2_df.iloc[valid_idx_class2]

     selected_rows_class2.to_csv('archivos_nuevos/class2_selected.csv', index=False, header=False)

     print(selected_rows_class2)

     print(f"Filas y columnas en class2_selected.csv: ({selected_rows_class2.shape[0]}, {selected_rows_class2.shape[1]})")


     #--------------------------------------------------------------------------------------------------------------------

     print(f"Total de filas en class3: {class3_df.shape[0]}")
     print(f"Total de índices en idx_class3: {len(idx3)}")

     valid_idx_class3 = [idx for idx in idx3 if idx < len(class3_df)]

     selected_rows_class3 = class3_df.iloc[valid_idx_class3]

     selected_rows_class3.to_csv('archivos_nuevos/class3_selected.csv', index=False, header=False)

     print(selected_rows_class3)

     print(f"Filas y columnas en class3_selected.csv: ({selected_rows_class3.shape[0]}, {selected_rows_class3.shape[1]})")



     #--------------------------------------------------------------------------------------------------------------------


     selected_rows_class1 = pd.read_csv('archivos_nuevos/class1_selected.csv', header=None)
     selected_rows_class2 = pd.read_csv('archivos_nuevos/class2_selected.csv', header=None)
     selected_rows_class3 = pd.read_csv('archivos_nuevos/class3_selected.csv', header=None)

     # Añadir la columna con el número de clase
     selected_rows_class1[selected_rows_class1.shape[1]] = 1  # Clase 1: Normal
     selected_rows_class2[selected_rows_class2.shape[1]] = 2  # Clase 2: DOS
     selected_rows_class3[selected_rows_class3.shape[1]] = 3  # Clase 3: Probe

     #print("selección1",selected_rows_class1)

     # Unir las tres muestras seleccionadas en un solo DataFrame
     dataClass = pd.concat([selected_rows_class1, selected_rows_class2, selected_rows_class3], ignore_index=True)

     #print("data class", dataClass)

     #desordenar los datos
     #dataClass = dataClass.sample(frac=1).reset_index(drop=True)
     #print("data class desordenado", dataClass)

     dataClass.to_csv('archivos_nuevos/dataClass.csv', index=False, header=False)

     print(f"Filas y columnas en DataClass.csv: ({dataClass.shape[0]}, {dataClass.shape[1]})")


def correrETL():
    df = lectura()
    df = preprocess_data(df)
    dfKDD = definicion(df)
    print("DFKDD LUEGP DE DEFINICION",len(dfKDD))
    class1_df, class2_df, class3_df = separacion(dfKDD)
    idx1, idx2, idx3 = lectura_idx()
    return seleccionMuestra2(class1_df, class2_df, class3_df, idx1, idx2, idx3)

def main():
    correrETL()       
   
      
if __name__ == '__main__':   
	 main()

