#Arquivo criado para definir as funções utilizadas na construção da classe Tree

#importando bibliotecas
import numpy as np

#Definir função auxiliar que retorna logx para x > 0, 0 para x == 0 e erro para x < 0
def log(x):
    if x>=0:
        return np.log2(x)
    elif x == 0:
        return 0
    else:
        return ValueError




