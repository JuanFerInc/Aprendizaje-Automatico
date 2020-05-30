""" 
dadas 2 listas `a` y `b`, retorna:
    - una lista con todos los elementos en a que NO se encuentran en b
    - una lista con todos los elementos en b que NO se encuentran en a
"""
def diff(A, B):
    setA = set(A)
    setB = set(B)
    en_A_no_en_B = setA - setB
    en_B_no_en_A = setB - setA
    return list(en_A_no_en_B), list(en_B_no_en_A)

def test_diff():
    dominio_posible = [1,2,3]
    dominio_actual = [4,5,1]
    res = diff(dominio_posible, dominio_actual)
    print('valores que no fueron usados en el dataset:', res[0])
    print('valores que son incompatibles:', res[1])