from teeko import adyacentes

def calcCuantasAdyacenes(jugadasNegro, jugadasRojo, char):
    jugadas = jugadasNegro if char == 'x' else jugadasRojo
    jugadasContrincante = jugadasNegro if not char == 'x' else jugadasRojo

    res = 0

    for jugadaMia in jugadas:
        posicionesAdyacentes = adyacentes(*jugadaMia)
        for pos in posicionesAdyacentes:
            if pos in jugadasContrincante:
                res += 1

    return res

def calcDispersion(jugadas, sentido):
    min, max = 9999999, -1
    
    if sentido == 'horizontal':
        for pos in jugadas:
            min = pos[1] if pos[1] < min else min
            max = pos[1] if pos[1] > min else min
    
    elif sentido == 'vertical':
        for pos in jugadas:
            min = pos[0] if pos[0] < min else min
            max = pos[0] if pos[0] > min else min
    
        
    return max - min

# normalizada
def calcPotencialCuadrado(jugadas):
    res = 0

    for jugada in jugadas:
        posicionesAdyacentes = adyacentes(*jugada)
        for adyacente in posicionesAdyacentes:
            if adyacente in jugadas:
                res += 1

    return res
