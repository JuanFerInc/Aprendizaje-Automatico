from teeko import *
from jugador import *

def calcFlechitaChar(desde, hasta):
    # desde
    dy = desde[0]
    dx = desde[1]
    # hasta
    hy = hasta[0]
    hx = hasta[1]
    # delta
    delta_x = hx - dx
    delta_y = hy - dy
    delta = (delta_y, delta_x) 
    
    # un mov son 8 casos: N, S, E, W, NE, NW, SE, SW
    
    if delta == (-1,0): # N
        return str('⭡')
        
    if delta == (1,0): # S
        return str('⭣')
    
    if delta == (0,1): # E
        return str('⭢')
    
    if delta == (0,-1): # W
        return str('⭠')
    
    if delta == (1,1): # SE
        return str('⭨')
    
    if delta == (1,-1): # SW
        return str('⭩')
    
    if delta == (-1,1): # NE
        return str('⭧')
    
    if delta == (-1,-1): # NW
        return str('⭦')

def print_debugger(debugger):
    for elem in debugger:
        
        partida = elem[0]
        desde = elem[1]
        hasta = elem[2]
        pesos = elem[3]

        hastaFila = hasta[0]
        hastaColumna = hasta[1]
        flechita = calcFlechitaChar(desde, hasta)
        print('desde:', desde, 'hasta:', hasta)
        print('pesos  =', pesos)
        print('negras =', partida.jugadasNegro)
        print('rojas  =', partida.jugadasRojo)
        partida.tablero.casilleros[hastaFila][hastaColumna] = flechita
        partida.tablero.print()

def debuggear_partida():
    # cargamos las variables necesarias
    # para recrear el escenario que se quiere debuggear
	pesos  = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
	jugadasNegro = [(2, 1), (3, 1), (3, 2), (3, 3)]
	jugadasRojo = [(1, 0), (0, 2), (1, 2), (2, 4)]
	
	# load partida
	partida = recrearPartida(jugadasNegro, jugadasRojo)
	jugador = Jugador('x', pesos)

	mejor = jugador.calularMejorProximaJugada(partida)
	
	debugger = []
	copia = copy.deepcopy(partida)
	debugger.append((copia, mejor[0], mejor[1], jugador.pesos))

	print_debugger(debugger)

def calcAttrsEjemplo():
    # cargamos las variables necesarias
    # para recrear el escenario que se quiere debuggear
    pesos = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    for x in range(4):
        # load partida
        partida = nuevaPartidaRandom()
        jugador = Jugador('x', pesos)

        atributos = jugador.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)
        partida.tablero.print()
        print('Los atributos son:', atributos)

if __name__ == "__main__":
    debuggear_partida()