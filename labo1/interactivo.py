import fileinput

from debugger import *

if __name__ == '__main__':

    pesos = [11.060556962194934, -27.840931412514074, -27.731564558494192, -45.99991988132413, -47.04913187746801, 72.80475842853778, 62.24802372659652, 664.8893569903137, -533.301341654264]
    jugadorNegro = Jugador('x', pesos)

    turno = True

    partida = nuevaPartidaRandom()
    historial = []
    debugger = []

    while True:
        mejor = None
        turnoColor, turnoEquipo  = ('x', 'Negras') if turno else ('o', 'Rojas')

        if turno:
            mejor = jugadorNegro.calularMejorProximaJugada(partida)
        else:
            print('Ingrese la siguiente jugada a realizar')
            print('dupla de fila-columna, 4 numeros [0..4] separados por un espacio')
            print('ejemplo: 2 3 2 4 \n')
            text = input(">")
            text = text.split()
            x = list(map(int, text))
            mejor = [ (x[0], x[1]), (x[2],x[3]) ]

        partida.mover(mejor[0], mejor[1])

        if turno:
            partida.tablero.print()

        if partida.termino(turnoColor):
            print('termino la partida, gano:', turnoEquipo)
            break
        
        turno = not turno

    