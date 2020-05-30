import math

from teeko import *
from atributos import *
import telemetria


def V(partida, jugador, pesos):
    atributos = jugador.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)
    total = 0
    totalAtributos = 0

    for x in atributos:
        totalAtributos += x * x

    totalPesos = 0
    for j in jugador.pesos:
        totalPesos += j * j

    totalPesos = math.sqrt(totalPesos)
    totalAtributos = math.sqrt(totalAtributos)
    for i, x in enumerate(atributos):
        #total = total + (jugador.pesos[i] * x / (totalPesos * totalAtributos))
        total = total + (jugador.pesos[i] * x)
    
    telemetria.data['valoresTableros'].append(total)
    return total



def tableroFinal(partida, jugador):
	colorContrincante = 'x' if jugador.color == 'o' else 'o'
	
	total = 0

	if partida.termino(jugador.color):
		total = 1
	elif partida.termino(colorContrincante):
		total = -1

	telemetria.data['valoresTableros'].append(total)
	return total


class Jugador:
    def __init__(self, miColor, pesos):
        self.pesos = pesos
        self.tasaAprendizaje = 0.001
        self.color = miColor
    def decTasaApren(self):
        # if(self.tasaAprendizaje > 0.1):
            self.tasaAprendizaje = self.tasaAprendizaje*1
        # else:
        #     self.tasaAprendizaje = 0.01
    def calcularValorTablero(self, atributos):
        total = 0
        for i, x in enumerate(atributos):
            total += self.pesos[i] * x

        return total

    # Calcula el proximo moviento retorna Desde, Hasta
    def calularMejorProximaJugada(self, partida):
        res = partida.obtenerMovimientosPosibles(self.color)
        jugadasPosibles = []

        for elem in res:
            ficha = elem['ficha']
            for mov in elem['movimientosPosibles']:
                tableroProximo = partida.simularMovimiento(ficha, mov)
                if tableroFinal(tableroProximo, self) > 0:
                    return ficha, mov
                valor = V(tableroProximo, self, self.pesos)
                jugadasPosibles.append((ficha, mov, valor))

        # ordeno segun valor de tablero de forma descendente
        jugadasPosibles.sort(key=lambda x: x[2], reverse=True)
        n = len(jugadasPosibles)

        # considero solo el top 25%
        # caso particular el 25% de 4 es 1 -> en ese caso no quiero hacer
        # el sorteo con solo 1 opcion, sino con 2
        # lo mismo para n=3,2

        if n < 5:
            aConsiderar = jugadasPosibles[:2]
        else:
            maxIdx = math.ceil(n * 0.25)
            aConsiderar = jugadasPosibles[:maxIdx]

        return random.choice(aConsiderar)

    # reotrna RANDOM Desde,Hasta
    def calcularJugadaRandom(self, partida):
        res = partida.obtenerMovimientosPosibles(self.color)
        elem = res[random.randrange(0, len(res), 1)]
        return elem['ficha'], elem['movimientosPosibles'][random.randrange(0, len(elem['movimientosPosibles']))]

    def calcularAtributos(self, jugadasNegro, jugadasRojo):

        misJugadas = jugadasNegro if self.color == 'x' else jugadasRojo
        jugadasOponente = jugadasNegro if not self.color == 'x' else jugadasRojo

        return (
            1,

            # casilleros adyacentes que tienen jugadas mias
            calcCuantasAdyacenes(jugadasNegro, jugadasRojo, 'x'),
            # casilleros adyacentes que tienen jugadas de mi oponente
            calcCuantasAdyacenes(jugadasNegro, jugadasRojo, 'o'),

            # calcular dispersion horizontal mia
            calcDispersion(misJugadas, 'horizontal'),
            calcDispersion(misJugadas, 'vertical'),
            # calcular dispersion horizontal mia
            calcDispersion(jugadasOponente, 'horizontal'),
            calcDispersion(jugadasOponente, 'vertical'),

            # cuadrados potenciales mios
            calcPotencialCuadrado(misJugadas),
            # cuadrados potenciales de mi oponente
            calcPotencialCuadrado(jugadasOponente)

        )

    # def actualizarPesosSegunHistorial(self, historial):

    #     # cuando mas turnos le haya llevado ganar la partida
    #     # menor sera el impacto en los nuevos pesos

    #     # a mayor el turno -> menor gamma -> menor impacto

    #     # lo doy vuelta para recorrerlo desde atras
    #     historial.reverse()
    #     for n, partida in enumerate(historial):

    #         gamma = 0.9
    #         errorCuadratico = 0
    #         esElTableroFinal = n == 0
    #         if esElTableroFinal:
    #             # atributos = self.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)
    #             Vent = tableroFinal(partida, self)
    #             Vop = V(partida, self, self.pesos)
    #             errorCuadratico = (Vent-Vop)**2 + errorCuadratico
    #         else:
    #             # la diferencia ahora es que es con los atributos de siguiente tablero
    #             # recordar que como estamoms iterando de atras para adelante,
    #             # el tablero siguiente se encuentra en el indice i-1
    #             partidaSig = historial[n - 1]
    #             Vent = V(partidaSig, self, self.pesos) * gamma
    #             Vop = V(partida, self, self.pesos)  # nota: probar con nuevos Pesos
    #             errorCuadratico = (Vent - Vop) ** 2 + errorCuadratico


    #         atributos = self.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)

    #         # actualizacion
    #         for i, w in enumerate(self.pesos):
    #             self.pesos[i] = w + self.tasaAprendizaje * (Vent - Vop) * atributos[i]
    #     telemetria.data['errorCuadraticoMedio'].append(errorCuadratico)
    #     telemetria.data['pesos'].append(self.pesos.copy())


    def actualizarPesosSegunHistorial(self, historial):

        errorCuadratico = 0

        viejosPesos = self.pesos
        nuevosPesos = self.pesos.copy()
        # lo doy vuelta para recorrerlo desde atras
        historial.reverse()
        for n, partida in enumerate(historial):

            # gamma = 1
            gamma = 0.9
            # gamma = 1/(n+1)

            esElTableroFinal = n == 0
            if esElTableroFinal:
                #atributos = self.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)
                Vent = tableroFinal(partida, self)
                Vop = V(partida, self, nuevosPesos)
                errorCuadratico = (Vent-Vop)**2 + errorCuadratico
            else:
                # la diferencia ahora es que es con los atributos de siguiente tablero
                # recordar que como estamoms iterando de atras para adelante,
                # el tablero siguiente se encuentra en el indice i-1
                partidaSig = historial[n-1]
                Vent = V(partidaSig, self, nuevosPesos) * gamma
                Vop = V(partida, self, nuevosPesos) # nota: probar con nuevos Pesos
                # nota probar con gamma
                errorCuadratico = (Vent - Vop) ** 2 + errorCuadratico

            atributos = self.calcularAtributos(partida.jugadasNegro, partida.jugadasRojo)

            # actualizacion
            aux = nuevosPesos.copy()
            for i, w in enumerate(nuevosPesos):
                aux[i] = w + self.tasaAprendizaje * (Vent - Vop) * atributos[i]
            nuevosPesos = aux

        telemetria.data['errorCuadraticoMedio'].append(errorCuadratico)
        telemetria.data['pesos'].append(nuevosPesos.copy())

        # fin de analizar todo el historial 
        self.pesos = nuevosPesos
