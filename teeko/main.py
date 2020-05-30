from debugger import *
from ploter import *
from math import ceil

import telemetria

def simulacro(jugadorNegro, jugadorRojo, cantMaxTurnos, esRandom, aprendeNegras):

	turno = True

	partida = nuevaPartidaRandom()
	historial = []
	debugger = []


	for t in range(cantMaxTurnos):
		
		mejor = None

		if turno:
			jugador = jugadorNegro
			mejor = jugador.calularMejorProximaJugada(partida)
			copia = copy.deepcopy(partida)
			historial.append(copia)
		else:
			jugador = jugadorRojo
			if esRandom:
				mejor = jugador.calcularJugadaRandom(partida)
			else:
				mejor = jugador.calularMejorProximaJugada(partida)

		# copia = copy.deepcopy(partida)
		# debugger.append((copia, mejor[0], mejor[1], jugador.pesos))

		partida.mover(mejor[0], mejor[1])

		if partida.termino(jugador.color):

			copia = copy.deepcopy(partida)
			historial.append(copia)
			
			if turno:
				telemetria.data['numVictorias'] += 1
				telemetria.data['resultadosPartidas'].append(True)
				# si gano y el fue el ultimo que movio
				# entonces tambien la agrego al historial


			else:
				telemetria.data['numDerrotas'] += 1
				telemetria.data['resultadosPartidas'].append(False)


			telemetria.data['cantTurnos'].append(t)
			break
		
		turno = not turno
		
	if 	aprendeNegras:	
		jugadorNegro.actualizarPesosSegunHistorial(historial)

	if not partida.termino(jugador.color):
		telemetria.data['numEmpates'] += 1


def juegoContraAleatorio(cantPartidas, cantMaxTurnos):
	negros = Jugador('x', generarPesosIniciales())

	# guardo los pesos iniciales, para trackear su evolucion
	telemetria.data['pesos'].append(negros.pesos.copy())
	telemetria.data['pesosIniciales'].append(negros.pesos.copy())
	rojos = Jugador('o', generarPesosIniciales())
	numeroADecreTaza = ceil(cantPartidas/30)

	for i in range(0,cantPartidas):
		simulacro(negros, rojos, cantMaxTurnos, True, True)
		# if (i+1)%numeroADecreTaza == 0:
		# 	negros.decTasaApren()

	telemetria.data['pesosFinales'].append(negros.pesos.copy())
	return negros.pesos


def juegoContraSombra(cantPartidas, cantMaxTurnos, cantAntesDeAprender):
	#pesosIniciales = generarPesosIniciales()
	negros = Jugador('x', generarPesosIniciales())
	rojos = Jugador('o', generarPesosIniciales())

	# guardo los pesos iniciales, para trackear su evolucion
	telemetria.data['pesos'].append(negros.pesos.copy())
	telemetria.data['pesosIniciales'].append(negros.pesos.copy())

	pesosViejos = negros.pesos
	numeroADecreTaza = ceil(cantPartidas/30)

	for i in range(1, cantPartidas + 1):
		simulacro(negros, rojos, cantMaxTurnos, esRandom=False, aprendeNegras=True)

		if i % cantAntesDeAprender == 0:
			rojos.pesos = pesosViejos
			pesosViejos = negros.pesos
		if (i+1)%numeroADecreTaza == 0:
			negros.decTasaApren()
	telemetria.data['pesosFinales'].append(negros.pesos.copy())
	return negros.pesos


def unoContraOtro(jugadorNegro, jugadorRojo, cantPartidas, cantMaxTurnos):
	for i in range(cantPartidas):
		simulacro(jugadorNegro, jugadorRojo, cantMaxTurnos, esRandom=False, aprendeNegras=False)


def generarPesosIniciales():
	signo = [1, 1, 1, -1, -1, 1, 1, 1, -1]
	res = []

	for i in signo:
		res.append(i * random.uniform(0.01, 0.08))
		
	return res


if __name__ == '__main__':

	# contra random
	telemetria.data = telemetria.newTelemetriaData()
	pesosFinalesRand = juegoContraAleatorio(cantPartidas=100, cantMaxTurnos=100)
	# input("Press Enter to continue...")
	plot(telemetria.data)

	# contra sombra
	telemetria.data = telemetria.newTelemetriaData()
	pesosFinalesSomb = juegoContraSombra(cantPartidas=1000, cantMaxTurnos=100, cantAntesDeAprender=100)
	# input("Press Enter to continue...")
	plot(telemetria.data)
		
	# 1 vs 1
	telemetria.data = telemetria.newTelemetriaData()
	negro = Jugador('x', pesosFinalesRand)
	rojo = Jugador('o', pesosFinalesSomb)
	unoContraOtro(negro, rojo, cantPartidas=100, cantMaxTurnos=100)
	# input("Press Enter to continue...")
	plot(telemetria.data, plotearPesos=False)

