import copy
import random

# constantes
inf = float('Inf')
numFilas = 5
numColumnas = 5
todosLosCasilleros = [ (fila, columna) for columna in range( numColumnas ) for fila in range( numFilas ) ]

# ┌───┬───┬───┬───┬───┐
# │   │ x │   │   │   │
# ├───┼───┼───┼───┼───┤
# │ o │   │ o │ x │   │
# ├───┼───┼───┼───┼───┤
# │ x │   │   │ x │   │
# ├───┼───┼───┼───┼───┤
# │   │   │   │   │   │
# ├───┼───┼───┼───┼───┤
# │   │   │   │   │ x │
# └───┴───┴───┴───┴───┘

def adyacentes(fila, columna):
	res = []

	# la fila de arriba
	filaSuperior = fila - 1
	filaSuperiorDentroDeRango = filaSuperior >= 0
	if filaSuperiorDentroDeRango:
		for x in range(max(columna - 1, 0), min(columna + 2, numColumnas)):
			res.append((filaSuperior, x))

	# la fila de abajo
	filaInferior = fila + 1
	filaInferiorDentroDeRango = filaInferior < numFilas
	if filaInferiorDentroDeRango:
		for x in range(max(columna - 1, 0), min(columna + 2, numColumnas)):
			res.append((filaInferior, x))

	# el casillero de la izq
	if columna - 1 >= 0:
		res.append((fila, columna - 1))

	# el casillero de la der
	if columna + 1 < numFilas:
		res.append((fila, columna + 1))

	return res

class Tablero:
	# " " ~ no judada
	# "x"
	# "o"
	def __init__(self):
		self.casilleros = [ [ ' ' for y in range( numColumnas ) ] for x in range( numFilas ) ]

	# retorna True si la posicion representa a un casillero valido
	def estaDentroDeRango(self, fila, columna):
		return (0 <= fila < numFilas) and (0 <= columna < numColumnas)

	# retorna True si el casillero esta ocupado
	def ocupado(self, fila, columna):
		return self.casilleros[fila][columna] != ' '

	# retorna True si ambas posiciones son adyacentes
	# ojo que no se fija que las posiciones sean validas/esten dentro de rango
	# arg pos1, pos2 : duplas del tipo (fila, columna)
	def sonAdyacentes(self, pos1, pos2):
		return (abs(pos1[0] - pos2[0]) <= 1) and (abs(pos1[1] - pos2[1]) <= 1) and (pos1 != pos2)

	# retorna una lista de duplas (fila, columna)
	# no incluye al punto en si
	# 
	# ejemplo:
	# 		┌───┬───┬───┬───┬───┐
	# 		│   │   │   │   │   │
	# 		├───┼───┼───┼───┼───┤
	# 		│   │   │   │   │   │
	# 		├───┼───┼───┼───┼───┤
	# 		│   │ A │ A │ A │   │
	# 		├───┼───┼───┼───┼───┤
	# 		│   │ A │ . │ A │   │
	# 		├───┼───┼───┼───┼───┤
	# 		│   │ A │ A │ A │   │
	# 		└───┴───┴───┴───┴───┘
	def obtenerAdyacentes(self, fila, columna):
		return adyacentes(fila, columna)

	def print(self):
		# la parte de arriba del tablero
		str = '┌' + '───┬' * (numColumnas - 1) + '───┐'
		print(str)

		# cada fila
		for i, fila in enumerate(self.casilleros):
			str = '│'
			for casillero in fila:
				str += ' ' + casillero + ' │'
			print(str)
			
			if i < numFilas - 1:
				str = '├' + '───┼' * (numColumnas - 1) + '───┤'
				print(str)

		# la parte de abajo del tablero
		str = '└' + '───┴' * (numColumnas - 1) + '───┘'
		print(str)





class Partida:
	def __init__(self):
		# empieza jugando negro
		# negro ~ 'x'
		# rojo ~ 'o'

		# array de duplas (fila, columna) que simbolizan
		# las jugadas hechas respectivamente
		self.jugadasNegro, self.jugadasRojo = [], []
		# inicialmente tienen todo el tablero libre
		self.casillerosValidosNegro = todosLosCasilleros.copy()
		self.casillerosValidosRojo = todosLosCasilleros.copy()
		self.tablero = Tablero()


	# es valido poner esa ficha en ese casillero sii:
	# 1. no habia jugado sus 4 fichas aun
	# 2. esta dentro de rango
	# 3. ese casillero estaba vacio
	def poner(self, char, fila, columna):

		jugadas = self.jugadasNegro if char == 'x' else self.jugadasRojo

		# 1.
		yaTieneTodasSusFichasEnJuego = len(jugadas) == 4
		if yaTieneTodasSusFichasEnJuego:
			return

		# 2.
		estaDentroDeRango = self.tablero.estaDentroDeRango(fila, columna)
		if not estaDentroDeRango:
			return

		# 3.
		estaOcupado = self.tablero.ocupado(fila, columna)
		if estaOcupado:
			return

		# ok
		jugadas.append((fila, columna))
		self.tablero.casilleros[fila][columna] = char
		# self.actualizarCasillerosValidos(char, fila, columna)

	# puede mover sii:
	# 1. ambos estan dentro de rango
	# 2. el casillero que selecciono tiene una ficha
	# 3. el casillero a donde quiere mover:
	# 	3.1 es adyacente
	# 	3.2 esta libre
	def mover(self, desde, hasta):

		# 1.
		ambosDentroDeRango = (self.tablero.estaDentroDeRango(*desde) and self.tablero.estaDentroDeRango(*hasta))
		if not ambosDentroDeRango:
			return
		
		# 2.
		tieneUnaFicha = self.tablero.ocupado(*desde)
		if not tieneUnaFicha:
			return

		# 3.1
		sonAdyacentes = self.tablero.sonAdyacentes(desde, hasta)
		if not sonAdyacentes:
			return

		# 3.2
		estaOcupado = self.tablero.ocupado(*hasta)
		if estaOcupado:
			return

		# todo ok
		# lo puede mover para ahi

		# obtengo el char 'x' o bien 'o'
		char = self.tablero.casilleros[desde[0]][desde[1]]
		jugadas = self.jugadasNegro if char == 'x' else self.jugadasRojo
		# actualizo las jgadas
		jugadas.remove(desde)
		jugadas.append(hasta)
		# actualizo el tablero
		self.tablero.casilleros[desde[0]][desde[1]] = ' '
		self.tablero.casilleros[hasta[0]][hasta[1]] = char


	# retorna una lista de casilleros posibles validos
	# para mover su ficha
	def obtenerMovimientosPosibles(self, char):
		jugadas = self.jugadasNegro if char == 'x' else self.jugadasRojo
		res = []
		for ficha in jugadas:
			adyacentes = self.tablero.obtenerAdyacentes(*ficha)
			adyacentesNoOcupados = []
			for adyacente in adyacentes:
				if not self.tablero.ocupado(*adyacente):
					# adyacentesNoOcupados += adyacente
					adyacentesNoOcupados.append(adyacente)
			if len(adyacentesNoOcupados) > 0:
				res.append({'ficha':ficha, 'movimientosPosibles':adyacentesNoOcupados})
		return res

	def simularMovimiento(self, desde, hasta):
		copia = copy.deepcopy(self)
		copia.mover(desde, hasta)
		return copia

	def hayAlineacionHorizontal(self,color):
		jugadas = self.jugadasNegro if color == 'x' else self.jugadasRojo

		menor = None
		for x in jugadas:
			if menor == None:
				menor = x
			else:
				if x[1] < menor[1]:
					menor = x
		for x in range(1,4):
			if (menor[0], menor[1]+x) not in jugadas:
				return False
		return True

	def hayAlineacionVertical(self,color):
		jugadas = self.jugadasNegro if color == 'x' else self.jugadasRojo

		menor = None
		for x in jugadas:
			if menor == None:
				menor = x
			else:
				if x[0] < menor[0]:
					menor = x
		for x in range(1, 4):
			if (menor[0]+x, menor[1]) not in jugadas:
				return False
		return True

	def hayAlineacionDiagonal(self,color):
		jugadas = self.jugadasNegro if color == 'x' else self.jugadasRojo

		menor = None
		for x in jugadas:
			if menor == None:
				menor = x
			else:
				if x[0] < menor[0]:
					menor = x
		if(menor[1] > 2):
			for x in range(1, 4):
				if (menor[0] + x, menor[1] - x) not in jugadas:
					return False
		else:
			for x in range(1, 4):
				if (menor[0] + x, menor[1] + x) not in jugadas:
					return False

		return True

	# retorna True sii:
	# 1. hay un cuadrado
	# 2. estan alieneados
	#  2.1 verticalmente
	#  2.2 horizontalmente
	#  2.3 diagonalmente
	def termino(self, char):
		return self.hayCuadrado(char) or self.hayAlineacionVertical(char) or self.hayAlineacionHorizontal(char) or self.hayAlineacionDiagonal(char)

	def hayCuadrado(self, char):
		jugadas = self.jugadasNegro if char == 'x' else self.jugadasRojo
		# busco el mas cercano al "origen"
		minY, minX  = jugadas[0][0], jugadas[0][1]
		for jugada in jugadas[1:]:
			if jugada[0] <= minY and jugada[1] <= minX:
				minY, minX  = jugada[0], jugada[1]

		# pre-check: tiene que como minimo tener un margen de 1
		# con el borde de abajo y el de la derecha
		# hayMargenDerecha = minX < numColumnas -1
		# hayMargenAbajo = minY < numFilas -1
		# if (not hayMargenDerecha) or (not hayMargenAbajo):
		# 	return False

		# ahora me fijo si estan:
		#  - uno a la derecha
		#  - uno abajo
		#  - uno en diagonal abajo derecha

		hayUnoALaDerecha = (minY, minX + 1) in jugadas
		hayUnoAbajo = (minY + 1, minX) in jugadas
		hayUnoEnDiagonalAbajoDerecha = (minY + 1, minX + 1) in jugadas

		return hayUnoALaDerecha and hayUnoAbajo and hayUnoEnDiagonalAbajoDerecha




# retorna una partida random NO FINAL
def nuevaPartidaRandom():	
	while True:
		lugaresElegidos = []
		for i in range(9):
			size = len(lugaresElegidos)
			while size == len(lugaresElegidos):
				x = random.randrange(0,numColumnas)
				y = random.randrange(0,numFilas)
				if not (y,x) in lugaresElegidos:
					lugaresElegidos.append((y,x))

		jugadasNegro = lugaresElegidos[:4]
		jugadasRojo = lugaresElegidos[4:]
		partida = recrearPartida(jugadasNegro, jugadasRojo)		
	
		esFinal = partida.termino('o') or partida.termino('x')
		if not esFinal:
			return partida
	

def recrearPartida(jugadasNegro, jugadasRojo):
	partida = Partida()
	
	# cargo los 'x'
	for pos in jugadasNegro:
		partida.poner('x', *pos)

	# cargo los 'o'
	for pos in jugadasRojo:
		partida.poner('o', *pos)
	
	return partida
