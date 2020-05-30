from teeko import *

# ┌───┬───┬───┬───┬───┐
# │   │   │   │   │   │
# ├───┼───┼───┼───┼───┤
# │   │ . │   │   │   │
# ├───┼───┼───┼───┼───┤
# │   │   │   │   │   │
# ├───┼───┼───┼───┼───┤
# │   │   │   │   │   │
# ├───┼───┼───┼───┼───┤
# │   │   │   │   │   │
# └───┴───┴───┴───┴───┘

# retorna True si ambas listas contienen los mismos casilleros
def sonIguales(l1, l2):
	
	if len(l1) != len(l2):
		return False

	for casillero in l1:
		if casillero not in l1:
			return False

	return True

def obtenerAdyacentes_test():
	tablero = Tablero()

	got = tablero.obtenerAdyacentes(1,1)
	expected = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
	ok = sonIguales(got, expected)
	if not ok:
		print('got: ', got)
		print('expected: ', expected)
		raise ValueError('algo salio mal')

	got = tablero.obtenerAdyacentes(0,0)
	expected = [(0,1),(1,0),(1,1)]
	ok = sonIguales(got, expected)
	if not ok:
		print('got: ', got)
		print('expected: ', expected)
		raise ValueError('algo salio mal')

	got = tablero.obtenerAdyacentes(3,4)
	expected = [(2,4),(2,3),(3,3),(4,3),(4,4)]
	ok = sonIguales(got, expected)
	if not ok:
		print('got: ', got)
		print('expected: ', expected)
		raise ValueError('algo salio mal')

	got = tablero.obtenerAdyacentes(0, 1)
	expected = [(0,0),(0,1),(1,1),(1,2),(2,1)]
	ok = sonIguales(got, expected)
	print(got)
	if not ok:
		print('got: ', got)
		print('expected: ', expected)
		raise ValueError('algo salio mal')

def sonAdyacentes_test():
	tablero = Tablero()

	got = tablero.sonAdyacentes((1,1),(1,2))
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	got = tablero.sonAdyacentes((1,1),(1,1))
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	got = tablero.sonAdyacentes((0,1),(0,0))
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	got = tablero.sonAdyacentes((0,0), (0,1))
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	got = tablero.sonAdyacentes((3,4), (4,4))
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	got = tablero.sonAdyacentes((2,2), (4,4))
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def obtenerMovimientosPosibles_test():
	partida1 = Partida()
	partida1.poner('x', 0, 1)
	partida1.poner('x', 2, 3)
	partida1.poner('o', 1, 0)
	partida1.poner('x', 2, 0)
	partida1.poner('o', 1, 2)
	partida1.poner('x', 1, 3)
	partida1.poner('x', 4, 4)

	partida1.tablero.print()

	res = partida1.obtenerMovimientosPosibles('x')
	print(res)

def simularMovimiento_test():
	partida1 = Partida()
	partida1.poner('x', 0, 1)
	partida1.poner('x', 2, 3)
	partida1.poner('o', 1, 0)
	partida1.poner('x', 2, 0)
	partida1.poner('o', 1, 2)
	partida1.poner('x', 1, 3)
	partida1.poner('x', 4, 4)

	copia = partida1.simularMovimiento((2, 3),(1, 4))
	copia.tablero.print()
	partida1.tablero.print()

def hayCuadrado_test():
	
	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = partida1.hayCuadrado('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 2)

	got = partida1.hayCuadrado('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = partida1.hayCuadrado('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 0)
	partida1.poner('x', 4, 1)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 4, 3)

	got = partida1.hayCuadrado('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def hayAlineacionVertical_test():
	
	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 2)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 0)
	partida1.poner('x', 4, 1)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 4, 3)

	got = partida1.hayAlineacionVertical('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 1)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 1)
	partida1.poner('x', 0, 2)
	partida1.poner('x', 0, 3)
	partida1.poner('x', 0, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 2)
	partida1.poner('x', 0, 3)
	partida1.poner('x', 0, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 0)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 1)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)

	got = partida1.hayAlineacionVertical('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 1)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 4, 0)

	got = partida1.hayAlineacionVertical('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def hayAlineacionDiagonal_test():
	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 2)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 0)
	partida1.poner('x', 4, 1)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 4, 3)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 1)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 1)
	partida1.poner('x', 0, 2)
	partida1.poner('x', 0, 3)
	partida1.poner('x', 0, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 2)
	partida1.poner('x', 0, 3)
	partida1.poner('x', 0, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 0)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 1)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 4)

	got = partida1.hayAlineacionDiagonal('x')
	expected = False
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 1, 1)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 3)

	got = partida1.hayAlineacionDiagonal('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 4)
	partida1.poner('x', 1, 3)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 1)

	got = partida1.hayAlineacionDiagonal('x')
	expected = True
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def maxFichasHorizontal_test():
	partida1 = Partida()
	jugador1 = Jugador('x',[.5,.5,.5])
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = jugador1.maxFichasHorizontal(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 1, 1)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 3)

	got = jugador1.maxFichasHorizontal(partida1.jugadasNegro)
	expected = 1
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 2, 3)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 4, 4)

	got = jugador1.maxFichasHorizontal(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 1)
	partida1.poner('x', 2, 1)
	partida1.poner('x', 3, 1)

	got = jugador1.maxFichasHorizontal(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 2, 2)
	partida1.poner('x', 1, 2)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 4, 2)

	got = jugador1.maxFichasHorizontal(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def maxFichasVertical_test():
	partida1 = Partida()
	jugador1 = Jugador('x',[.5,.5,.5])
	partida1.poner('x', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = jugador1.maxFichasVertical(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 1, 1)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 3)

	got = jugador1.maxFichasVertical(partida1.jugadasNegro)
	expected = 1
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 4, 4)
	partida1.poner('x', 3, 3)
	partida1.poner('x', 3, 2)
	partida1.poner('x', 3, 4)


	got = jugador1.maxFichasVertical(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 1, 1)
	partida1.poner('x', 1, 2)
	partida1.poner('x', 1, 3)
	partida1.poner('x', 1, 4)

	got = jugador1.maxFichasVertical(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 2, 2)
	partida1.poner('x', 2, 1)
	partida1.poner('x', 2, 3)
	partida1.poner('x', 2, 4)

	got = jugador1.maxFichasVertical(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def maxFichasDiagonalUpDown_test():
	partida1 = Partida()
	jugador1 = Jugador('x', [.5, .5, .5])
	partida1.poner('x', 3, 2)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 2, 3)
	partida1.poner('x', 4, 1)

	got = jugador1.maxFichasDiagonalUpDown(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 1, 1)
	partida1.poner('x', 0, 0)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 1, 3)

	got = jugador1.maxFichasDiagonalUpDown(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 2, 4)
	partida1.poner('x', 4, 4)

	got = jugador1.maxFichasDiagonalUpDown(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 2, 1)
	partida1.poner('x', 3, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 2)

	got = jugador1.maxFichasDiagonalUpDown(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 0, 0)
	partida1.poner('x', 1, 1)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 3)

	got = jugador1.maxFichasDiagonalUpDown(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

def maxFichasDiagonalDownUp_test():
	partida1 = Partida()
	jugador1 = Jugador('x',[.5,.5,.5])
	partida1.poner('x', 3, 2)
	partida1.poner('x', 4, 3)
	partida1.poner('x', 2, 3)
	partida1.poner('x', 4, 1)

	got = jugador1.maxFichasDiagonalDownUp(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 1, 1)
	partida1.poner('x', 0, 0)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 1, 3)

	got = jugador1.maxFichasDiagonalDownUp(partida1.jugadasNegro)
	expected = 2
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 3, 3)
	partida1.poner('x', 4, 2)
	partida1.poner('x', 2, 4)
	partida1.poner('x', 4, 4)

	got = jugador1.maxFichasDiagonalDownUp(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 2, 1)
	partida1.poner('x', 3, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 2)


	got = jugador1.maxFichasDiagonalDownUp(partida1.jugadasNegro)
	expected = 3
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')

	partida1 = Partida()
	partida1.poner('x', 1, 3)
	partida1.poner('x', 2, 2)
	partida1.poner('x', 3, 1)
	partida1.poner('x', 4, 0)

	got = jugador1.maxFichasDiagonalDownUp(partida1.jugadasNegro)
	expected = 4
	ok = got == expected
	if not ok:
		raise ValueError('algo salio mal')


def testCrash_test():
	
	partida1 = Partida()
	partida1.poner('o', 0, 0)
	partida1.poner('x', 0, 1)
	partida1.poner('x', 1, 0)
	partida1.poner('x', 1, 1)

	got = partida1.obtenerMovimientosPosibles('o')
	print(got)

#obtenerAdyacentes_test()
#sonAdyacentes_test()
#obtenerMovimientosPosibles_test()
#simularMovimiento_test()
#hayCuadrado_test()
#hayAlineacionVertical_test()
#hayAlineacionDiagonal_test()
# testCrash_test()
#debuggear_partida()
