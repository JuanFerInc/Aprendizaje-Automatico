def newTelemetriaData():
	return {
		'pesos': [], # historial de pesos
		'valoresTableros': [], # historial de valores de tableros
		'numVictorias': 0,
		'numDerrotas': 0,
		'numEmpates': 0,
		'resultadosPartidas': [], # historial de resultados, true = gano
		'cantTurnos': [], # historial de turnos que le llevo terminar la partida i
		'errorCuadraticoMedio':[],
		'pesosIniciales' : [],
		'pesosFinales': []
	}

data = newTelemetriaData()