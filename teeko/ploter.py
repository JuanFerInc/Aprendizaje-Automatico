import sys
import re
import io
from datetime import datetime

# para plotting
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd # colores
import matplotlib.patches as mpatches

# from matplotlib.font_manager import FontProperties # leyendas
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# solve for a and b
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


def plot_historial_pesos(ax, historial_pesos):


    ax.set_xlabel('tiempo')

    ax.title.set_text('Evolucion pesos')


    if len(historial_pesos) == 0:
        return

    cantPesos = len(historial_pesos[0])

    colores = get_cmap(cantPesos)

    data = {}
    for i in range(cantPesos):
        data[i] = []

    for pesos in historial_pesos:
        for peso, valor in enumerate(pesos):
            data[peso].append(valor)
    
    for pesoIdx, evolucion in data.items():
        cn = colores(pesoIdx)
        label = 'w' + str(pesoIdx)
        ax.plot(evolucion, color=cn, linewidth=0.5, label=label)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.7, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.5, 0.8), shadow=True, ncol=1)



def plot_errorCuadratico(ax, historial_pesos):
    ax.plot(historial_pesos, '-', linewidth=0.5)
    ax.set_xlabel('Partidas')
    ax.set_ylabel('Error')
    ax.title.set_text('Error cometido con el paso del tiempo')

    # best fit line
    X = [x for x in range(len(historial_pesos))]
    a, b = best_fit(X, historial_pesos)
    yfit = [a + b * x for x in X]

    # promedio
    avg = np.mean(historial_pesos)
    avg = round(avg, 2)

    ax.plot(X, yfit, label='promedio={}'.format(avg))
    ax.legend()


def plot_historial_valores_tableros(ax, historialValoresTableros):
    ax.plot(historialValoresTableros, '.', linewidth=0.5)
    ax.set_xlabel('tiempo')
    ax.set_ylabel('valor tablero')
    ax.title.set_text('V(b)')


def plot_resultados(ax, historialResultados, numVictorias, numDerrotas):
    ax.plot(historialResultados, 'ob')

    # promedio
    avg = numVictorias / (numVictorias + numDerrotas)
    ax.axhline(y=avg, color='g', linestyle='-', label='win rate={}'.format(avg))
    ax.legend()

    ax.title.set_text('Resultados partidas')


def plot_cantTurnos(ax, historialCantTurnos):
    ax.plot(historialCantTurnos, '-', linewidth=0.5)
    ax.set_xlabel('numero de partida')
    # ax.set_ylabel('cant. turnos terminar la partida')
    ax.title.set_text('cant. turnos terminar la partida')

    # best fit line
    X = [x for x in range(len(historialCantTurnos))]
    a, b = best_fit(X, historialCantTurnos)
    yfit = [a + b * x for x in X]

    # promedio
    avg = np.mean(historialCantTurnos)
    avg = round(avg, 2)

    ax.plot(X, yfit, label='promedio={}'.format(avg))
    ax.legend()

def plotPesosInicialFinal(ax, pesosIniciales,pesosFinales):
    ax = plt.subplot(111)
    w = 0.3
    labels = ['W0','W1','W2','W3','W4','W5','W6','W7','W8']
    x = np.arange(len(labels))

    ax.bar(x,pesosIniciales[0], width=w, color='r', align='center')
    ax.bar(x+.4,pesosFinales[0],width=w, color='b', align='center')

    red_patch = mpatches.Patch(color='red', label='Pesos antes')
    blue_patch = mpatches.Patch(color='blue', label='Pesos despues')
    ax.legend(handles=[red_patch, blue_patch])

    ax.set_xlabel('Pesos')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Valor')

    ax.set_title('Pesos Iniciales y Finles')
    
    ax.autoscale(tight=True)

def plot(telemetria, plotearPorSeparado = True,plotearPesos = True):

    print('Ganadas por Jugador Negro:', telemetria['numVictorias'])
    print('Ganadas por Jugador Rojo:', telemetria['numDerrotas'])

    historial_pesos          = telemetria['pesos']
    historialValoresTableros = telemetria['valoresTableros']
    resultados               = telemetria['resultadosPartidas']
    numVictorias             = telemetria['numVictorias']
    numDerrotas              = telemetria['numDerrotas']
    hitorialTurnos           = telemetria['cantTurnos']
    errorCuadratico           = telemetria['errorCuadraticoMedio']

    pesosIniciales           = telemetria['pesosIniciales']
    pesosFinales           = telemetria['pesosFinales']


    if plotearPorSeparado:
        if plotearPesos:
            fig1, ax1 = plt.subplots()
            plot_historial_pesos(ax1, historial_pesos)
            fig2, ax2 = plt.subplots()
            plotPesosInicialFinal(ax2, pesosIniciales, pesosFinales)

        fig3, ax3 = plt.subplots()
        plot_historial_valores_tableros(ax3, historialValoresTableros)

        fig4, ax4 = plt.subplots()
        plot_resultados(ax4, resultados, numVictorias, numDerrotas)

        fig5, ax5 = plt.subplots()
        plot_cantTurnos(ax5, hitorialTurnos)

        if plotearPesos:
            fig6, ax6 = plt.subplots()
            plot_errorCuadratico(ax6, errorCuadratico)

    else:

        fig, ax = plt.subplots(2, 2)

        # historial pesos
        plot_historial_pesos(ax[0, 0], historial_pesos)

        # valoraciones tableros
        plot_historial_valores_tableros(ax[0, 1], historialValoresTableros)
        ax[1, 0].plot(range(100), 'b') #row=1, col=0
        ax[1, 1].plot(range(50), 'k') #row=1, col=1

    plt.show()

if __name__ == '__main__':
    # con aleatoriedad
    hitorialTurnos = [59, 139, 39, 17, 49, 63, 1, 73, 115, 95, 33, 67, 67, 19, 33, 57, 37, 83, 5, 14, 26, 67, 45, 47, 149, 83, 53, 45, 85, 29, 25, 21, 31, 171, 39, 43, 15, 83, 27, 121, 15, 25, 53, 25, 15, 85, 8, 8, 5, 89, 29, 5, 23, 71, 13, 43, 41, 23, 6, 64, 223, 77, 63, 29, 51, 63, 49, 43, 227, 93, 27, 95, 15, 45, 7, 57, 19, 49, 47, 51, 51, 69, 73, 53, 33, 37, 25, 69, 79, 63, 49, 59, 3, 19, 23, 29, 11, 11, 47, 7, 11, 89, 39, 75, 56, 4, 67, 41, 115, 19, 27, 63, 69, 59, 35, 41, 47, 48, 12, 35, 33, 10, 110, 99, 63, 39, 3, 57, 109, 93, 17, 57, 3, 11, 21, 19, 35, 85, 43, 7, 35, 11, 23, 73, 23, 19, 23, 33, 51, 41, 69, 9, 77, 15, 65, 71, 47, 19, 11, 21, 65, 93, 27, 41, 51, 55, 23, 4, 6, 21, 93, 37, 77, 215, 55, 37, 11, 49, 17, 7, 23, 69, 131, 235, 3, 57, 49, 183, 145, 61, 59, 15, 13, 57, 27, 31, 81, 31, 21, 51, 23, 139, 11, 3, 3, 49, 37, 6, 50, 83, 55, 59, 13, 36, 56, 103, 81, 49, 83, 61, 27, 3, 6, 12, 65, 43, 39, 15, 29, 27, 41, 17, 7, 49, 77, 49, 27, 21, 13, 51, 27, 43, 13, 5, 13, 97, 9, 7, 39, 31, 179, 83, 11, 49, 19, 77, 39, 93, 39, 199, 63, 49, 7, 55, 67, 159, 27, 53, 51, 117, 13, 5, 5, 75, 7, 5, 31, 125, 53, 39, 25, 13, 41, 14, 32, 63, 29, 61, 57, 13, 35, 27, 109, 67, 21, 43, 21, 65, 7, 197, 63, 131, 33, 13, 57, 11, 31, 71, 21, 5, 162, 32, 55, 25, 21, 7, 19, 11, 17, 9, 11, 9, 75, 57, 17, 131, 27, 3, 79, 25, 25, 19, 23, 11, 33, 33, 35, 73, 9, 23, 87, 79, 48, 8, 19, 41, 63, 117, 149, 39, 41, 27, 29, 51, 25, 9, 45, 49, 67, 23, 257, 13, 57, 45, 189, 63, 83, 73, 5, 1, 49, 25, 41, 89, 79, 53, 23, 53, 37, 43, 41, 33, 137, 5, 173, 9, 35, 35, 3, 201, 35, 31, 93, 115, 49, 15, 29, 165, 5, 53, 47, 45, 11, 9, 13, 23, 19, 4, 80, 25, 113, 3, 65, 57, 7, 23, 61, 33, 57, 19, 3, 39, 31, 33, 95, 19, 25, 9, 29, 39, 41, 9, 59, 6, 6, 73, 13, 9, 49, 9, 91, 9, 37, 45, 25, 21, 7, 15, 33, 167, 29, 63, 25, 5, 33, 53, 40, 34, 83, 105, 45, 23, 57, 5, 100, 58, 105, 7, 47, 15, 23, 85, 21, 61, 33, 119, 121, 5, 13, 7, 55, 7, 143, 15, 45, 45, 47, 19, 25, 119, 13, 127, 27, 53, 7, 29, 9, 15, 9, 25, 27, 27, 13, 15, 21, 21, 23, 23, 3, 35, 45, 23, 13, 69, 3, 27, 39, 47, 90, 26, 7, 63, 19, 59, 37, 25, 1, 43, 89, 83, 87, 7, 143, 85, 5, 7, 53, 125, 36, 32, 81, 49, 15, 7, 25, 93, 47, 7, 5, 63, 9, 89, 19, 21, 35, 13, 15, 155, 49, 27, 79, 65, 7, 9, 7, 35, 97, 31, 39, 25, 321, 29, 7, 63, 9, 31, 85, 3, 61, 101, 10, 42, 5, 55, 39, 61, 195, 7, 13, 69, 51, 19, 7, 11, 65, 3, 51, 29, 15, 111, 63, 33, 115, 145, 19, 35, 109, 57, 23, 13, 59, 51, 15, 11, 91, 35, 119, 133, 15, 49, 43, 31, 19, 8, 78, 31, 107, 27, 15, 31, 95, 25, 45, 5, 67, 35, 29, 17, 47, 25, 47, 47, 53, 101, 147, 31, 49, 33, 29, 31, 39, 13, 63, 29, 127, 43, 19, 43, 2, 22, 35, 47, 26, 34, 37, 25, 9, 35, 123, 11, 175, 15, 5, 7, 81, 19, 39, 77, 29, 49, 25, 13, 97, 15, 31, 231, 29, 11, 147, 137, 75, 39, 121, 63, 21, 75, 99, 49, 109, 65, 47, 47, 49, 15, 9, 27, 53, 79, 19, 67, 5, 25, 9, 35, 113, 59, 35, 23, 57, 47, 3, 29, 57, 25, 23, 27, 106, 24, 53, 15, 7, 87, 103, 27, 37, 33, 55, 55, 3, 17, 33, 55, 51, 1, 1, 41, 31, 57, 21, 33, 15, 55, 31, 17, 9, 41, 13, 43, 45, 15, 27, 35, 27, 33, 73, 55, 59, 41, 53, 63, 49, 125, 7, 39, 33, 21, 17, 51, 111, 23, 75, 13, 89, 47, 131, 49, 71, 27, 133, 13, 9, 71, 33, 49, 119, 23, 37, 9, 19, 61, 54, 66, 69, 19, 17, 107, 59, 19, 19, 19, 29, 9, 49, 55, 13, 63, 13, 21, 7, 93, 33, 57, 41, 19, 23, 29, 21, 13, 15, 7, 85, 11, 61, 43, 43, 7, 97, 13, 43, 29, 11, 83, 53, 85, 21, 149, 21, 41, 11, 63, 27, 13, 45, 23, 29, 23, 15, 23, 157, 27, 13, 75, 63, 69, 27, 23, 129, 21, 31, 15, 21, 9, 29, 11, 120, 16, 85, 15, 71, 7, 93, 41, 35, 31, 91, 9, 6, 42, 23, 1, 39, 86, 10, 51, 5, 31, 77, 51, 117, 87, 85, 57, 15, 29, 15, 63, 95, 19, 31, 15, 65, 61, 33, 45, 23, 47, 15, 23, 15, 45, 85, 29, 47, 11, 19, 37, 35, 17, 45, 29, 10, 10, 121, 107, 75, 35, 101, 51, 29, 80, 20, 23, 37, 61, 59, 65, 65, 39, 27, 55, 17, 5, 7, 21, 41, 19, 91, 6, 22, 55, 47, 21, 21, 17, 79, 103, 43, 45, 96, 56, 19, 13, 19, 7, 23, 1, 21, 21, 41, 57, 7, 35, 15, 83, 33, 7, 73, 19, 25, 55, 33, 5, 65, 21, 25, 83, 22, 7, 116, 9]
    fig4, ax4 = plt.subplots()
    plot_cantTurnos(ax4, hitorialTurnos)
    
    # sin aleatoriedad
    hitorialTurnos = [7, 7, 19, 11, 13, 3, 7, 37, 13, 3, 5, 13, 13, 19, 33, 7, 5, 15, 17, 27, 81, 21, 11, 15, 39, 9, 9, 7, 7, 21, 13, 21, 11, 25, 101, 17, 29, 7, 7, 19, 43, 17, 1, 5, 31, 5, 19, 95, 29, 27, 7, 19, 43, 1, 11, 33, 7, 9, 45, 7, 29, 13, 7, 43, 13, 4, 10, 3, 5, 21, 9, 13, 11, 13, 9, 25, 11, 3, 5, 9, 9, 21, 13, 15, 47, 11, 47, 11, 13, 3, 17, 13, 35, 25, 45, 11, 31, 11, 7, 3, 27, 15, 17, 5, 25, 17, 19, 7, 13, 15, 9, 5, 107, 9, 9, 13, 23, 17, 3, 15, 6, 14, 9, 25, 11, 39, 15, 11, 17, 27, 33, 9, 9, 5, 5, 3, 3, 1, 41, 19, 11, 31, 13, 47, 11, 41, 19, 3, 5, 17, 3, 11, 9, 5, 9, 15, 5, 101, 49, 13, 11, 45, 33, 7, 25, 11, 5, 17, 33, 1, 5, 65, 97, 5, 15, 19, 32, 10, 5, 15, 61, 53, 21, 3, 11, 7, 11, 23, 7, 59, 21, 13, 5, 7, 5, 3, 23, 29, 51, 1, 13, 25, 13, 19, 27, 5, 7, 15, 11, 15, 11, 7, 3, 33, 17, 3, 9, 19, 3, 11, 3, 1, 71, 67, 1, 95, 3, 53, 1, 9, 7, 27, 11, 17, 13, 9, 9, 1, 11, 3, 7, 5, 5, 19, 9, 27, 13, 19, 17, 59, 13, 3, 13, 43, 13, 23, 67, 1, 77, 19, 37, 11, 5, 3, 23, 105, 13, 63, 11, 41, 9, 13, 17, 9, 11, 7, 35, 21, 11, 15, 21, 19, 9, 11, 53, 3, 5, 21, 11, 15, 5, 3, 87, 19, 15, 1, 11, 5, 117, 11, 7, 3, 7, 3, 39, 5, 9, 81, 51, 45, 3, 3, 33, 3, 1, 141, 9, 5, 23, 7, 9, 31, 17, 45, 29, 7, 3, 17, 17, 5, 3, 5, 15, 173, 11, 9, 215, 15, 1, 51, 11, 11, 1, 5, 7, 19, 15, 5, 9, 47, 21, 5, 31, 139, 69, 103, 21, 5, 11, 5, 83, 225, 17, 43, 9, 5, 39, 47, 23, 13, 145, 21, 25, 11, 31, 19, 39, 1, 213, 33, 19, 173, 11, 39, 9, 25, 7, 13, 11, 19, 7, 11, 3, 19, 19, 15, 9, 9, 4, 40, 17, 17, 9, 3, 19, 13, 3, 29, 27, 13, 13, 23, 9, 25, 5, 114, 149, 252, 11, 5, 13, 49, 12, 10, 9, 11, 51, 19, 25, 13, 6, 6, 7, 365, 47, 53, 41, 5, 13, 9, 43, 11, 12, 18, 3, 11, 19, 15, 9, 5, 15, 3, 9, 23, 87, 13, 5, 9, 7, 5, 9, 31, 9, 17, 1, 21, 41, 15, 9, 9, 5, 7, 27, 21, 11, 29, 7, 23, 1, 7, 27, 23, 9, 25, 7, 39, 73, 1, 23, 21, 3, 5, 5, 21, 39, 7, 57, 7, 13, 5, 7, 13, 3, 45, 1, 53, 97, 19, 9, 63, 9, 35, 17, 23, 3, 67, 19, 13, 11, 69, 9, 37, 21, 13, 5, 11, 17, 5, 21, 13, 9, 13, 31, 13, 7, 51, 21, 1, 65, 11, 23, 53, 9, 43, 17, 13, 1, 19, 3, 39, 21, 1, 13, 19, 15, 35, 17, 15, 5, 17, 59, 63, 3, 9, 17, 15, 9, 43, 117, 19, 43, 35, 27, 7, 1, 13, 27, 103, 17, 11, 15, 15, 11, 1111, 35, 11, 17, 11, 3, 69, 5, 29, 5, 39, 17, 31, 309, 31, 61, 23, 7, 13, 15, 15, 3, 29, 35, 21, 11, 55, 5, 35, 5, 19, 13, 5, 31, 9, 19, 31, 1, 45, 11, 217, 7, 243, 21, 15, 13, 101, 185, 47, 9, 7, 3, 19, 1, 5, 123, 11, 265, 15, 11, 11, 19, 11, 23, 3, 63, 37, 5, 21, 5, 1, 9, 15, 35, 3, 19, 27, 11, 13, 7, 57, 7, 35, 9, 13, 131, 9, 11, 1, 89, 21, 27, 9, 13, 7, 41, 23, 5, 11, 9, 3, 19, 27, 15, 61, 9, 11, 18, 10, 7, 13, 9, 11, 7, 55, 15, 5, 5, 9, 159, 29, 5, 101, 15, 77, 39, 23, 15, 13, 7, 23, 9, 11, 1, 17, 19, 13, 5, 29, 9, 11, 21, 21, 31, 5, 23, 156, 6, 1, 17, 5, 11, 17, 181, 3, 25, 7, 25, 21, 15, 19, 5, 7, 5, 9, 5, 17, 2, 2, 21, 5, 5, 3, 9, 7, 23, 5, 9, 27, 39, 15, 3, 9, 13, 3, 13, 47, 17, 51, 11, 13, 5, 241, 17, 33, 21, 23, 9, 13, 17, 6, 8, 63, 11, 3, 13, 27, 21, 9, 61, 3, 5, 15, 9, 23, 1, 97, 7, 15, 14, 6, 5, 9, 13, 17, 7, 11, 21, 1, 15, 5, 17, 35, 13, 31, 49, 9, 17, 23, 69, 7, 25, 15, 23, 1, 27, 7, 45, 5, 9, 21, 5, 7, 7, 9, 7, 29, 9, 11, 13, 15, 19, 7, 101, 21, 5, 35, 91, 27, 5, 13, 31, 17, 17, 23, 13, 113, 1, 7, 3, 1, 27, 9, 51, 3, 25, 11, 5, 17, 85, 21, 55, 7, 13, 15, 17, 51, 15, 49, 11, 9, 15, 11, 11, 7, 25, 29, 47, 11, 15, 9, 21, 11, 35, 7, 3, 7, 7, 9, 13, 23, 25, 23, 25, 33, 315, 9, 23, 5, 13, 11, 21, 5, 125, 13, 9, 7, 15, 9, 17, 13, 17, 9, 43, 27, 101, 103, 95, 5, 21, 19, 5, 13, 11, 59, 3, 43, 55, 21, 17, 11, 5, 29, 401, 33, 23, 17, 5, 17, 17, 1, 7, 3, 15, 41, 7, 13, 11, 7, 57, 19, 9, 11, 13, 7, 3, 11, 60, 6, 9, 1, 43, 25, 3, 121, 139, 37, 9, 13, 23, 15, 25, 9, 9, 19, 11, 11, 13, 9, 5, 17, 5, 15, 25, 7, 15, 19]
    fig5, ax5 = plt.subplots()
    plot_cantTurnos(ax5, hitorialTurnos)

    plt.show()