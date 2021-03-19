Para todos los laboratorios se escribio el codigo que soluciona el Problema en Pyton y
el informe en Jupyter Notebook

LAB 1

Problema
Considere al juego Teeko1 con la siguiente variación para el comienzo de la partida: el
tablero comienza con las ocho fchas en una posición aleatoria no fnal, en vez de ser
colocadas una a una por los jugadores como en la versión original.
a) Modele e implemente un jugador de Teeko utilizando como guía el ejemplo visto en el
teórico para el juego de las damas.
b) Entrene su solución con: (i) un oponente que siempre juegue de forma aleatoria y (ii) un
oponente que sea la versión del propio jugador en alguna iteración anterior.
c) Con los pesos fjos, haga competir entre sí a los dos jugadores obtenidos en la parte
anterior durante 100 partidas.

LAB 2

Problema
Considere al conjunto de datos QSAR1, un juego de datos con 1025 atributos, 1024 de
entrada y uno de salida, todos binarios.
Se pide:
a) Implemente el algoritmo ID3 visto en el teórico. Evalúe sus resultados sobre QSAR.
b) Implemente el algoritmo Random Forest, utilizando como base su implementación de la
parte (a), realizándole las modifcaciones necesarias. Evalúe sus resultados sobre QSAR.
c) Aplique las implementaciones de Scikit-learn para (a) y (b) sobre QSAR y compare los
resultados con los de sus implementaciones.
En las partes (a) y (b) se podrá utilizar pandas y scikit-learn para la carga del dataset y la
generación de archivos de entrenamiento, testeo, etc.

LAB 3

Problema
Considere al conjunto de datos Adult1, un conjunto de datos con 48842 instancias, 14
atributos de entrada y uno de salida.
Se pide:
a) Describa las principales características del conjunto de datos, incluyendo, por ejemplo,
valores posibles para cada atributo, valores faltantes, distribución de la clase objetivo. Esto
tiene como objeto entender el dataset con el que se está trabajando.
b) Describa cómo dividirá el conjunto para el aprendizaje, sabiendo que deberá evaluar sobre
un dataset de evaluación, y que la validación debe hacerse utilizando validación cruzada.
Indique si es relevante utilizar estratificación o no, justificando.
c) Implemente el algoritmo KNN (vecinos más ceranos) visto en el teórico. Evalúe sus
resultados sobre el conjunto Adult.
d) Implemente el algoritmo Naive Bayes visto en teórico. Evalúe sus resultados sobre el
conjunto Adult.
e) Aplique las implementaciones de Scikit-learn para (d) y (e) sobre Adult y compare los
resultados con los de sus implementaciones.
f) Presente los resultados de evaluación, incluyendo medidas de accuracy, precision, recall y
medida-F.
g) Comente brevemente los resultados obtenidos. Compare con otros resultados reportados
en el mismo conjunto de datos.
Observaciones:
- Está permitido (y recomendado) utilizar bibliotecas Python para toda tarea auxiliar, no
incluyendo, por supuesto, la implementación de los algoritmos en c) y d)
- Al implementar los algoritmos, recomendamos utilizar la biblioteca NumPy para trabajar con
operaciones vectoriales y mejorar su rendimiento computacional.
- No es necesario, aunque tampoco prohibido, realizar selección de atributos