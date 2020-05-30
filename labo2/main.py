from funcionesAux import *

if __name__ == "__main__":
    # ################ parte a: id3 (descomentar para activar)

    # ################ parte b: random forest (descomentar para activar)

    start_time = time.time()
    # runID3()
    runRandomForest()
    print("TIEMPO DE EJECUCION --- %s seconds ---" % (time.time() - start_time))


