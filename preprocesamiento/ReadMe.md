***********************************
*Indice de archivos del directorio*				  *		
***********************************

--> discretizacion_750k.ipynb: Notebook donde se discretiza el las variables continuas del conjunto de datos de entrenamiento, previo a utilizar la tasa de inconsistencia como 
medida de calidad. Se utilizó un método supervisado, con el modulo DecisionTreeDiscretiser de la libreria  feature_engine.discretisation.

--> EDA_b_flavor.ipynb: Notebook donde se realiza un analisis exploratorio del conjunto de datos de etiquetado de mesones.

--> seleccion_instancias_lightgbm.ipynb: Notebook que se encarga del preprosesamiento de instancias mediante un clasificador 'Oraculo', el cual es entrenado con millones de instancias en un tiempo corto (lightgbm), le asigna a cada una un valor de probabilidad de pertenecer a la clase 1. Esto permite crear subconjuntos de instancias mas pequeños, con las instancias mas informativas (las instanciasde mayoy incertidumbre, es decir, con probabilidad alrededor de 0.5).

--> seleccion_instancias_aleatoria.ipynb: Notebook baseline que selecciona 750,000 instancias de forma aleatoria uniforme (semilla 42) de las primeras 2.5M filas, para comparar contra la selección inteligente 80/20 basada en el Oráculo LightGBM (seleccion_instancias_lightgbm.ipynb). Verifica que la muestra preserve la proporción de clases y guarda X_train_750k_random.npy / y_train_750k_random.npy.
