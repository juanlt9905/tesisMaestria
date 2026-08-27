import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import inconsistencia_optimizada 
import pickle

def run_model(X_train, X_test, Y_train, Y_test):
    clf_base = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg")
    clf_base.fit(X_train, Y_train)
    return clf_base
def evaluate_model(model, train_features, train_target, test_features, test_target):
   
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        nombre_grafica = 'Entrenamiento' if type == 'train' else 'Prueba'

        # Ajuste de tamaño de fuente
        size_ejes = 16
        size_titulo = 18
        size_leyenda = 14
        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{nombre_grafica}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=12)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Umbral', fontsize= size_ejes, fontweight = 'bold')
        ax.set_ylabel('F1', fontsize = size_ejes, fontweight = 'bold')
        ax.legend(loc='lower center', fontsize = size_leyenda)
        ax.set_title(f'Valor F1', fontsize= size_titulo, fontweight = 'bold') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{nombre_grafica}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=12) #tamanio del punto
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR', fontsize = size_ejes, fontweight = 'bold')
        ax.set_ylabel('TPR', fontsize =size_ejes, fontweight = 'bold')
        ax.legend(loc='lower center', fontsize= size_leyenda)        
        ax.set_title(f'Curva ROC', fontsize= size_titulo, fontweight = 'bold')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{nombre_grafica}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales        
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=12)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall', fontsize= size_ejes, fontweight = 'bold')
        ax.set_ylabel('precision', fontsize= size_ejes, fontweight = 'bold')
        ax.legend(loc='lower center', fontsize= size_leyenda)
        ax.set_title(f'PRC', fontsize= size_titulo, fontweight = 'bold')   
        
        eval_stats[type]['Exactitud (0.5)'] = metrics.accuracy_score(target, pred_target)
        acc_scores = [metrics.accuracy_score(target, pred_proba >= t) for t in f1_thresholds]
        eval_stats[type]['Exactitud Max'] = np.max(acc_scores)
        
        eval_stats[type]['F1 (0.5)'] = metrics.f1_score(target, pred_target)
        eval_stats[type]['F1 Max'] = np.max(f1_scores)
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Exactitud (0.5)', 'Exactitud Max', 'F1 (0.5)', 'F1 Max', 'APS', 'ROC AUC'))
    for ax in axs:
        ax.tick_params(axis='both', labelsize=16)
    print(df_eval_stats)
    
    return  df_eval_stats, fig#eval_stats['train']['F1'], eval_stats['test']['F1']


def tasa_inconsistencia(df, subconjunto_caracteristicas, clase):

    """
    df: conjunto de datos

    subconjunto_caracteristicas: Lista con el nomnbre de las caracteristicas
    que el algoritmo genetoco esta usando

    clase: cadena con el nombre de la columna objetivo.
    """
    m = len(df)
    total_inconsistencias = 0
    grupos = df.groupby(subconjunto_caracteristicas)
    for nombre_grupo, df_grupo in grupos:
        #numero total de instancias en cada grupo
        n = len(df_grupo)

        conteo_clases = df_grupo[clase].value_counts()
        # numero de instancias del grupo pertenecientes a la clase mas frecuente.
        if len(conteo_clases)>1:
            max_clase = conteo_clases.max()
            inconsistencias_grupo = n - max_clase
            total_inconsistencias+= inconsistencias_grupo

    return total_inconsistencias/m

def unicidad(df, columnas):
    """
    Calcula la razon de patrones únicos (Numero de grupos con instancias identicas).
    Parametros de la funcion:
    -----------
    df : pandas.DataFrame
        El dataset (ej. tus 100k eventos discretizados).
    columnas : list
        Lista con los nombres de las columnas (genes) seleccionadas.
        
    Retorna:
    --------
    float : Un valor entre [0, 1]. 
            0 significa que todos los eventos son idénticos.
            1 significa que cada evento es un patrón único.
    """
    # Si no hay columnas seleccionadas, no hay patrones que comparar
    if not columnas:
        return 0.0
    
    # El atributo .ngroups devuelve el número total de combinaciones únicas
    # de valores que existen en el subconjunto de columnas seleccionado.
    num_patrones = df.groupby(columnas).ngroups
    total_instancias = len(df)
    
    # razon  de unicidad
    razon_unicidad = num_patrones / total_instancias
    
    return razon_unicidad

def fitness_inconsistencia_unicidad(df, nombre_columnas, clase):

    m = len(df)
    total_inconsistencias = 0
    grupos = df.groupby(nombre_columnas)
    for nombre_grupo, df_grupo in grupos:
        #numero total de instancias en cada grupo
        n = len(df_grupo)

        conteo_clases = df_grupo[clase].value_counts()
        # numero de instancias del grupo pertenecientes a la clase mas frecuente.
        if len(conteo_clases)>1:
            max_clase = conteo_clases.max()
            inconsistencias_grupo = n - max_clase
            total_inconsistencias+= inconsistencias_grupo

    inconsistencia = total_inconsistencias/m
    patrones = grupos.ngroups
    uni = patrones /m

    #fitness a maximizar
    fitness_nuevo = (1 - inconsistencia) * (1 - uni)
    return fitness_nuevo


# FUNCION FITNESS BIOBJETIVO

def fitness_mod_nsga2(solution, dataframe, feature_names, target):
    """
        Toma como argumento una solucion codificada en binario, un dataframe con 
        los atributos a calcular la TI, el nombre de las caracteristicas y la clase (tarjet)
        

    """

    # decodificar solucion 
    selected_features = []

    for i in range(len(solution)):
        if solution[i]==1:
            selected_features.append(feature_names[i])
    
    num_total_features = len(feature_names)
    num_selected_features = len(selected_features)

    #print(f"DEBUG: Seleccionadas: {len(selected_features)} de {len(feature_names)}")

    if not selected_features:
        #print("DEBUG: ¡Lista vacía! Retornando penalización.")
        return [0,0]
        
    # Llamar a tasa_inconsistencia
    inconsistency_rate = tasa_inconsistencia(
        df=dataframe, 
        subconjunto_caracteristicas=selected_features, 
        clase=target
    )
    
    # Si inconsistency_rate es 0 (perfecto), el fitness sera maximo.
    fitness1 = 1.0 - inconsistency_rate

    fitness2 = (num_total_features - num_selected_features) / num_total_features
    
    return [fitness1, fitness2]

def fitness_mod_nsga2_inconsistencia_unicidad(solution, dataframe, feature_names, target):
    """
    
    Objetivo 1: Maximizar ( (1-Inconsistencia) * (1-Unicidad) )
    Objetivo 2: Maximizar Reducción de variables
    """
    # Decodificar solución
    selected_features = [feature_names[i] for i in range(len(solution)) if solution[i] == 1]
    
    num_total_features = len(feature_names)
    num_selected_features = len(selected_features)

    # Penalización si no se selecciona ninguna variable
    if not selected_features:
        return [0.0, 0.0]
        
    # 2. Cálculo de Tasa de Inconsistencia)
    #inconsistency_rate = tasa_inconsistencia(
    #    df=dataframe, 
    #    subconjunto_caracteristicas=selected_features, 
    #    clase=target
    #)
    
    fitness1 = fitness_inconsistencia_unicidad(dataframe, selected_features, target) 
    # --- OBJETIVO 2: Reducción de dimensionalidad ---
    # Calculamos el porcentaje de variables retiradas. 
    # Entre más variables quitemos, mayor es este valor (Maximización).
    fitness2 = (num_total_features - num_selected_features) / num_total_features
    
    return [fitness1, fitness2]

def fitness_wrapper_lgb(solution, X_train, y_train):
    selected_indices = np.where(solution == 1)[0]
    num_selected = len(selected_indices)
    num_total = len(solution)

    if num_selected == 0:
        return [0.0, 0.0]

    # Indexación correcta para NumPy
    X_subset = X_train[:, selected_indices]
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1, # Evita conflictos con la paralelización de PyGAD
        random_state=42,
        verbose=-1
    )

    try:
        cv_scores = cross_val_score(model, X_subset, y_train, cv=3, scoring='accuracy')
        fitness_accuracy = np.mean(cv_scores)
    except:
        return [0.0, 0.0]

    fitness_reduction = (num_total - num_selected) / num_total
    return [fitness_accuracy, fitness_reduction]

def fitness_inconsistencia_hibrida_cpp(solution, X_train, y_train, total_features):
    """
       Calcula la tasa de inconsistencia junto con la Unicidad en c++
    """
    # Filtrar columnas según el cromosoma
    indices_activos = np.where(solution == 1)[0]
    
    if len(indices_activos) == 0:
        return [0.0, 0.0]

    # Preparar subconjunto contiguidad
    X_sub = np.ascontiguousarray(X_train[:, indices_activos])

    # funcion 
    aptitud_hibrida = inconsistencia_optimizada.calcular_hibrida(X_sub, y_train)

    #estandarizacion
    reduccion = 1.0 - (len(indices_activos) / total_features)

    return [aptitud_hibrida, reduccion]

def fitness_inconsistencia_cpp(solution, X_train, y_train, total_features):
    """
      Calcula la tasa de inconsistencia en C++  
    """
    # Filtrar columnas según el cromosoma
    indices_activos = np.where(solution == 1)[0]
    
    if len(indices_activos) == 0:
        return [0.0, 0.0]

    # Preparar subconjunto contiguidad
    X_sub = np.ascontiguousarray(X_train[:, indices_activos])

    # funcion 
    aptitud = 1 - inconsistencia_optimizada.calcular(X_sub, y_train)

    #estandarizacion
    reduccion = 1.0 - (len(indices_activos) / total_features)

    return [aptitud, reduccion]


def frente_pareto(instancia, titulo_grafico, titulo_eje_x, titulo_eje_y):
    """
        Grafica el Frente de Pareto.
    """

    # Helper function to identify non-dominated solutions (the Pareto front)
    def identify_pareto(scores):
        """Identifica las soluciones no dominadas (Maximización)."""
        pop_size = scores.shape[0]
        is_pareto = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    # Compara si la solución j domina a la i
                    if (scores[j, 0] >= scores[i, 0] and scores[j, 1] >= scores[i, 1]) and \
                    (scores[j, 0] > scores[i, 0] or scores[j, 1] > scores[i, 1]):
                        is_pareto[i] = False
                        break
        return is_pareto

    #Extraemos las aptitudes de la última generación 
    # Recordatorio de objetivos: 
    # Obj 0 = Calidad (Accuracy) [0, 1]
    # Obj 1 = Reducción Normalizada [0, 1]
    fitness_valores = instancia.last_generation_fitness

    # Separamos para visualización general
    y_todos = fitness_valores[:, 0]  # Calidad
    x_todos = fitness_valores[:, 1]  # Reducción

    #Encontramos el Frente de Pareto real 
    indices_pareto = identify_pareto(fitness_valores)
    pareto_front = fitness_valores[indices_pareto]

    # Ordenamos el frente por Reducción (para dibujar la línea continua)
    sorted_indices = np.argsort(pareto_front[:, 1])
    x_pareto = pareto_front[sorted_indices, 1]
    y_pareto = pareto_front[sorted_indices, 0]

    plt.figure(figsize=(8, 8))
    # Graficamos todas las soluciones de la última generación en gris suave
    plt.scatter(x_todos, y_todos, color='gray', alpha=0.4, s=60, label='Última Generación (Todos)')

    # Graficamos las soluciones del Frente de Pareto en rojo
    plt.scatter(x_pareto, y_pareto, color='red', s=150, marker='D', label='Frente de Pareto', edgecolors='black')
    # Dibujamos la línea del Frente de Pareto
    plt.plot(x_pareto, y_pareto, color='red', linestyle='--', linewidth=3)
    
    
    plt.title(titulo_grafico, fontsize= 22)
    plt.xlabel(titulo_eje_x, fontsize = 20) #Calidad
    plt.ylabel(titulo_eje_y, fontsize=20) #NUm variables

    plt.tick_params(axis='both', which='major', labelsize=18)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    # escala sea de 0 a 1
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    #plt.legend(fontsize=20, loc='best', frameon=True, shadow=True)
    plt.legend(prop={'size': 20}, loc='best',                 
        frameon=True,               
        shadow=True,                
        markerscale=1.5,            
        labelspacing=1.2,           
        borderpad=1.0              
    )


    
    leg = plt.legend(
        loc='best', 
        frameon=True, 
        shadow=True,
        markerscale=1.8,  # Aumenta el tamaño de los iconos en la leyenda
        borderpad=1.2,    # Espacio interno del cuadro
        labelspacing=1.2  # Espacio entre líneas de texto
    )
 

    for text in leg.get_texts():
        text.set_fontsize(20) 

    leg.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.show()
    # Guardamos la gráfic
    # plt.savefig('frente_pareto_10k_wrapper.png', dpi=300, bbox_inches='tight')
    plt.show()

    import numpy as np
import matplotlib.pyplot as plt

def frente_pareto2(instancia, titulo_grafico, titulo_eje_x, titulo_eje_y):
    """
    Grafica el Frente de Pareto y devuelve los datos exactos del frente (k y Exactitud).
    """

    # Helper function to identify non-dominated solutions (the Pareto front)
    def identify_pareto(scores):
        """Identifica las soluciones no dominadas (Maximización)."""
        pop_size = scores.shape[0]
        is_pareto = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    # Compara si la solución j domina a la i
                    if (scores[j, 0] >= scores[i, 0] and scores[j, 1] >= scores[i, 1]) and \
                       (scores[j, 0] > scores[i, 0] or scores[j, 1] > scores[i, 1]):
                        is_pareto[i] = False
                        break
        return is_pareto

    # Extraemos las aptitudes de la última generación 
    # Obj 0 = Calidad (Exactitud) [0, 1]
    # Obj 1 = Reducción Normalizada [0, 1]
    fitness_valores = instancia.last_generation_fitness

    # Separamos para visualización general
    y_todos = fitness_valores[:, 0]  # Calidad
    x_todos = fitness_valores[:, 1]  # Reducción

    # Encontramos el Frente de Pareto real 
    indices_pareto = identify_pareto(fitness_valores)
    pareto_front = fitness_valores[indices_pareto]

    # ==========================================
    # NUEVO BLOQUE: Extracción de datos para exportar
    # ==========================================
    pareto_datos_exportados = []
    posiciones_optimas = np.where(indices_pareto)[0]
    
    for idx in posiciones_optimas:
        calidad = fitness_valores[idx, 0]
        # Extraer el cromosoma y sumar para tener la cantidad exacta de variables (k)
        cromosoma = instancia.population[idx]
        numero_variables = int(np.sum(cromosoma))
        
        pareto_datos_exportados.append({
            'num_vars': numero_variables,
            'exactitud': calidad  # <-- ACTUALIZADO AQUÍ
        })
    
    # Ordenar la lista por número de variables
    pareto_datos_exportados = sorted(pareto_datos_exportados, key=lambda x: x['num_vars'])
    # ==========================================

    # Ordenamos el frente por Reducción (para dibujar la línea continua en la gráfica)
    sorted_indices = np.argsort(pareto_front[:, 1])
    x_pareto = pareto_front[sorted_indices, 1]
    y_pareto = pareto_front[sorted_indices, 0]

    plt.figure(figsize=(8, 8))
    # Graficamos todas las soluciones de la última generación en gris suave
    plt.scatter(x_todos, y_todos, color='gray', alpha=0.4, s=60, label='Última Generación (Todos)')

    # Graficamos las soluciones del Frente de Pareto en rojo
    plt.scatter(x_pareto, y_pareto, color='red', s=150, marker='D', label='Frente de Pareto', edgecolors='black')
    
    # Dibujamos la línea del Frente de Pareto
    plt.plot(x_pareto, y_pareto, color='red', linestyle='--', linewidth=3)
    
    plt.title(titulo_grafico, fontsize= 22)
    plt.xlabel(titulo_eje_x, fontsize = 20)
    plt.ylabel(titulo_eje_y, fontsize=20)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    leg = plt.legend(
        prop={'size': 20}, loc='best',                 
        frameon=True, shadow=True,                
        markerscale=1.5, labelspacing=1.2,           
        borderpad=1.0              
    )

    for text in leg.get_texts():
        text.set_fontsize(20) 

    leg.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.show()

    # Devolvemos la lista estructurada
    return pareto_datos_exportados


def evolucion(instancia, titulo_grafico):

    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Extraemos el historial de fitness
    # PyGAD guarda el mejor fitness de cada generación en best_solutions_fitness
    history = np.array(instancia.best_solutions_fitness)

    # Separamos los objetivos
    # history[:, 0] -> Calidad (Accuracy)
    # history[:, 1] -> Reducción
    generaciones = np.arange(len(history))

    plt.figure(figsize=(8, 8))

    # Graficamos Calidad
    plt.plot(generaciones, history[:, 0], label='Mejor Calidad', color='blue', linewidth=4)

    # Graficamos Reducción
    plt.plot(generaciones, history[:, 1], label='Mejor Reducción de Variables', color='green', linestyle='--', linewidth=4)

    # Configuración de la gráfica
    plt.title(titulo_grafico, fontsize= 16, fontweight='bold', pad=20)
    plt.xlabel('Generación', fontsize=20, fontweight='bold')
    plt.ylabel('Valor [0, 1]', fontsize=20, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)

    leg = plt.legend(
        prop={'size': 18},      # Tamaño de fuente grande
        loc='lower right',      
        frameon=True, 
        shadow=True,
        borderpad=1.0,
        labelspacing=1.0
    )
    
    # Forzar grosor de las líneas en la leyenda para que se distingan bien
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    # 8. Eliminar márgenes blancos innecesarios
    plt.tight_layout()
    # Opcional: Zoom en el eje Y si los valores están muy cerca
    # plt.ylim(0.5, 1.0) 

    plt.show()

def seleccionar_codo_punto_utopia(fitness_scores):
    """
    Recibe el fitness de las soluciones del frente de pareto.
    Selecciona la mejor solución basada en la distancia al punto ideal (Utopia).
    Basado en Marler & Arora (2004).
    """
    # fitness_scores: Matriz (N, 2) -> Col 0: Exactitud, Col 1: Reducción

    f_max = np.max(fitness_scores, axis=0)
    f_min = np.min(fitness_scores, axis=0)
    
    #Normalizacion
    norm_scores = (fitness_scores - f_min) / (f_max - f_min)
    
    #Punto de utopia
    utopia_normalizada = np.array([1.0, 1.0])
    
    # Distancia euclidiana 
    distancias = np.sqrt(np.sum((norm_scores - utopia_normalizada)**2, axis=1))
    
    #La solucion elegida minimiza la distancia euclidiana.
    idx_mejor = np.argmin(distancias)
    
    return idx_mejor, fitness_scores[idx_mejor]

def obtener_fitness_pareto(fitness_scores):
    """Filtra los individuos para quedarse solo con el Frente de Pareto"""
    n_puntos = fitness_scores.shape[0]
    es_pareto = np.ones(n_puntos, dtype=bool)
    
    for i in range(n_puntos):
        for j in range(n_puntos):
            # Si el punto j domina al punto i
            # np.all(..) es true si todos son true. np.any(...) estrue si al menos uno es true
            if np.all(fitness_scores[j] >= fitness_scores[i]) and np.any(fitness_scores[j] > fitness_scores[i]):
                es_pareto[i] = False
                break
    
    return fitness_scores[es_pareto]

def extraccion_caracteristicas_compromiso(ruta_experimento):
    '''
        Argumento: Ruta de archivo de experimento de AG. Devuelve lista de variables seleccionadas 
        de la solucion mas cerca al punto utopico del espacio de soluciones.
    '''
    from ejecucion_AG_TI import mutation_func, fitness_exp, callback_generacion
    
    #Cargar nombre de variables
    with open('./nombre_variables_b_flavor.pkl', 'rb') as f:
        nombres_variables = pickle.load(f)

    #cargar resultados de AG a analizar
    with open(ruta_experimento, 'rb') as f:
        resumen = pickle.load(f)


    nombres_variables.keys()
    nombre_caracteristicas = nombres_variables['nombre_caracteristicas']  
    nombre_clase = nombres_variables['nombre_clase'] 

    fitness_ultima_generacion = resumen['fitness_final']
    poblacion_ultima_generacion = resumen['poblacion_final']
    fitness_pareto = obtener_fitness_pareto(fitness_ultima_generacion)

    idx_individuo_seleccionado, fitness_mejor_individuo = seleccionar_codo_punto_utopia(fitness_pareto)


    #guardamos la exactitud del mejor individuo para almacenarla y usarla para encontrar al individuo
    mejor_exactitud = fitness_mejor_individuo[0]

    idx_original = np.where(fitness_ultima_generacion[:, 0] == mejor_exactitud)[0][0]
    mascara_ganadora = poblacion_ultima_generacion[idx_original]

    indices_activos = np.where(mascara_ganadora == 1)[0]
    variables_seleccionadas = [nombre_caracteristicas[i] for i in indices_activos]

    return variables_seleccionadas