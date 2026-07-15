import pygad
import time
import numpy as np
import pickle
import pandas as pd
import gc
import funciones
import inconsistencia_optimizada # Tu módulo de C++

# Carga de datos .npy
X_path = 'X_train_10k_DISC.npy'
y_path = 'y_train_10k_8020.npy'

# Variables globales para paralelizacion
global_X = np.load(X_path)
global_y = np.load(y_path).astype(np.int32).ravel()
global_n_features = global_X.shape[1] # 190

#--------------------------------------------
# FUNCION FITNESS 
#--------------------------------------------
def fitness_exp(ga_instance, solution, solution_idx):

    return funciones.fitness_wrapper_lgb(solution, global_X, global_y)

historial_promedios = []
def callback_generacion(ga_instance):
    # En NSGA-II, el fitness es una matriz [n_individuos, n_objetivos]
    last_fitness = ga_instance.last_generation_fitness
    avg_fitness = np.mean(last_fitness, axis=0)

    historial_promedios.append(avg_fitness.copy())
    print(f"Gen {ga_instance.generations_completed} | "
          f"Calidad Prom: {avg_fitness[0]:.4f} | Reducción Prom: {avg_fitness[1]:.4f}")

def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() <= ga_instance.mutation_probability:
                # El flip: 1 - valor_actual
                offspring[chromosome_idx, gene_idx] = 1 - offspring[chromosome_idx, gene_idx]
    return offspring
# Ejecucion del Algoritmo Genetico

if __name__ == "__main__":
    start_time = time.time()
    
    sol_per_pop = 50
    num_genes = global_n_features # 190
    
    # Población inicial incluye al individuo con todas las variables activas
    poblacion_inicial = np.random.randint(low=0, high=2, size=(sol_per_pop, num_genes))
    poblacion_inicial[0, :] = 1 
    
    ga_instance = pygad.GA(
                num_generations=50, 
                num_parents_mating=50,
                initial_population=poblacion_inicial,
                fitness_func=fitness_exp,    
                on_generation= callback_generacion,       
                parallel_processing=["process", 8],
                sol_per_pop=50,
                num_genes=190,
                gene_type=int,
                gene_space = [0,1],
                mutation_type = mutation_func,
                #init_range_low=0,
                #init_range_high=2,
                parent_selection_type="tournament_nsga2",
                mutation_probability=0.000554 #equivalente a 0.1 por individuo
        )
    
    ga_instance.run()
    
    end_time = time.time()
    
    #-------------------------------------------
    # ARCHIVOS DE RESULTADOS
    #-------------------------------------------
    #Instancia de pyGAD
    ga_instance.save(filename="./resultados/instancia_10k_hibrida_cpp")
    
    # Informacion extra
    resultados_finales = {
        'fitness_final': ga_instance.last_generation_fitness,
        'poblacion_final': ga_instance.population,
        'historial_promedios': np.array(historial_promedios),
        'tiempo_total': end_time - start_time,
        'config': ga_instance.__dict__
    }
    
    with open('./resultados/resumen_10k_hibrida.pkl', 'wb') as f:
        pickle.dump(resultados_finales, f)

    print(f"\n Proceso completado en {end_time - start_time:.2f} segundos.")