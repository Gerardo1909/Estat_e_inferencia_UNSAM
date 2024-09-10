'''
Este script contiene funciones para estimar regresiones lineales.
'''

import pandas as pd
import numpy as np

def generar_regresion_lineal_simple(X:pd.Series, Y:pd.Series) -> tuple[float, float, list[float]]:
    '''
    Estima los coeficientes de una regresión lineal simple, B1 y B0, 
    junto a los errores de estimación a partir de dos series de datos X e Y.
    '''
    
    # Calculo las medias de X e Y
    X_promedio = X.mean()
    Y_promedio = Y.mean()
    
    # Calculo el numerador y el denominador de la fórmula de B1
    numerador = sum((X - X_promedio)*(Y - Y_promedio))
    denominador = sum((X - X_promedio)**2)
    
    # Calculo B1
    B1 = numerador / denominador
    
    # Calculo B0
    B0 = Y_promedio - B1*X_promedio
    
    # Calculo el error de estimación
    errores = Y - (B0 + B1 * X)
    
    return B1, B0, errores.tolist() 

estimar_bethas = lambda X, Y: np.linalg.inv(X.T @ X) @ X.T @ Y

def calcular_sse(X, Y, betas):
    """Calcula la suma de los errores al cuadrado (SSE)."""
    predicciones = X @ betas
    errores = Y - predicciones
    SSE = np.sum(errores ** 2)
    return SSE

def calcular_sigma2(X, Y, betas):
    """Calcula la varianza estimada del modelo completo."""
    n = len(Y)
    SSE = calcular_sse(X, Y, betas)
    sigma2 = SSE / (n - X.shape[1])  # Ajuste de grados de libertad
    return sigma2

def mallows_cp(X_train, Y_train, X_test, Y_test, betas_full):
    """
    Calcula el estadístico de Mallows Cp.
    
    Parámetros:
    X_train: Subconjunto de predictores para entrenar el modelo.
    Y_train: Valores de la variable dependiente del conjunto de entrenamiento.
    X_test: Conjunto de predictores completo.
    Y_test: Valores de la variable dependiente del conjunto completo.
    betas_full: Betas estimadas para el modelo completo.

    Retorna:
    cp: Estadístico Cp de Mallows.
    """
    # Estimo los betas del subconjunto
    betas_subset = estimar_bethas(X_train, Y_train)
    
    # Calculo la suma de los errores cuadráticos (SSE) del modelo con el subconjunto
    SSE_p = calcular_sse(X_train, Y_train, betas_subset)
    
    # Calculo la varianza estimada (sigma^2) usando el modelo completo
    sigma2 = calcular_sigma2(X_test, Y_test, betas_full)
    
    # Tomo la cantidad de predictores del modelo ajustado en el subconjunto, incluyendo el intercepto
    p = X_train.shape[1] 
    
    # Tomo la cantidad de observaciones
    n = len(Y_train)
    
    # Calculo el estadístico Cp
    return (SSE_p / sigma2) + 2 * p - n