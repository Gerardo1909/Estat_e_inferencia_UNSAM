'''
Este script contiene funciones para calcular medidas sobre distribuciones de probabilidad.
'''
import numpy as np

def calcular_moda(dist):
    """
    Calcula la moda de una distribución continua de scipy.stats.
    """
    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 1000)
    pdf = dist.pdf(x)
    moda = x[np.argmax(pdf)]
    return moda


def gen_media_n(n: int, distribucion) -> float:
    """
    Calcula la media muestral de una distribución dada.
    Parámetros:
    n (int): El número de valores aleatorios a generar a partir de la distribución.
    distribucion: El objeto de distribución del cual generar los valores aleatorios.
    Retorna:
        La media muestral de los valores generados.
    """
    # Genero n valores aleatorios de la distribución
    valores = distribucion.rvs(size=n)
    
    # Calculo la media muestral
    media = sum(valores) / n
    
    return media