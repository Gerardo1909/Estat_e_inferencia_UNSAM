'''
Este script contiene funciones para graficar distribuciones de probabilidad.
'''

import numpy as np
import matplotlib.pyplot as plt


def graficar_pdf_scipy(dist, dist_name):
    """
    Se encarga de graficar la PDF de una distribución de probabilidad de scipy.
    """
    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 1000)
    pdf = dist.pdf(x)
    median = dist.median()
    mean = dist.mean()
    

    fig, ax = plt.subplots()

    ax.plot(x, pdf, color="k", label='PDF')
    
    ax.axvline(median, color='r', linestyle='--', label=f'median: {median:.2f}')
    ax.axvline(mean, color='m', linestyle='--', label=f'Mean: {mean:.2f}')
    
    ax.grid(True)
    plt.title(f'Gráfico de la PDF de {dist_name}')
    plt.legend()
    plt.show()
    
def graficar_cdf_scipy(dist, dist_name):
    """
    Se encarga de graficar la CDF de una distribución de probabilidad de scipy.
    """
    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 1000)
    cdf = dist.cdf(x)
    median = dist.median()
    mean = dist.mean()
    
    fig, ax = plt.subplots()

    ax.plot(x, cdf, color="k", label='CDF')
    
    ax.axvline(median, color='r', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(mean, color='m', linestyle='--', label=f'Mean: {mean:.2f}')
    
    ax.grid(True)
    plt.title(f'Gráfico de la CDF de {dist_name}')
    plt.legend()
    plt.show()