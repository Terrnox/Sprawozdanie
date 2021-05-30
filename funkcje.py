import numpy as np
import matplotlib.pyplot as plt

def generujTabliceGeometrii(x_0, x_p, n):
    temp = (x_p - x_0) / (n - 1)
    matrix = np.array([1, 0 * temp + x_0])

    for i in range(1, n, 1):
        matrix = np.block([
            [matrix],
            [i + 1, i * temp + x_0],
        ])

    matrix2 = np.array([1, 1, 2])

    for i in range(1, n - 1, 1):
        matrix2 = np.block([
            [matrix2],
            [i + 1, i + 1, i + 2],
        ])

    return matrix, matrix2

def alokacja_pamieci_na_zmienne_globalne(n):
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    return A, b

def rysuj_geometrie(wez, el, WB):
    fh = plt.figure()
    
    plt.plot(wez[:,1], np.zeros( (np.shape(wez)[0], 1) ), '-b|' )
        
    nodeNo = np.shape(wez)[0]
        
    for ii in np.arange(0,nodeNo):
        
        ind = wez[ii,0]
        x = wez[ii,1]
        plt.text(x, 0.01, str( int(ind) ), c="b")
        plt.text(x, -0.01, str(x))
     
    elemNo = np.shape(el)[0]
    for ii in np.arange(0,elemNo):

        wp = el[ii,1]
        wk = el[ii,2]

        x = (wez[wp-1,1] + wez[wk-1,1] ) / 2  
        plt.text(x, 0.01, str(ii+1), c="r")

    plt.show()
    return fh
    
def funkcje_bazowe(n):
    if n==0:
        f = lambda x: x*0 + 1
        df = lambda x: x*0
    elif n==1:
        f = (lambda x: -1/2*x+1/2, lambda x: 1/2*x+1/2)
        df = (lambda x: -1/2+x*0, lambda x: 1/2+x*0)
    else:
        raise Exception("Nieobsługiwany wielomian")
    return f,df

def elementy_macierzy(dphi1,dphi2,c,phi1,phi2):
    Aij = lambda x: -dphi1(x)*dphi2(x)+ c *phi1(x)*phi2(x)
    return Aij

def rysuj_rozwiazanie(wez, el, WB, u):
    
    from funkcje import rysuj_geometrie
    
    rysuj_geometrie(wez,el,WB)
    
    x = wez[:,1]
    
    plt.plot(x, u, 'm*')