import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
from funkcje import generujTabliceGeometrii as GenTab
from funkcje import rysuj_geometrie as rGeo
from funkcje import alokacja_pamieci_na_zmienne_globalne as Alok
from funkcje import funkcje_bazowe as FBaz
from funkcje import elementy_macierzy as ElMat
from funkcje import rysuj_rozwiazanie as Rezult
#Preprocesing

#Parametry sterujące
c = 0
f = lambda x:0
twb_L = 'D'
twb_R = 'D'


x_a = 1
x_b = 2
n = 5

wez, el = GenTab(x_a, x_b, n)
print('Tablica węzłów:\n',wez)
print('Tablica elementów:\n',el)

WB    = [{"ind": 1, "typ":'D', "wartosc":1}, 
         {"ind": n, "typ":'D', "wartosc":2}]

rGeo(wez,el,WB)

#Procesing

A,b = Alok(n)
#print(A)
#print(b)

SW = 1 #stopien wielomianu
phi,dphi = FBaz(SW)
xx = np.linspace(-1,1,101)
plt.plot(xx,phi[0](xx),'r')
plt.plot(xx,phi[1](xx),'g')
plt.plot(xx,dphi[0](xx),'b')
plt.plot(xx,dphi[1](xx),'c')

#PROCESING
liczbaElementow = np.shape(el)[0]
for ee in np.arange(0, liczbaElementow ):
    
    EIR = ee
    EIG = el[ee,0]
    EW1 = el[ee,1]
    EW2 = el[ee,2]
    IGW = np.array([EW1,EW2])
    
    x_a = wez[EW1-1,1]
    x_b = wez[EW2-1,1]
    J = (x_b-x_a)/2
    
    Ml = np.zeros([SW+1,SW+1])
    
    n = 0 
    m = 0
    Ml[n,m]=J*spint.quad(ElMat(dphi[n],dphi[m],c,phi[n],phi[m]),-1,1)[0]
    n = 0 
    m = 1
    Ml[n,m]=J*spint.quad(ElMat(dphi[n],dphi[m],c,phi[n],phi[m]),-1,1)[0]
    n = 1
    m = 0
    Ml[n,m]=J*spint.quad(ElMat(dphi[n],dphi[m],c,phi[n],phi[m]),-1,1)[0]
    n = 1 
    m = 1
    Ml[n,m]=J*spint.quad(ElMat(dphi[n],dphi[m],c,phi[n],phi[m]),-1,1)[0]
    
    A[np.ix_(IGW-1, IGW-1  ) ] =  \
            A[np.ix_(IGW-1, IGW-1  ) ] + Ml

print(A)
print(WB)

# UWZGLEDNIENIE WARUNKOW BRZEGOWYCH    
if WB[0]['typ'] == 'D':
    indw = WB[0]['ind']
    wwb = WB[0]['wartosc']
        
    iwp = indw - 1
        
    wzm = 10**14
        
    b[iwp] = A[iwp,iwp]*wzm*wwb
    A[iwp, iwp] = A[iwp,iwp]*wzm
        
        
if WB[1]['typ'] == 'D':
    indw = WB[1]['ind']
    wwb = WB[1]['wartosc']
        
    iwp = indw - 1
        
    wzm = 10**14
        
    b[iwp] = A[iwp,iwp]*wzm*wwb
    A[iwp, iwp] = A[iwp,iwp]*wzm        
    
    
if WB[0]['typ'] == 'N':
    print('Nie zaimplementowano jeszcze. Zad.dom')
    
    
if WB[1]['typ'] == 'N':
    print('Nie zaimplementowano jeszcze. Zad.dom')    
    
    
    # print(A)
    # print(b)
    
    
    
    # Rozwiazanie ukl row lin    
u = np.linalg.solve(A,b)
    
print(u)
Rezult(wez, el, WB, u)