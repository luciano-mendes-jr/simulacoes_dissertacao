from joblib import Parallel, delayed
import estima_correlacao as estc
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import grafos as gf
import numpy as np
import time as tm 
plt.style.use("ggplot")


def salva_dados(tempo, COV, COR): 
    arquivo = open('cov_grafo_nxn_mkt.dat', 'w')
    for i in range(len(tempo)):
        arquivo.write(f'{str(tempo[i])}:{str(COV[i])} \n')  
    arquivo.close()

    arquivo = open('cor_grafo_nxn_mkt.dat', 'w')
    for i in range(len(tempo)):
        arquivo.write(f'{str(tempo[i])}:{str(COR[i])} \n')  
    arquivo.close()
    
    
def CV(X0, X1, dt):
    
    T        =  sum(dt) 
    med_X0X1 =  sum(X0*X1*dt)/T 
    med_X0   =  sum(X0*dt)/T 
    med_X1   =  sum(X1*dt)/T 
    
    med2_X0 = sum((X0**2)*dt)/T 
    med2_X1 = sum((X1**2)*dt)/T 
    
    v_X0   =  (med2_X0  - (med_X0)**2)**0.5
    v_X1   =  (med2_X1  - (med_X1)**2)**0.5
    
    cv       =   med_X0X1 - (med_X0*med_X1)
                
    cr = cv/(v_X0*v_X1)
    
    return [cv,  cr]

def MT(entrada):
    n, t  = entrada 
    k     = 0
    Rp    = 100
    grafo = gf.grafo_dente_dleao(n, n)
    x0, x1, Dt = [], [], []
    ti = 0.01
    tf = t

    while k < Rp:
        
        X0, X1, dt, t_f = estc.simula_mkt_cor(grafo, ti, tf, 1.0, 1.0)
        
        if sum(dt) >= (tf - ti):
            x0 = x0 + list(X0)
            x1 = x1 + list(X1)
            Dt = Dt + list(dt)
            k += 1
                     
    X0 = np.array(x0)
    X1 = np.array(x1)
    dt = np.array(Dt)
    
    cv, cr = CV(X0, X1, dt)
     
    return [cv, cr] 
    

print('\n\nModelo MKT - Grafo n x n \n\n')

K  =  [i for i in range(70, 1030, 60)]
T  = tqdm([3.92667368, 4.19577559, 4.37921436, 4.49485771,
           4.5967475,  4.6980381,  4.75436769, 4.80351794,
           4.84805218, 4.91969252, 4.95135609, 4.98544613,
           5.04000262, 5.05922725, 5.09443537, 5.12441238])

num_cores = multiprocessing.cpu_count()
inicio    = tm.time()
processed_list = Parallel(n_jobs = (num_cores))(delayed(MT)((n, t)) for n, t in zip(K, T))
final     = tm.time()
processed_list = np.array(processed_list)
COV   = processed_list[:, 0]
COR   = processed_list[:, 1]
t     = (final - inicio)

print(f'Tempo de execução: {t:.3f}s')

plt.figure(figsize=(10, 7))
plt.plot(K, COV , '--o')
plt.xlabel('n', fontsize = 25)
plt.ylabel(r'$COV_n \left(X_0, X_1\right)$', fontsize = 20)
plt.savefig('cov_grafo_nxn_mkt.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(K, COR, '--o', color = 'black')
plt.xlabel('n', fontsize = 25)
plt.ylabel(r'$COR_n \left(X_0, X_1\right)$', fontsize = 20)
plt.savefig('cor_grafo_nxn_mkt.pdf', format='pdf')
plt.show()

salva_dados(K, COV, COR)
