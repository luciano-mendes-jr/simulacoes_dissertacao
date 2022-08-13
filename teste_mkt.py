# -*- coding: utf-8 -*-



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import statsmodels.api as smi
from tqdm import tqdm
from   scipy import stats
import pylab
import numpy as np
import time as tm 
from   numpy.random import uniform, choice, exponential, seed
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")

def grafo_completo(n, plot = False):
    
    G = nx.complete_graph(n)    
    lista_nos = list(G.edges())
    V = list(G.nodes())
   
    
    if plot == True:
            plt.figure(figsize=(5, 5), dpi = 80)
            nx.draw_circular(G,node_size  = 500, 
                    node_color  = '#4A90E2', 
                    with_labels = True, 
                    width = 1.5)
            
            plt.show()
        
    return[V, lista_nos, G]

def simula_mkt_dina(graph ,ti, tf, α, λ):
   
    V, nos, G  = graph
    k          = {} 
    ns         = {}
    vizinhos   = {}
    
    xs, xi, xr = [], [], []
    VS, VI, VR = [], [], []
    Te, ε      = [], []
    
    Nri, Ns, t, μ = 0, 0, 0, 0
    
    j = choice(V, 1)[0]
    VI.append(j)
    
    # inicia alguns vertices no estado infectado
    for i in V: 
        vizinhos[i]  = set(list(nx.neighbors(G, i)))
        k[i]         = len(vizinhos[i])
        if  i != j:
            VS.append(i)


    Te.append(t)
    xs.append(len(VS))
    xi.append(len(VI))
    xr.append(len(VR))
        
    while len(VI) > 0:
        Nri = 0  # quantidade de vertices infectados
        Ns  = 0
        vs  = set(VS)
        ε.clear()
      
        for i in VI:
            ns[i]   = len(vs & vizinhos[i])
            Nri     = Nri + (k[i] - ns[i])
            Ns = Ns + ns[i]
                  
        ε = [(((k[i] - ns[i])*α	+ ns[i] * λ) / (Nri * α + Ns * λ)) for i in VI]
        
        # incrementa o tempo
        dt = exponential(1.0 / (Nri * α + Ns * λ)) 
    
        # tempo total 
        t = t + dt 
         
        # sorteia um vertice infectado
        i = choice(VI, 1, True, ε)[0]

        # probabilidade do vertice i ser curado
        μ = (k[i] - ns[i])*α / ((k[i] - ns[i])*α + ns[i] * λ)

        # gera um β ~ U(0,1)
        β = uniform(0, 1)

        if β <= μ:
            VI.remove(i)
            VR.append(i)
            ns.pop(i)  # remove o vertice i

        elif β > μ:
            # sorteia um dos vizinhos
            n = list(vs & vizinhos[i])
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
              
        Te.append(t)
        xs.append(len(VS))
        xi.append(len(VI))
        xr.append(len(VR))

    return xs[-1]/len(V)

def MKT(gf, i):
    seed(i)
    xs =  simula_mkt_dina(gf, 0, 10000, 1.0, 1.0)
    return xs 

sementes = [] 

for i in range(1000):
    i = i  + 1
    m = tm.time() 
    sementes.append(int(m)) 
    tm.sleep(1.0)

N      = tqdm(sementes)
grafo  = grafo_completo(2000, plot = False)

num_cores = multiprocessing.cpu_count()
print(f'\n \n Num de processadores {num_cores} \n \n')

inicio    = tm.time()
processed_list = Parallel(n_jobs = (num_cores))(delayed(MKT)(grafo, n) for n in N)
final     = tm.time()
s = np.array(processed_list)

print('tempo de simulação: ', final - inicio)
print('Média = ', s.mean())

mu     = s.mean()
k      = 1
sigma  = ((mu*(1 - mu))/(1 - (k + 1)*mu))
print('Var   = ', sigma)
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('',fontsize = 25.0) 
ax.set_xlabel('',fontsize = 25.0) 
smi.qqplot(s, line = "r", loc = mu, 
           scale  = sigma, 
           ylabel = 'Quantis Empíricos',
           xlabel = 'Quantis Teóricos',
           ax = ax)

plt.savefig('qqplot.pdf', format='pdf')
pylab.show()

shapiro_test = stats.shapiro(s)

print(shapiro_test)

print(shapiro_test)
