#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.random import uniform, choice, exponential
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import time as tm 
import networkx as nx
import grafos as gf 
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime

plt.style.use('ggplot')

def simula_sis_dina(graph, ti, tf, P, λ, δ):
   
    #seed(int(semente))
    #ss = SeedSequence(semente)
    #PCG64(ss)
    
    ns         = {}
    vizinhos   = {}
    
    V, nos, G  = graph
    
    xs, xi, xr = [], [], []
    VS, VI, VR = [], [], []
    Te, ε      = [], []
   
    Ni, Ns, t, μ = 0, 0, 0, 0
    
    # inicia alguns vertices no estado infectado
    for i in V:
        m = uniform(0, 1)
        if m <= P:
            VI.append(i)  # vertices infectados
        else:
            VS.append(i)  # vertices suscetiveis
        
        vizinhos[i]  = set(list(nx.neighbors(G, i)))
        
    while len(VI) > 0:

        Ni = len(VI)  # quantidade de vertices infectados
        Ns = 0
        ε.clear()
        vs      = set(VS)
        for i in VI:
           
            ns[i]   = len(vs & vizinhos[i])
            Ns = Ns + ns[i]
           
       
        ε = [((δ + ns[i] * λ) / (Ni * δ + Ns * λ)) for i in VI]
        
       
        
        # incrementa o tempo
        dt = exponential(1.0 / (Ni * δ + Ns * λ)) 
    
        # tempo total 
        t = t + dt 
         
        # sorteia um vertice infectado
        i = choice(VI, 1, True, ε)[0]

        # quantidade de vizinhos suscetiveis de o vertice infectado i.
       

        # probabilidade do vertice i ser curado
        μ = δ / (δ + ns[i] * λ)

        # gera um β ~ U(0,1)
        β = uniform(0, 1)

        if β <= μ:
            VI.remove(i)
            VS.append(i)
            ns.pop(i)  # remove o vertice i

        elif β > μ:
            # sorteia um dos vizinhos
            n = list(vs & vizinhos[i])
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
                
       
        if t >= ti:
            Te.append(t)
            xs.append(len(VS))
            xi.append(len(VI))
            xr.append(len(VR))

        if t >= tf:
            break
            
    return [Te, xs ,xi, xr]


def simula_sir_dina(graph, ti, tf, λ, δ):
   
    V, nos, G  = graph
    
    ns         = {}
    vizinhos   = {}
    
    xs, xi, xr = [], [], []
    VS, VI, VR = [], [], []
    Te, ε      = [], []
    
    Ni, Ns, t, μ = 0, 0, 0, 0
    
    j = choice(V, 1)[0]
    VI.append(j)
    
    # inicia alguns vertices no estado infectado
    for i in V: 
        vizinhos[i]  = set(list(nx.neighbors(G, i)))
        if i != j:
            VS.append(i)
        
        
    Te.append(t)
    xs.append(len(VS))
    xi.append(len(VI))
    xr.append(len(VR))
        
    while len(VI) > 0:

        Ni = len(VI)  # quantidade de vertices infectados
        Ns = 0
        ε.clear()
        vs      = set(VS)
        for i in VI:
            ns[i]   = len(vs & vizinhos[i])
            Ns = Ns + ns[i]
                  
        ε = [((δ + ns[i] * λ) / (Ni * δ + Ns * λ)) for i in VI]
        
        # incrementa o tempo
        dt = exponential(1.0 / (Ni * δ + Ns * λ)) 
    
        # tempo total 
        t = t + dt 
         
        # sorteia um vertice infectado
        i = choice(VI, 1, True, ε)[0]

        # probabilidade do vertice i ser curado
        μ = δ / (δ + ns[i] * λ)

        # gera um β ~ U(0,1)
        β = uniform(0, 1)

        if β <= μ:
            VI.remove(i)
            VR.append(i)
            ns.pop(i)  # remove o vertice i

        elif β > μ:
            # sorteia um dos vizinhos
            #n = sorted(list(vs & vizinhos[i]))
            n = list(vs & vizinhos[i])
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
                
       
        if t >= ti:
            Te.append(t)
            xs.append(len(VS))
            xi.append(len(VI))
            xr.append(len(VR))

        if t >= tf:
            break
            
    return [Te, xs ,xi, xr]


def simula_mkt_dina(graph, ti, tf, α, λ):
   
    V, nos, G  = graph
    k          = {} 
    ns         = {}
    vizinhos   = {}
    j = 0
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
            n = sorted(list(vs & vizinhos[i]))
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
                
       
        if t >= ti:
            Te.append(t)
            xs.append(len(VS))
            xi.append(len(VI))
            xr.append(len(VR))

        if t >= tf:
            break
            
    return [Te, xs ,xi, xr]


def F(k):
    N      =  100
    tmf = 0
    tmp = 0
    grafo = gf.grafo_dente_dleao(610, 610)
    l = 20.0/610.0
    for i in range(N):
        t, xs, xi, xr =  simula_sir_dina(grafo, 0, 100,0.1, l, 1.0) 
        x = np.array(xi).min()
        idx = xi.index(x)
        tmf += t[idx]
        
        x = np.array(xi).max()
        idx = xi.index(x)
        tmp += t[idx]
        
    return [tmp/N, tmf/N]
    

print('\n\n Simulação MKT - grafo k regular \n\n')

tamanho = [2, 22]

 
inicio  = tm.time()
N = tqdm(tamanho)
num_cores = multiprocessing.cpu_count()
processed_list = Parallel(n_jobs = (num_cores))(delayed(F)(n) for n in N)
processed_list = np.array(processed_list)
final  = tm.time() 

print(final - inicio)
print(processed_list)

arquivo = open('dados_tempo_grafo_dente_dleao_kxk_sir.dat', 'w')

  
for i in processed_list:
    arquivo.write(f'{str(i)} \n')  
    
arquivo.close()

hora =  datetime.now().strftime('%H:%M:%S')
plt.figure(figsize=(10, 7))
plt.plot(tamanho, processed_list, '--o', label = ['Tempo de pico', 'Tempo de extinção'])
plt.xlabel('n', fontsize = 25)
plt.ylabel('Tempo', fontsize = 25)
plt.legend(fontsize = 20)
#plt.savefig('grafo_dente_dleao_kxk_sir_'+hora+'.png', format='png')
