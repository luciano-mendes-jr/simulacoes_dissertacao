#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.random import uniform, choice, exponential
from numpy.random import PCG64, SeedSequence
import networkx as nx
import numpy as np 

def simula_sis_cor(graph, ti, tf, P, λ, δ):
    
    V, nos, G = graph
    μ = 0.0

    VS, VI     = [], []
    n, Te, ε   = [], [], []
    X0, X1, T  = [], [], []
   
    t = 0
   
    Ni, Ns = 0, 0
    ns       = {}
    vizinhos = {}
    
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
        
        for i in VI:
            vs      = set(VS)
            ns[i]   = len(vs & vizinhos[i])
            Ns = Ns + ns[i]
           
              
        ε = [((δ + ns[i] * λ) / (Ni * δ + Ns * λ)) for i in VI]
        
       
        # incrementa o tempo
        dt = exponential(1.0 / (Ni * δ + Ns * λ)) 
        
        Te.append(dt)
        
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
            n = sorted(list(vs & vizinhos[i]))
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
        
        
        if t >= ti:
            T.append(Te[-2])
            if 0 in VS:
                X0.append(0) 
            elif 0 in VI:
                X0.append(1)
            if 1 in VS:
                X1.append(0)
            elif 1 in VI:
                X1.append(1)
                
        if t >= tf: 
            break

    return [np.array(X0), np.array(X1), np.array(T), t]


def simula_sir_cor(graph, ti, tf, λ, δ):
    

    V, nos, G = graph
    Nv, μ = len(V), 0

    VS, VI, VR = [], [], []
    n, Te, ε   = [], [], []

    ns, t  = 0, 0
    Ni, Ns = 0, 0

    X0 = []
    X1 = []
    T  = []
    ns       = {}
    vizinhos = {}
    
    j = choice(V, 1)[0]
    VI.append(j)
    
    # inicia alguns vertices no estado infectado
    for i in V: 
        vizinhos[i]  = set(list(nx.neighbors(G, i)))
        if  i != j:
            VS.append(i)
                
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
        
        Te.append(dt)
        
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
            n = sorted(list(vs & vizinhos[i]))
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
        
        
        if t >= ti and len(Te) > 1:
            T.append(Te[-2])
            if 0 in VI:
                X0.append(1) 
            else:
                X0.append(0)
            if 1 in VI:
                X1.append(1) 
            else:
                X1.append(0) 
                
                
        if t >= tf: 
            break

    return [np.array(X0), np.array(X1), np.array(T), t]


def simula_mkt_cor(graph, ti, tf, λ, α):
   
    
    V, nos, G = graph
    μ =  0
    VS, VI, VR = [], [], []
    n, Te, ε   = [], [], []

    t  = 0
    Nri, Ns = 0, 0

    X0 = []
    X1 = []
    T  = []
    ns       = {}
    k        = {}
    vizinhos = {}
    
    j = choice(V, 1)[0]
    VI.append(j)
    
    # inicia alguns vertices no estado infectado
    for i in V: 
        vizinhos[i]  = set(list(nx.neighbors(G, i)))
        k[i] = len(vizinhos[i])
        if  i != j:
            VS.append(i)
                
    while len(VI) > 0:
      
        Ns  = 0
        Nri = 0
        
        ε.clear()
        vs      = set(VS)
              
        for i in VI:
            ns[i]   = len(vs & vizinhos[i])
            Nri     = Nri + (k[i] - ns[i])
            Ns = Ns + ns[i]
                  
        ε = [(((k[i] - ns[i])*α	+ ns[i] * λ) / (Nri * α + Ns * λ)) for i in VI]
           
        
        
        # incrementa o tempo
        dt = exponential(1.0 / (Nri * α + Ns * λ)) 
        
        Te.append(dt)
        
        t = t + dt 

        # sorteia um vertice infectado
        i = choice(VI, 1, True, ε)[0]


        # probabilidade do vertice i ser curado
        μ = (k[i] - ns[i])*α / ((k[i] - ns[i])*α + ns[i]*λ)

        # gera um β ~ U(0,1)
        β = uniform(0, 1)

        if β <= μ:
            VI.remove(i)
            VR.append(i)
            ns.pop(i)  # remove o vertice i
           
            
        elif β > μ:
            n = list(vs & vizinhos[i])
            j = choice(n, 1)[0]
            VS.remove(j)
            VI.append(j)
        
        
        if t >= ti and len(Te) > 1:
            T.append(Te[-2])
            if 0 in VI:
                X0.append(1) 
            else:
                X0.append(0)
            if 1 in VI:
                X1.append(1) 
            else:
                X1.append(0) 
        
                
        if t >= tf: 
            break

    return [np.array(X0), np.array(X1), np.array(T), t]

