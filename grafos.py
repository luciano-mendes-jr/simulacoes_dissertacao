#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn.neighbors as sknn
import networkx as nx
import matplotlib.pyplot as plt  
from numpy import  zeros, arange, array
from math import pi, sin, cos
import numpy as np 

def grafo_k_regular(K, Nv, plot = False):
    
    i = 0   
    pontos = zeros((Nv,2),dtype = 'float') #Define a matriz n x n 
    
    ANG = arange(0, 2*pi, 2*pi/Nv)
    G = nx.Graph()
    
    #gera as coordenadas dos vertices  
    for ang in ANG:
        pontos[i][0] =  cos(ang)
        pontos[i][1] =  sin(ang)
        G.add_node(i)            #cria os vertices 
        i = i + 1
        
    # Cria um grafo utilizando a regra KNN
    knnGraph = sknn.kneighbors_graph(pontos, 
                                     n_neighbors = K, 
                                     mode  =  'connectivity',
                                     p = 2)
    
    #Converte para matriz de adjacências
    adj = knnGraph.toarray()
    
    #gera o grafo a partir da matriz de adjacencias 
    for i in range(Nv):
        for j in range(Nv):
            if(adj[i][j] == 1 ):
                G.add_edge(i,j) #conecta os vertices 
    
    if (nx.is_k_regular(G,K)) == False:
        print("Grafo regular: {}".format(nx.is_k_regular(G,K)))
    
    V = list(G.nodes())
   
    if plot == True:
        
        plt.figure(figsize=(5, 5), dpi = 80) 
        nx.draw_circular(G,node_size = 500, 
                         node_color  =  '#4A90E2',
                         with_labels = True, 
                         width = 1.5)
        plt.show()
            
    lista_nos = list(G.edges())    
    return[V, lista_nos, G]

def grafo_k_regular_aleatorio(k, Nv, seed = 1995, plot = False):
    
    G = nx.random_regular_graph(k, Nv, seed)
    lista_nos = list(G.edges())
    V = list(G.nodes())
    #adj = array(nx.adjacency_matrix(G).todense())
    
    if plot == True:
        np.random.seed(2000)
        nx.draw(G, node_size = 500, 
                         node_color   = '#4A90E2', 
                         with_labels  = True, 
                         width = 1.5)
        plt.show()
    
    return[V, lista_nos, G]

def grafo_binomial(Nv, plot = False):
    
    np.random.seed(1930)
    n = int(np.log(Nv)/np.log(2))
    
    if 2**n == Nv:
        
        G = nx.binomial_tree(n)    
        lista_nos = list(G.edges())
        V = list(G.nodes())
        #adj = array(nx.adjacency_matrix(G).todense())
        
        if plot == True:
            plt.figure(figsize=(5, 5), dpi = 80)
            nx.draw(G,node_size  = 500, 
                    node_color  = '#4A90E2', 
                    with_labels = True, 
                    width = 1.5)
       
            plt.show()
        
        return[V, lista_nos, G]
     
    else:
        print("Entre com um parametro correto")
        return 0 
        
def grafo_estrela(n, plot = False):
    
    G = nx.star_graph(n)    
    lista_nos = list(G.edges())
    V = list(G.nodes())
    #adj = array(nx.adjacency_matrix(G).todense())
    
    if plot == True:
            plt.figure(figsize=(5, 5), dpi = 80)
            nx.draw(G,node_size  = 500, 
                    node_color  = '#4A90E2', 
                    with_labels = True, 
                    width = 1.5)
       
            plt.show()
        
    return[V, lista_nos, G]

def grafo_roda(n, plot = False):
    
    G = nx.wheel_graph(n)    
    lista_nos = list(G.edges())
    V = list(G.nodes())
    #adj = array(nx.adjacency_matrix(G).todense())
    
    if plot == True:
            plt.figure(figsize=(5, 5), dpi = 80)
            nx.draw(G,node_size  = 500, 
                    node_color  = '#4A90E2', 
                    with_labels = True, 
                    width = 1.5)
       
            plt.show()
        
    return[V, lista_nos, G]

def grafo_dente_dleao(n, k, plot = False):

    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    
    N =  n + k + 2 

    for i in range(2, k + 2):
        G.add_edge(0, i)   
      
    for i in range(k + 2, N):
        G.add_edge(1, i)
       
        
    lista_nos = list(G.edges())
    V = list(G.nodes())
   # adj = array(nx.adjacency_matrix(G).todense())
    
    if plot == True:
            plt.figure(figsize=(5, 5), dpi = 80)
            nx.draw(G,node_size  = 500, 
                    node_color  = '#4A90E2', 
                    with_labels = True, 
                    width = 1.5)
       
            plt.show()
        
    return[V, lista_nos, G]

def grafo_dente_dleao_expandido(k, n, plot = False):
    
       G = nx.Graph()
       N = 2*k + 2 

       for i in range(2, k + 2):
           G.add_edge(0, i)    
    
       for i in range(k + 2, N):
           G.add_edge(1, i)
    
       for i in range(N, N+n-1):
           G.add_edge(i, i + 1)
    
       G.add_edge(0, N+n-1)
       G.add_edge(1, N)
           
       lista_nos = list(G.edges())
       V = list(G.nodes())
       #adj = array(nx.adjacency_matrix(G).todense())
    
       if plot == True:
           plt.figure(figsize=(5, 5), dpi = 80)
           nx.draw(G,node_size  = 500, 
                   node_color  = '#4A90E2', 
                   with_labels = True, 
                   width = 1.5)
           
           plt.show()
        
       return[V, lista_nos, G]
   
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




