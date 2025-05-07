#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para análise de clustering (Questão 1).
Identifica grupos incomuns de passageiros no aeroporto SFO.

Autor: [Seu Nome]
Data: [Data]
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from src.utils.data_loader import preprocess_data

def find_optimal_clusters(df_scaled, max_clusters=10):
    """
    Determina o número ótimo de clusters usando o método do cotovelo.
    
    Args:
        df_scaled (numpy.ndarray): Dados normalizados.
        max_clusters (int, optional): Número máximo de clusters a considerar.
    
    Returns:
        list: Lista de valores de inércia para cada número de clusters.
    """
    inertia = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df_scaled)
        inertia.append(km.inertia_)
    
    # Plotar o gráfico do cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, marker='o')
    plt.title('Método do Cotovelo para Determinar Número Ótimo de Clusters')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.grid(True)
    plt.savefig('resultados/questao1_elbow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return inertia

def analyze_clusters(df_preprocessed, clusters):
    """
    Analisa os clusters gerados, identificando o cluster incomum.
    
    Args:
        df_preprocessed (pandas.DataFrame): Dados pré-processados.
        clusters (numpy.ndarray): Array com os rótulos dos clusters.
    
    Returns:
        int: Índice do cluster incomum.
        pandas.DataFrame: Estatísticas descritivas do cluster incomum.
    """
    df_preprocessed['Cluster'] = clusters
    
    # Calcular o tamanho dos clusters (%)
    cluster_counts = df_preprocessed['Cluster'].value_counts(normalize=True) * 100
    print("\nTamanho dos clusters (%):")
    for cluster, percentage in cluster_counts.items():
        print(f"Cluster {cluster}: {percentage:.2f}%")
    
    # Identificar o cluster incomum (o menor)
    menor_cluster = cluster_counts.idxmin()
    print(f"\nCluster incomum identificado: {menor_cluster}")
    
    # Calcular perfil do cluster incomum
    perfil = df_preprocessed[df_preprocessed['Cluster'] == menor_cluster].describe()
    
    # Comparar com o perfil geral
    perfil_geral = df_preprocessed.describe()
    
    print("\nPerfil do cluster incomum vs Perfil Geral:")
    comparison = pd.DataFrame({
        'Cluster Incomum': perfil.loc['mean'],
        'Todos os Dados': perfil_geral.loc['mean']
    })
    print(comparison)
    
    return menor_cluster, perfil

def visualize_clusters(df_preprocessed, df_scaled):
    """
    Visualiza os clusters usando PCA para redução de dimensionalidade.
    
    Args:
        df_preprocessed (pandas.DataFrame): Dados pré-processados com rótulos de cluster.
        df_scaled (numpy.ndarray): Dados normalizados.
    """
    # Reduzir dimensionalidade para visualização
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled)
    
    # Adicionar componentes principais ao dataframe
    df_viz = df_preprocessed.copy()
    df_viz['PCA1'] = reduced_data[:, 0]
    df_viz['PCA2'] = reduced_data[:, 1]
    
    # Criar visualização dos clusters
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df_viz, 
        x='PCA1', 
        y='PCA2', 
        hue='Cluster', 
        palette='tab10',
        s=100,
        alpha=0.7
    )
    
    # Adicionar centroides
    centroids = []
    for cluster in df_viz['Cluster'].unique():
        centroid = df_viz[df_viz['Cluster'] == cluster][['PCA1', 'PCA2']].mean()
        centroids.append(centroid)
        plt.scatter(centroid['PCA1'], centroid['PCA2'], s=200, c='black', marker='X')
    
    plt.title('Visualização dos Clusters com PCA', fontsize=16)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('resultados/questao1_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_clustering_analysis(df, n_clusters=4):
    """
    Realiza a análise de clustering para identificar grupos incomuns de passageiros.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados originais.
        n_clusters (int, optional): Número de clusters a serem criados.
    """
    print("Iniciando análise de clustering...")
    
    # Selecionar e pré-processar os dados
    cols = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_preprocessed = preprocess_data(df, cols, encode_cols=['Q21Gender', 'Q22Income'])
    
    # Normalizar os dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_preprocessed)
    
    # Determinar número ótimo de clusters (opcional)
    # inertia = find_optimal_clusters(df_scaled)
    
    # Aplicar K-means com número de clusters definido
    print(f"Aplicando K-means com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Analisar os clusters
    menor_cluster, perfil = analyze_clusters(df_preprocessed, clusters)
    
    # Visualizar os clusters
    visualize_clusters(df_preprocessed, df_scaled)
    
    print("\nAnálise de clustering concluída!")
    print(f"O grupo incomum de passageiros foi identificado no Cluster {menor_cluster}.")
    
    return menor_cluster, perfil