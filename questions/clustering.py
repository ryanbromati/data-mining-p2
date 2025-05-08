import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df, columns_to_use):
    """
    Pré-processa os dados, tratando valores ausentes e preparando para clustering
    """
    # Selecionar apenas as colunas relevantes
    df_subset = df[columns_to_use].copy()
    
    # Converter colunas para tipos numéricos onde possível
    for col in df_subset.columns:
        try:
            df_subset[col] = pd.to_numeric(df_subset[col])
        except:
            pass  # Se não puder converter, manter como está
    
    # Tratar valores ausentes
    # Substituir valores não numéricos por NaN
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    
    # Remover linhas com valores ausentes
    df_clean = df_subset.dropna()
    
    return df_clean

def encode_categorical(df):
    """
    Codifica variáveis categóricas
    """
    label_encoders = {}
    df_encoded = df.copy()
    
    # Codificar colunas categóricas
    for col in ['Q21Gender', 'Q22Income', 'Q23FLY']:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders

def find_optimal_clusters(data, max_k):
    """
    Determina o número ideal de clusters usando método do cotovelo e silhouette score
    """
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        # Calcular o silhouette score
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
        
    # Plotar método do cotovelo
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertias, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Número de clusters')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    plt.close()
    
    # Encontrar o melhor k pelo silhouette score
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return best_k

def perform_clustering(df, n_clusters):
    """
    Realiza o clustering e retorna os dados com os labels
    """
    # Normalizar os dados
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Executar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    
    # Adicionar labels ao dataframe original
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Calcular tamanho de cada cluster
    cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
    cluster_percentages = 100 * cluster_sizes / len(df_with_clusters)
    
    # Exibir informações sobre os clusters
    print("\nTamanho dos clusters:")
    for i, (size, percentage) in enumerate(zip(cluster_sizes, cluster_percentages)):
        print(f"Cluster {i}: {size} passageiros ({percentage:.2f}%)")
    
    return df_with_clusters, kmeans, df_scaled

def visualize_clusters(df_with_clusters, df_scaled, kmeans):
    """
    Visualiza os clusters usando PCA para redução de dimensionalidade
    """
    # Reduzir dimensionalidade para visualização
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    
    # Criar dataframe com componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df_with_clusters['cluster']
    
    # Visualizar clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=50)
    
    # Plotar centroides
    centroids = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroides')
    
    plt.title('Visualização dos Clusters (PCA)')
    plt.legend()
    plt.savefig('cluster_visualization.png')
    plt.close()
    
    # Retornar informação sobre a variância explicada pelo PCA
    var_explained = pca.explained_variance_ratio_
    return var_explained

def analyze_clusters(df_with_clusters):
    """
    Analisa o perfil de cada cluster
    """
    # Calcular média de cada característica por cluster
    cluster_profiles = df_with_clusters.groupby('cluster').mean()
    
    # Visualizar perfis dos clusters
    plt.figure(figsize=(14, 10))
    sns.heatmap(cluster_profiles, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Perfis dos Clusters (Médias)')
    plt.savefig('cluster_profiles.png')
    plt.close()
    
    # Análise detalhada por cluster
    print("\nPerfis dos Clusters:")
    for cluster in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
        print(f"\nCluster {cluster} (n={len(cluster_data)}, {100 * len(cluster_data) / len(df_with_clusters):.2f}%):")
        
        for col in df_with_clusters.columns:
            if col != 'cluster':
                cluster_mean = cluster_data[col].mean()
                overall_mean = df_with_clusters[col].mean()
                diff = cluster_mean - overall_mean
                print(f"  - {col}: {cluster_mean:.2f} (diferença da média geral: {diff:+.2f})")
    
    # Criar visualizações de distribuição para cada variável
    cols = df_with_clusters.columns.drop('cluster')
    plt.figure(figsize=(15, 5 * len(cols)))
    
    for i, col in enumerate(cols):
        plt.subplot(len(cols), 1, i+1)
        for cluster in sorted(df_with_clusters['cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
            sns.kdeplot(cluster_data[col], label=f'Cluster {cluster}')
        
        plt.title(f'Distribuição de {col} por Cluster')
        plt.xlabel(col)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('cluster_distributions.png')
    plt.close()
    
    return cluster_profiles

def identify_anomalous_cluster(df_with_clusters, cluster_profiles):
    """
    Identifica o cluster que mais se desvia da média geral
    """
    # Calcular médias gerais
    overall_means = df_with_clusters.drop('cluster', axis=1).mean()
    
    # Calcular desvio de cada cluster em relação à média geral
    deviations = {}
    for cluster in cluster_profiles.index:
        cluster_profile = cluster_profiles.loc[cluster]
        deviation = sum((cluster_profile - overall_means) ** 2)
        deviations[cluster] = deviation
    
    # Identificar o cluster mais anômalo
    most_anomalous = max(deviations, key=deviations.get)
    
    print(f"\nCluster mais anômalo: Cluster {most_anomalous}")
    print(f"Percentual do total de passageiros: {100 * len(df_with_clusters[df_with_clusters['cluster'] == most_anomalous]) / len(df_with_clusters):.2f}%")
    
    # Descrever o perfil do cluster anômalo
    anomalous_profile = cluster_profiles.loc[most_anomalous]
    print("\nPerfil do cluster anômalo:")
    for col in anomalous_profile.index:
        diff = anomalous_profile[col] - overall_means[col]
        print(f"  - {col}: {anomalous_profile[col]:.2f} (diferença da média: {diff:+.2f})")
    
    return most_anomalous

def main():
    # Carregar o dataset
    file_path = "data/dataset.csv"  # Substitua pelo caminho correto do arquivo
    df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip')
    
    # Variáveis relevantes conforme sugerido no enunciado
    columns = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 
               'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    
    # Pré-processar os dados
    df_preprocessed = preprocess_data(df, columns)
    print(f"Dados pré-processados: {df_preprocessed.shape[0]} registros com {df_preprocessed.shape[1]} características")
    
    # Codificar variáveis categóricas
    df_encoded, encoders = encode_categorical(df_preprocessed)
    
    # Encontrar número ideal de clusters
    max_k = 10  # Testar até 10 clusters
    best_k = find_optimal_clusters(df_encoded, max_k)
    print(f"\nNúmero ideal de clusters: {best_k}")
    
    # Realizar clustering
    df_with_clusters, kmeans, df_scaled = perform_clustering(df_encoded, best_k)
    
    # Visualizar clusters
    var_explained = visualize_clusters(df_with_clusters, df_scaled, kmeans)
    print(f"\nVariância explicada pelas componentes principais: {var_explained[0]:.2f}, {var_explained[1]:.2f}")
    print(f"Total de variância explicada: {sum(var_explained):.2f}")
    
    # Analisar perfis dos clusters
    cluster_profiles = analyze_clusters(df_with_clusters)
    
    # Identificar cluster anômalo
    anomalous_cluster = identify_anomalous_cluster(df_with_clusters, cluster_profiles)
    
    print("\nAnálise concluída. Visualizações salvas em arquivos PNG.")

if __name__ == "__main__":
    main()