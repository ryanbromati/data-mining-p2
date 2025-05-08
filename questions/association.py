import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
import random
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_sessions=50000, n_categories=15, max_items_per_session=8):
    """
    Gera dados sintéticos para sessões de navegação de usuários
    """
    # Categorias de conteúdo do portal de notícias
    categories = [
        'Política', 'Economia', 'Esportes', 'Entretenimento', 'Tecnologia',
        'Ciência', 'Saúde', 'Educação', 'Negócios', 'Internacional',
        'Cultura', 'Lifestyle', 'Opinião', 'Viagem', 'Gastronomia'
    ]
    
    # Definir alguns padrões de comportamento para tornar os dados mais realistas
    common_patterns = [
        ['Política', 'Economia', 'Negócios'],
        ['Esportes', 'Entretenimento'],
        ['Tecnologia', 'Ciência', 'Educação'],
        ['Saúde', 'Lifestyle'],
        ['Internacional', 'Política'],
        ['Cultura', 'Entretenimento'],
        ['Viagem', 'Gastronomia', 'Lifestyle'],
        ['Tecnologia', 'Negócios']
    ]
    
    # Criar lista para armazenar as sessões
    data = []
    
    # Gerar sessões
    for i in range(n_sessions):
        # Determinar número de itens na sessão
        n_items = random.randint(1, max_items_per_session)
        
        # Decidir se vamos usar um padrão comum (70% de chance)
        if random.random() < 0.7 and n_items >= 2:
            # Escolher um padrão aleatório
            pattern = random.choice(common_patterns)
            
            # Limitar ao número de itens
            pattern = pattern[:n_items]
            
            # Se o padrão for menor que n_items, adicionar itens aleatórios
            while len(pattern) < n_items:
                new_category = random.choice(categories)
                if new_category not in pattern:
                    pattern.append(new_category)
                    
            session = pattern
        else:
            # Criar sessão totalmente aleatória
            session = []
            while len(session) < n_items:
                category = random.choice(categories)
                if category not in session:
                    session.append(category)
        
        # Adicionar ID de sessão e timestamp
        session_id = f"s{i+1:06d}"
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
        
        # Adicionar cada visualização à lista de dados
        for cat in session:
            data.append({
                'session_id': session_id,
                'category': cat,
                'timestamp': timestamp
            })
            # Avançar tempo para próxima visualização
            timestamp += timedelta(minutes=random.randint(1, 15))
    
    # Converter para DataFrame
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Pré-processa os dados para uso com algoritmos de regras de associação
    """
    # Contar número de categorias visitadas por sessão
    session_counts = df['session_id'].value_counts()
    
    # Filtrar sessões com no mínimo 2 categorias visitadas
    valid_sessions = session_counts[session_counts >= 2].index
    df_filtered = df[df['session_id'].isin(valid_sessions)]
    
    # Transformar para formato transacional (sessão x categoria)
    df_pivot = pd.crosstab(df_filtered['session_id'], df_filtered['category'])
    
    # Converter para formato binário (1 = visitou, 0 = não visitou)
    df_binary = df_pivot.applymap(lambda x: 1 if x > 0 else 0)
    
    return df_binary

def generate_association_rules(df_binary, min_support=0.01, min_threshold=1.5):
    """
    Gera regras de associação usando o algoritmo FP-Growth
    """
    # Encontrar conjuntos frequentes
    frequent_itemsets = fpgrowth(df_binary, min_support=min_support, use_colnames=True)
    
    # Gerar regras de associação
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    # Ordenar regras por lift (indicador de força da associação)
    rules = rules.sort_values('lift', ascending=False)
    
    return rules

def visualize_rules(rules, top_n=10):
    """
    Visualiza as regras de associação mais importantes
    """
    # Filtrar para mostrar apenas as top N regras
    top_rules = rules.head(top_n)
    
    # Converter frozensets para strings para facilitar visualização
    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Plot de dispersão de suporte vs confiança com lift como tamanho
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="support",
        y="confidence", 
        size="lift",
        hue="lift",
        data=rules,
        sizes=(50, 500),
        palette="viridis"
    )
    plt.title('Regras de Associação: Suporte vs Confiança')
    plt.xlabel('Suporte')
    plt.ylabel('Confiança')
    plt.grid(True, alpha=0.3)
    plt.savefig('rules_visualization.png')
    plt.close()
    
    # Heatmap para visualizar a força das associações entre categorias
    categories = list(set(
        [item for sublist in rules['antecedents'].tolist() + rules['consequents'].tolist() 
         for item in sublist]
    ))
    
    association_matrix = pd.DataFrame(0, index=categories, columns=categories)
    
    for _, rule in rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                association_matrix.loc[ant, cons] = max(association_matrix.loc[ant, cons], rule['lift'])
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(association_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Força das Associações entre Categorias de Conteúdo (Lift)')
    plt.savefig('q2/category_associations.png')
    plt.close()
    
    return top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

def main():
    # Gerar dados sintéticos (em um caso real, carregaríamos de um arquivo)
    print("Gerando dados sintéticos...")
    df = generate_synthetic_data()
    
    print(f"Dataset gerado com {len(df)} registros de navegação.")
    print(f"Número de sessões únicas: {df['session_id'].nunique()}")
    print(f"Categorias disponíveis: {', '.join(df['category'].unique())}")
    
    # Pré-processar os dados
    print("\nPré-processando dados...")
    df_binary = preprocess_data(df)
    print(f"Dados em formato transacional: {df_binary.shape[0]} sessões x {df_binary.shape[1]} categorias")
    
    # Gerar regras de associação
    print("\nGerando regras de associação com FP-Growth...")
    rules = generate_association_rules(df_binary, min_support=0.02, min_threshold=1.3)
    print(f"Total de regras geradas: {len(rules)}")
    
    # Visualizar e analisar regras
    print("\nAnalisando as regras mais importantes...")
    top_rules = visualize_rules(rules)
    
    print("\nTop 10 regras de associação por lift:")
    pd.set_option('display.max_colwidth', None)
    print(top_rules)
    
    print("\nAnálise concluída. Visualizações salvas em arquivos PNG.")
    
    # Salvar exemplos em arquivo CSV
    df.sample(1000).to_csv('sample_browsing_data.csv', index=False)
    rules.to_csv('association_rules.csv', index=False)
    
    print("\nAmostras dos dados e regras completas salvas em arquivos CSV.")

if __name__ == "__main__":
    main()