#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para análise de regras de associação (Questão 2).
Identifica padrões de comportamento dos passageiros do aeroporto SFO.

Autor: [Seu Nome]
Data: [Data]
"""

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import numpy as np
from src.utils.data_loader import preprocess_data
import os

def binarize_data(df_preprocessed):
    """
    Converte os dados em formato binário para análise de regras de associação.
    
    Args:
        df_preprocessed (pandas.DataFrame): DataFrame pré-processado.
    
    Returns:
        pandas.DataFrame: DataFrame binarizado.
    """
    print("Binarizando os dados...")
    df_bin = df_preprocessed.copy()
    
    # Binarizar variáveis numéricas
    df_bin['NETPRO_HIGH'] = df_bin['NETPRO  '].apply(lambda x: 1 if x >= 9 else 0)
    df_bin['AGE_OVER_40'] = df_bin['Q20Age'].apply(lambda x: 1 if x > 40 else 0)
    df_bin['FREQ_FLYER'] = df_bin['Q5TIMESFLOWN'].apply(lambda x: 1 if x > 3 else 0)
    df_bin['LONG_TERM_USER'] = df_bin['Q6LONGUSE'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Criar colunas dummy para variáveis categóricas
    gender_dummies = pd.get_dummies(df_bin['Q21Gender'], prefix='GENDER')
    income_dummies = pd.get_dummies(df_bin['Q22Income'], prefix='INCOME')
    fly_dummies = pd.get_dummies(df_bin['Q23FLY'], prefix='FLY_TYPE')
    
    # Combinar todas as variáveis binárias
    df_final = pd.concat([
        df_bin[['NETPRO_HIGH', 'AGE_OVER_40', 'FREQ_FLYER', 'LONG_TERM_USER']],
        gender_dummies, income_dummies, fly_dummies
    ], axis=1)
    
    return df_final

def generate_association_rules(df_bin, min_support=0.1, lift_threshold=1.0):
    """
    Gera regras de associação usando o algoritmo FP-Growth.
    
    Args:
        df_bin (pandas.DataFrame): DataFrame binarizado.
        min_support (float, optional): Suporte mínimo para as regras.
        lift_threshold (float, optional): Limite mínimo de lift.
    
    Returns:
        pandas.DataFrame: DataFrame com as regras de associação.
    """
    print(f"Gerando conjunto de itens frequentes (min_support={min_support})...")
    frequent_itemsets = fpgrowth(df_bin, min_support=min_support, use_colnames=True)
    print(f"Encontrados {len(frequent_itemsets)} conjuntos de itens frequentes.")
    
    print(f"Gerando regras de associação (lift_threshold={lift_threshold})...")
    rules = association_rules(
        frequent_itemsets, 
        metric="lift", 
        min_threshold=lift_threshold
    )
    
    # Ordenar regras por lift (decrescente)
    rules = rules.sort_values('lift', ascending=False)
    
    return rules

def format_rules(rules):
    """
    Formata as regras de associação para melhor legibilidade.
    
    Args:
        rules (pandas.DataFrame): DataFrame com as regras de associação.
    
    Returns:
        pandas.DataFrame: DataFrame com as regras formatadas.
    """
    # Converter frozensets para strings mais legíveis
    def format_itemset(itemset):
        return ', '.join(sorted(list(itemset)))
    
    formatted_rules = rules.copy()
    formatted_rules['antecedents_str'] = formatted_rules['antecedents'].apply(format_itemset)
    formatted_rules['consequents_str'] = formatted_rules['consequents'].apply(format_itemset)
    
    return formatted_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]

def visualize_rules(rules):
    """
    Visualiza as regras de associação através de gráficos.
    
    Args:
        rules (pandas.DataFrame): DataFrame com as regras de associação.
    """
    if len(rules) == 0:
        print("Não há regras para visualizar.")
        return
    
    # Criar diretório para resultados se não existir
    os.makedirs('resultados', exist_ok=True)
    
    # Gráfico de dispersão: Suporte vs Confiança, colorido por lift
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=rules, 
        x="support", 
        y="confidence", 
        hue="lift", 
        size="lift", 
        sizes=(20, 200),
        palette="viridis"
    )
    
    plt.title("Regras de Associação: Suporte vs Confiança", fontsize=16)
    plt.xlabel("Suporte", fontsize=12)
    plt.ylabel("Confiança", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(scatter.get_children()[0], label="Lift")
    plt.savefig('resultados/questao2_rules_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico de barras das 10 melhores regras por lift
    top_rules = rules.head(10).copy()
    top_rules['rule'] = top_rules.apply(
        lambda row: f"{', '.join(sorted(list(row['antecedents'])))} → {', '.join(sorted(list(row['consequents'])))}", 
        axis=1
    )
    
    plt.figure(figsize=(12, 8))
    bars = sns.barplot(
        data=top_rules,
        y='rule',
        x='lift',
        palette='viridis'
    )
    
    plt.title("Top 10 Regras de Associação por Lift", fontsize=16)
    plt.xlabel("Lift", fontsize=12)
    plt.ylabel("Regra", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('resultados/questao2_top_rules.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_association_analysis(df, min_support=0.1, lift_threshold=1.0):
    """
    Realiza a análise de regras de associação para identificar padrões de comportamento.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados originais.
        min_support (float, optional): Suporte mínimo para as regras.
        lift_threshold (float, optional): Limite mínimo de lift.
    
    Returns:
        pandas.DataFrame: DataFrame com as regras de associação.
    """
    print("Iniciando análise de regras de associação...")
    
    # Problema sendo resolvido
    print("\nProblema: Identificação de padrões de comportamento dos passageiros do aeroporto SFO")
    print("Objetivo: Descobrir quais características dos passageiros estão associadas à alta satisfação")
    
    # Dados utilizados
    print("\nDados utilizados:")
    print("- Dataset: aeroporto SFO (San Francisco International Airport)")
    print("- Variáveis: satisfação (NETPRO), idade, gênero, renda, frequência de voos, experiência")
    
    # Selecionar e pré-processar os dados
    cols = ['NETPRO  ', 'Q20Age', 'Q21Gender', 'Q22Income', 'Q23FLY', 'Q5TIMESFLOWN', 'Q6LONGUSE']
    df_preprocessed = preprocess_data(df, cols)
    
    # Passos para geração de regras
    print("\nPassos para geração de regras:")
    print("1. Pré-processamento dos dados")
    print("2. Codificação de variáveis categóricas")
    print("3. Binarização dos dados")
    print("4. Aplicação do algoritmo FP-Growth para encontrar conjuntos frequentes")
    print("5. Geração de regras de associação")
    
    # Codificar variáveis categóricas
    label_encoders = {}
    for col in ['Q21Gender', 'Q22Income', 'Q23FLY']:
        le = LabelEncoder()
        df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
        label_encoders[col] = le
    
    # Binarizar os dados
    df_bin = binarize_data(df_preprocessed)
    
    # Gerar regras de associação
    rules = generate_association_rules(df_bin, min_support, lift_threshold)
    
    # Formatar e exibir as regras
    if len(rules) > 0:
        formatted_rules = format_rules(rules)
        print("\nRegras de Associação Geradas:")
        print(formatted_rules.head(10))
        
        # Visualizar as regras
        visualize_rules(rules)
    else:
        print("\nNenhuma regra de associação foi encontrada com os parâmetros especificados.")
        print("Tente reduzir o valor de min_support ou lift_threshold.")
    
    print("\nAnálise de regras de associação concluída!")
    
    return rules