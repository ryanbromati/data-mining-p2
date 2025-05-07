#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para análise de regressão logística (Questão 3).
Prediz se um passageiro está viajando pelo aeroporto SFO pela primeira vez.

Autor: [Seu Nome]
Data: [Data]
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils.data_loader import preprocess_data
import os

def visualize_data_distribution(df_preprocessed):
    """
    Visualiza a distribuição dos dados para entender o problema.
    
    Args:
        df_preprocessed (pandas.DataFrame): DataFrame pré-processado.
    """
    # Criar diretório para resultados se não existir
    os.makedirs('resultados', exist_ok=True)
    
    # Distribuição da variável alvo
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Q5FIRSTTIME', data=df_preprocessed)
    plt.title('Distribuição de Passageiros de Primeira Viagem', fontsize=14)
    plt.xlabel('Primeira Viagem (1 = Sim, 0 = Não)', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    
    # Adicionar rótulos nas barras
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                f'{height} ({height/len(df_preprocessed)*100:.1f}%)',
                ha="center", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('resultados/questao3_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Relação entre variáveis preditoras e a variável alvo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Idade vs. Primeira Viagem
    sns.boxplot(x='Q5FIRSTTIME', y='Q20Age', data=df_preprocessed, ax=axes[0, 0])
    axes[0, 0].set_title('Idade vs. Primeira Viagem', fontsize=12)
    axes[0, 0].set_xlabel('Primeira Viagem', fontsize=10)
    axes[0, 0].set_ylabel('Idade', fontsize=10)
    
    # Gênero vs. Primeira Viagem
    sns.countplot(x='Q21Gender', hue='Q5FIRSTTIME', data=df_preprocessed, ax=axes[0, 1])
    axes[0, 1].set_title('Gênero vs. Primeira Viagem', fontsize=12)
    axes[0, 1].set_xlabel('Gênero', fontsize=10)
    axes[0, 1].set_ylabel('Contagem', fontsize=10)
    
    # Renda vs. Primeira Viagem
    sns.boxplot(x='Q5FIRSTTIME', y='Q22Income', data=df_preprocessed, ax=axes[1, 0])
    axes[1, 0].set_title('Renda vs. Primeira Viagem', fontsize=12)
    axes[1, 0].set_xlabel('Primeira Viagem', fontsize=10)
    axes[1, 0].set_ylabel('Renda', fontsize=10)
    
    # Frequência de Voos vs. Primeira Viagem
    sns.boxplot(x='Q5FIRSTTIME', y='Q5TIMESFLOWN', data=df_preprocessed, ax=axes[1, 1])
    axes[1, 1].set_title('Frequência de Voos vs. Primeira Viagem', fontsize=12)
    axes[1, 1].set_xlabel('Primeira Viagem', fontsize=10)
    axes[1, 1].set_ylabel('Frequência de Voos', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('resultados/questao3_feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_logistic_model(X_train, X_test, y_train, y_test):
    """
    Treina um modelo de regressão logística e avalia seu desempenho.
    
    Args:
        X_train (pandas.DataFrame): Features de treinamento.
        X_test (pandas.DataFrame): Features de teste.
        y_train (pandas.Series): Rótulos de treinamento.
        y_test (pandas.Series): Rótulos de teste.
    
    Returns:
        tuple: Modelo treinado, predições, probabilidades e acurácia.
    """
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar o modelo
    print("Treinando modelo de regressão logística...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Fazer predições
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.4f}")
    
    return model, y_pred, y_prob, accuracy

def evaluate_model(model, X_train, X_test, y_test, y_pred, y_prob):
    """
    Avalia o modelo de regressão logística com diferentes métricas.
    
    Args:
        model (LogisticRegression): Modelo treinado.
        X_train (pandas.DataFrame): Features de treinamento.
        X_test (pandas.DataFrame): Features de teste.
        y_test (pandas.Series): Rótulos de teste.
        y_pred (numpy.ndarray): Predições do modelo.
        y_prob (numpy.ndarray): Probabilidades das predições.
    """
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão', fontsize=14)
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.savefig('resultados/questao3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curva ROC', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('resultados/questao3_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Importância das features
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Importância das Features', fontsize=14)
    plt.xlabel('Importância Absoluta', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('resultados/questao3_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def perform_logistic_regression(df):
    """
    Realiza a análise de regressão logística para prever se um passageiro
    está viajando pelo aeroporto SFO pela primeira vez.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados originais.
    
    Returns:
        tuple: Modelo treinado e métricas de desempenho.
    """
    print("Iniciando análise de regressão logística...")
    
    # Problema sendo resolvido
    print("\nProblema: Previsão de Passageiros de Primeira Viagem no Aeroporto SFO")
    print("Objetivo: Prever se um passageiro está viajando pelo aeroporto pela primeira vez")
    print("         com base em características demográficas e comportamentais.")
    
    # Dados utilizados
    print("\nDados utilizados:")
    print("- Dataset: aeroporto SFO (San Francisco International Airport)")
    print("- Variável alvo: Q5FIRSTTIME (primeira vez no aeroporto: 1=Sim, 0=Não)")
    print("- Features: gênero, idade, renda, tipo de passageiro (Q23FLY), frequência de voos")
    
    # Por que regressão logística é apropriada
    print("\nPor que a Regressão Logística é apropriada:")
    print("1. A variável alvo é binária (primeira vez ou não)")
    print("2. Queremos prever a probabilidade de um passageiro ser de primeira viagem")
    print("3. Precisamos entender quais fatores influenciam essa probabilidade")
    print("4. A regressão logística permite interpretação dos coeficientes")
    print("5. Diferente da regressão linear, a logística é específica para variáveis dependentes categóricas")
    
    # Selecionar e pré-processar os dados
    cols = ['Q5FIRSTTIME', 'Q21Gender', 'Q22Income', 'Q20Age', 'Q23FLY', 'Q5TIMESFLOWN']
    df_preprocessed = preprocess_data(df, cols, encode_cols=['Q21Gender', 'Q23FLY'])
    
    # Visualizar distribuição dos dados
    visualize_data_distribution(df_preprocessed)
    
    # Preparar dados para modelagem
    X = df_preprocessed.drop('Q5FIRSTTIME', axis=1)
    y = df_preprocessed['Q5FIRSTTIME']
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar o modelo de regressão logística
    model, y_pred, y_prob, accuracy = train_logistic_model(X_train, X_test, y_train, y_test)
    
    # Avaliar o modelo
    evaluate_model(model, X_train, X_test, y_test, y_pred, y_prob)

    return model, accuracy

if __name__ == "__main__":
    # Carregar dados
    df = pd.read_csv('data/sfo_passengers.csv')

    # Realizar regressão logística
    model, accuracy = perform_logistic_regression(df)
    print(f"Acurácia final do modelo: {accuracy:.4f}")
