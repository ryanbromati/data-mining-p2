#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para execução da avaliação de Mineração de Dados.
Este script carrega os dados e executa as análises para as três questões.

Autor: [Seu Nome]
Data: [Data]
"""

import os
import argparse
from src.utils.data_loader import load_data
from src.question1.clustering import perform_clustering_analysis
from src.question2.association import perform_association_analysis
from src.question3.logistic import perform_logistic_regression

def main(file_path, questions=None):
    """
    Função principal que executa as análises para as questões especificadas.
    
    Args:
        file_path (str): Caminho para o arquivo de dados.
        questions (list, optional): Lista de questões a serem executadas. 
                                   Se None, executa todas as questões.
    """
    # Criar diretório para resultados
    os.makedirs('resultados', exist_ok=True)
    
    print(f"Carregando dados de: {file_path}")
    try:
        df = load_data(file_path)
        print(f"Dados carregados com sucesso. Formato: {df.shape}")
        
        if questions is None or 1 in questions:
            print("\n" + "="*50)
            print("Questão 1: Análise de Clustering")
            print("="*50)
            perform_clustering_analysis(df)
        
        if questions is None or 2 in questions:
            print("\n" + "="*50)
            print("Questão 2: Regras de Associação")
            print("="*50)
            perform_association_analysis(df)
        
        if questions is None or 3 in questions:
            print("\n" + "="*50)
            print("Questão 3: Regressão Logística")
            print("="*50)
            perform_logistic_regression(df)
            
    except Exception as e:
        print(f"Erro ao executar análises: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise de Mineração de Dados")
    parser.add_argument("--file", "-f", default="./data/dataset.csv", 
                        help="Caminho para o arquivo de dados (default: ./data/dataset.csv)")
    parser.add_argument("--questions", "-q", nargs="+", type=int, 
                        help="Questões a serem executadas (1, 2, 3). Se não especificado, executa todas.")
    
    args = parser.parse_args()
    main(args.file, args.questions)