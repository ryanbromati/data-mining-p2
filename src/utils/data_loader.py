#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo responsável por carregar e pré-processar os dados.

Autor: [Seu Nome]
Data: [Data]
"""

import pandas as pd
import os

def load_data(file_path, sep=';', encoding='latin1', on_bad_lines='skip'):
    """
    Carrega os dados de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV.
        sep (str, optional): Separador de campos. Padrão: ';'.
        encoding (str, optional): Codificação do arquivo. Padrão: 'latin1'.
        on_bad_lines (str, optional): Comportamento para linhas inválidas. Padrão: 'skip'.
    
    Returns:
        pandas.DataFrame: DataFrame contendo os dados carregados.
    
    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
        Exception: Para outros erros de leitura.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")
    
    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            on_bad_lines=on_bad_lines
        )
        return df
    except Exception as e:
        raise Exception(f"Erro ao carregar o arquivo {file_path}: {e}")

def preprocess_data(df, cols, encode_cols=None):
    """
    Pré-processa os dados selecionando colunas e codificando variáveis categóricas.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados originais.
        cols (list): Lista de colunas para seleção.
        encode_cols (list, optional): Lista de colunas categóricas para codificação.
    
    Returns:
        pandas.DataFrame: DataFrame pré-processado.
    """
    df_selected = df[cols].copy().dropna()
    
    if encode_cols:
        for col in encode_cols:
            df_selected[col] = df_selected[col].astype('category').cat.codes
    
    return df_selected