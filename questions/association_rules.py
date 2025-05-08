import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Carregar o dataset Pima Indians Diabetes
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

print("Dimensões do dataset:", df.shape)
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Verificar valores ausentes
print("\nVerificando valores ausentes:")
print(df.isnull().sum())

# Alguns zeros podem representar valores ausentes em certos campos
print("\nContagem de zeros que poderiam ser valores ausentes:")
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = len(df[df[column] == 0])
    print(f"{column}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")

# Tratar valores zero como ausentes em colunas médicas relevantes
def replace_zeros(df):
    # Cópia para não modificar o original
    df_clean = df.copy()
    
    # Substituir zeros por NaN
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df_clean[column] = df_clean[column].replace(0, np.nan)
    
    # Substituir NaN pela mediana de cada coluna
    for column in df_clean.columns[:-1]:  # Excluir a coluna 'Outcome'
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    
    return df_clean

# Aplicar tratamento de dados
df_clean = replace_zeros(df)

# Verificar distribuição da variável alvo
print("\nDistribuição da variável alvo (Outcome):")
print(df_clean['Outcome'].value_counts())
print(f"Percentual de pacientes diabéticos: {df_clean['Outcome'].mean()*100:.1f}%")

# Visualizar correlação entre variáveis
plt.figure(figsize=(12, 10))
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.savefig('correlation_matrix.png')
plt.close()

# Visualizar distribuição das características por classe
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_clean.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.histplot(data=df_clean, x=column, hue='Outcome', kde=True, bins=30)
    plt.title(f'Distribuição de {column} por Classe')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Preparar dados para modelagem
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nDimensões dos dados de treino e teste:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Criar um pipeline com padronização e regressão logística
pipeline_logistic = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000, random_state=42))
])

# Criar um pipeline com padronização e regressão linear (para comparação)
pipeline_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# Treinar ambos os modelos
pipeline_logistic.fit(X_train, y_train)
pipeline_linear.fit(X_train, y_train)

# Avaliar modelo logístico
y_pred_logistic = pipeline_logistic.predict(X_test)
y_prob_logistic = pipeline_logistic.predict_proba(X_test)[:, 1]

print("\n=== Resultados da Regressão Logística ===")
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Acurácia: {accuracy_logistic:.4f}")

print("\nMatriz de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred_logistic)
print(conf_matrix)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_logistic))

# Avaliar modelo linear (para comparação)
y_pred_linear_raw = pipeline_linear.predict(X_test)
# Arredondar previsões para 0 ou 1
y_pred_linear = np.round(np.clip(y_pred_linear_raw, 0, 1)).astype(int)

print("\n=== Resultados da Regressão Linear (para comparação) ===")
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Acurácia: {accuracy_linear:.4f}")

print("\nMatriz de Confusão:")
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print(conf_matrix_linear)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_linear))

# Visualizar curva ROC para regressão logística
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob_logistic)
auc = roc_auc_score(y_test, y_prob_logistic)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Regressão Logística')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Visualizar importância das características (coeficientes)
coef = pipeline_logistic.named_steps['logistic'].coef_[0]
features = X.columns
sorted_idx = np.argsort(abs(coef))
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), coef[sorted_idx])
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.xlabel('Coeficiente')
plt.title('Importância das Características - Regressão Logística')
plt.savefig('feature_importance.png')
plt.close()

# Validação cruzada para modelo logístico
cv_scores = cross_val_score(pipeline_logistic, X, y, cv=5, scoring='accuracy')
print("\nResultados da Validação Cruzada (5-fold):")
print(f"Acurácia média: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")

# Por que a regressão logística é mais apropriada que a linear para esse problema?
print("\n=== Comparação entre Regressão Logística e Linear ===")
print("1. A regressão logística é mais apropriada para variáveis de resposta binárias")
print("2. A regressão logística fornece probabilidades entre 0 e 1")
print("3. A regressão linear pode produzir valores fora do intervalo [0,1]")

# Demonstrar o problema da regressão linear para classificação
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Valores Reais', alpha=0.7)
plt.scatter(range(len(y_test)), y_pred_linear_raw, label='Previsões Lineares', alpha=0.7)
plt.scatter(range(len(y_test)), y_prob_logistic, label='Probabilidades Logísticas', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Valor Previsto')
plt.xlabel('Amostra')
plt.title('Comparação: Regressão Linear vs. Logística')
plt.legend()
plt.savefig('linear_vs_logistic.png')
plt.close()

# Resumo final
print("\n=== Resumo Final ===")
print(f"Acurácia da Regressão Logística: {accuracy_logistic:.4f}")
print(f"Acurácia da Regressão Linear: {accuracy_linear:.4f}")
print(f"Ganho de desempenho: {(accuracy_logistic - accuracy_linear) * 100:.2f}%")

# Exemplo de interpretação de um paciente
print("\n=== Interpretação do Modelo ===")
# Pegar um exemplo do conjunto de teste
exemplo = X_test.iloc[0:1]
probabilidade = pipeline_logistic.predict_proba(exemplo)[0, 1]
print(f"Dados do paciente exemplo:")
for feature, value in zip(X.columns, exemplo.values[0]):
    print(f"  - {feature}: {value:.2f}")

print(f"\nProbabilidade de diabetes: {probabilidade:.2f}")
if probabilidade >= 0.5:
    print("Previsão: Diabético")
else:
    print("Previsão: Não diabético")