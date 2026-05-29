import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# ==========================================
# PARTE 1: PREPARAÇÃO E MÉTODO DO COTOVELO
# ==========================================

# 1. Leitura dos dados ilusórios
df_original = pd.read_csv('seu_arquivo_ilusorio.csv')

# Criando uma cópia para não alterar os dados originais durante a análise final
df = df_original.copy()

# 2. Identificação das variáveis numéricas e categóricas
# (Substitua as listas abaixo com os nomes reais das suas colunas)
colunas_numericas = ['idade', 'renda', 'score_credito'] 
colunas_categoricas = ['genero', 'estado_civil', 'profissao']

# 3. Normalização das variáveis numéricas
scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# O algoritmo K-Prototypes precisa saber a posição (índice) das colunas categóricas
posicoes_categoricas = [df.columns.get_loc(col) for col in colunas_categoricas]

# Convertendo o DataFrame para uma matriz NumPy (necessário para o kmodes)
matriz_dados = df.to_numpy()

# 4. Método do Cotovelo (Elbow Method) para testar 10 valores de K
custos = []
K_maximo = 10

print("Calculando o Método do Cotovelo (isso pode levar alguns minutos)...")
for k in range(1, K_maximo + 1):
    # init='Cao' é um método eficiente de inicialização para dados categóricos
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=3, random_state=42, n_jobs=-1)
    kproto.fit(matriz_dados, categorical=posicoes_categoricas)
    custos.append(kproto.cost_)

# Plotando o gráfico do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, K_maximo + 1), custos, marker='o', linestyle='--')
plt.title('Método do Cotovelo para K-Prototypes')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Custo (Cost)')
plt.grid(True)
plt.show()

# ==========================================
# PARTE 2: RETREINAMENTO E ANÁLISE
# ==========================================

# Após analisar o gráfico acima, você deve identificar o "cotovelo" 
# (o ponto onde a queda do custo deixa de ser acentuada). 
# Digite o valor de K ideal escolhido:
k_ideal = int(input("Analisando o gráfico, qual o número ideal de clusters (K)? "))

print(f"\nTreinando o modelo final com K={k_ideal}...")
kproto_final = KPrototypes(n_clusters=k_ideal, init='Cao', n_init=5, random_state=42, n_jobs=-1)
clusters_preditos = kproto_final.fit_predict(matriz_dados, categorical=posicoes_categoricas)

# Adicionando a coluna de clusters ao dataframe ORIGINAL (sem normalização) 
# para facilitar a interpretação do mundo real
df_original['Cluster'] = clusters_preditos

# 5. Análise dos Resultados
print("\n--- RESUMO DOS CLUSTERS ---")

# Contagem de registros por cluster
print("\nTamanho de cada Cluster:")
print(df_original['Cluster'].value_counts().sort_index())

# Análise das variáveis NUMÉRICAS (Média por cluster)
print("\nPerfil Numérico (Média):")
perfil_numerico = df_original.groupby('Cluster')[colunas_numericas].mean()
print(perfil_numerico)

# Análise das variáveis CATEGÓRICAS (Moda / Valor mais frequente por cluster)
print("\nPerfil Categórico (Moda):")
perfil_categorico = df_original.groupby('Cluster')[colunas_categoricas].agg(lambda x: x.mode().iloc[0])
print(perfil_categorico)

# Opcional: Analisando as proporções de uma variável categórica específica dentro de cada cluster
# Exemplo: Distribuição de 'genero' dentro de cada cluster
print("\nDistribuição de Gênero por Cluster:")
distribuicao_genero = pd.crosstab(df_original['Cluster'], df_original['genero'], normalize='index') * 100
print(distribuicao_genero.round(2).astype(str) + '%')