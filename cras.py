import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Carregamento e Preparação dos Dados
# ==========================================
# Lê o arquivo que você disponibilizou (ajuste o caminho se necessário)
df = pd.read_csv('coleta_final_bcc_4.csv', delimiter=';')

# Limpeza básica
df = df.dropna(subset=['Hora', 'Minuto', 'Data', 'Cenario'])

# Converte horas e minutos para o total absoluto em minutos
df['Total_Minutos'] = df['Hora'] * 60 + df['Minuto']

# Ordena os eventos cronologicamente para garantir que o sequenciamento esteja correto
df = df.sort_values(by=['Data', 'Cenario', 'Total_Minutos', 'Sequencia'])

# Calcula o Tempo Entre Chegadas (TEC) fazendo a diferença entre o registro atual e o anterior
df['TEC'] = df.groupby(['Data', 'Cenario'])['Total_Minutos'].diff()

# Remove a primeira linha de cada grupo (que fica vazia ao fazer o diff)
tec_data = df['TEC'].dropna().reset_index(drop=True)

# ==========================================
# 2. Tratamento de Dados (Outliers)
# ==========================================
Q1 = tec_data.quantile(0.25)
Q3 = tec_data.quantile(0.75)
A = Q3 - Q1
limite_sup_extremo = Q3 + 3 * A

# Filtramos removendo os outliers extremos (aqueles valores > 20 minutos que distorciam a variância)
df_sem_discrepancia = tec_data[tec_data <= limite_sup_extremo]

print("--- Resumo Estatístico (Sem Outliers) ---")
print(f"Média: {df_sem_discrepancia.mean():.2f} minutos")
print(f"Mediana: {df_sem_discrepancia.median():.2f} minutos")
print(f"Desvio-padrão: {df_sem_discrepancia.std():.2f}")
print(f"Variância: {df_sem_discrepancia.var():.2f}\n")

# ==========================================
# 3. Visualização de Dados (Gráficos)
# ==========================================
sns.set_theme(style="whitegrid")

# -> A) Gráfico de Dispersão (Correlação: Observação k vs k+1)
# Diferente do supermercado, plotamos n vs n+1 para provar a independência dos dados
x = df_sem_discrepancia.iloc[:-1].values
y = df_sem_discrepancia.iloc[1:].values

plt.figure(figsize=(8, 5))
sns.scatterplot(x=x, y=y, color='#1f77b4', alpha=0.6, edgecolor='k')
plt.title('Diagrama de Dispersão: Observação $k$ vs $k+1$')
plt.xlabel('Observação $k$')
plt.ylabel('Observação $k+1$')
plt.show()

# -> B) Histograma com Regra de Sturges
n = len(df_sem_discrepancia)
K = int(np.ceil(1 + 3.3 * np.log10(n))) # Aplicação da regra matemática

plt.figure(figsize=(8, 5))
sns.histplot(data=df_sem_discrepancia, bins=K, color='#2ca02c', edgecolor='black', kde=True)
plt.title(f'Histograma do TEC CadÚnico ({K} classes)')
plt.xlabel('Tempo Entre Chegadas (Minutos)')
plt.ylabel('Frequência')
plt.show()

# -> C) Curva da Média Cumulativa
# Recurso legal do notebook do supermercado para ver se o sistema atinge o "Steady State"
media_cumulativa = df_sem_discrepancia.expanding().mean()

plt.figure(figsize=(8, 5))
sns.lineplot(data=media_cumulativa, color='#d62728', linewidth=2)
plt.title('Curva de Convergência da Média')
plt.xlabel('Número de Chegadas')
plt.ylabel('Média Cumulativa (Minutos)')
plt.axhline(df_sem_discrepancia.mean(), color='black', linestyle='--', label='Média Final')
plt.legend()
plt.show()