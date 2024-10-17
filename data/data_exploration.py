import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

df_treino = pd.read_csv('train.csv')

print('Dados de Treino:')
print(tabulate(df_treino.head(), headers='keys', tablefmt='psql'))
print()

df_treino.columns = ['id_passageiro', 'sobreviveu', 'classe_social', 'nome', 'sexo', 'idade', 'qtd_irmaos_conjuges',
                     'qtd_pais_filhos', 'ticket', 'valor_ticket', 'cabine', 'porta_embarque']

df_treino.drop(columns=['id_passageiro', 'nome', 'ticket', 'cabine'], inplace=True, axis=1)

print('Info:')
print(df_treino.info())
print()

print('Pessoas com idade null:')
print(tabulate(df_treino[df_treino['idade'].isnull()], headers='keys', tablefmt='psql'))
print('Quantidade de Pessoas com idade null: ', df_treino[df_treino['idade'].isnull()].shape[0])
print()

df_treino.dropna(subset=['idade'], inplace=True)

print('Dados de Treino Tratados:')
print(tabulate(df_treino.head(), headers='keys', tablefmt='psql'))
print()


