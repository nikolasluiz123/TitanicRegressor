import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data.data_processing import get_train_data

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)
y = df_train['sobreviveu']

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criando o seletor RFE
rfe = RFECV(model)  # Seleciona 5 variáveis
rfe = rfe.fit(X_train, y_train)

# Variáveis selecionadas
selected_features = X_train.columns[rfe.support_]
print("Variáveis selecionadas:", selected_features)

# Selecionando as 5 melhores variáveis com base em correlação (f_regression)
selector = SelectKBest(score_func=f_regression, k=3)
selector.fit(X_train, y_train)

# Variáveis selecionadas
selected_features = X_train.columns[selector.get_support()]
print("Variáveis selecionadas:", selected_features)