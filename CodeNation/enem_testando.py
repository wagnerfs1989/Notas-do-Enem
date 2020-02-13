


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



df = pd.read_csv('dados/test.csv')
df.update(df.fillna(0))
df_en = df.iloc[:,1:]
df_en = df_en.drop(['TP_ENSINO','TP_DEPENDENCIA_ADM_ESC','Q027'], axis = 1)
df_en['NU_NOTA_MT'] = 0
head = df.columns
 
df_data = df_en.iloc[:,:-1]
df_target = df_en.iloc[:,-1]
Z_data = df_data
W_target = df_target

encoder = LabelEncoder()
Z_data = Z_data.apply(encoder.fit_transform)

dt = pd.read_csv('dados/train.csv')
dt_en = dt.loc[:,head]
dt_en = dt_en.iloc[:,1:]
dt_en = dt_en.drop(['TP_ENSINO','TP_DEPENDENCIA_ADM_ESC','Q027'], axis = 1)
nota = dt.loc[:,'NU_NOTA_MT']
dt_en['NU_NOTA_MT'] = nota
dt_en.update(dt_en.fillna(0))

dt_data = dt_en.iloc[:,:-1]
dt_target = dt_en.iloc[:,-1]
X_data = dt_data
y_target = dt_target
# encoder = LabelEncoder()

# dt_en = dt_en.apply(encoder.fit_transform)

X_data = X_data.apply(encoder.fit_transform)

lm = LinearRegression()
lm.fit(X_data,y_target)


print(pd.DataFrame(list(zip(X_data.columns, lm.coef_)), columns = ['features','estimatedCoefficients']))


plt.scatter(X_data.SG_UF_RESIDENCIA, y_target)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_target, test_size=0.33, random_state = 5)
lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
nota_test = lm.predict(Z_data)
lista = nota_test
x = 0
num_elementos_lista = len(lista)
#while(x < num_elementos_lista):
#    lista[x] = round(lista[x],1)
#    if(lista[x] < 0):
#        lista[x] = 0
#    elif(lista[x] >1000):
#        lista[x] = 1000
#    x+=1

nota_test = lista
teste = pd.Series(nota_test)
novo = pd.DataFrame(data=df['NU_INSCRICAO'] ,columns=['NU_INSCRICAO'])
novo['NU_NOTA_MT'] = teste
novo = novo.sort_values(by=['NU_NOTA_MT'],ascending=False).head(20)
novo.to_csv("answer2.csv")

