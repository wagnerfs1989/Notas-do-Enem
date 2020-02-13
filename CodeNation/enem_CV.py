import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


d_Train= pd.read_csv('dados/train.csv')
colunas = ('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT')
#colunas = ('NU_NOTA_CN','NU_NOTA_LC','NU_NOTA_MT')
df = d_Train.loc[:,colunas]
df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)
df.update(df.fillna(-100))

X_Data = df.iloc[:,:-1]
Y_Target = df.iloc[:,-1]
#X_Data, Y_Target = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
padronizacao = StandardScaler().fit(X_Data)
#X_Data, Y_Target = shuffle(X_Data, Y_Target, random_state=13)
X_p = padronizacao.transform(X_Data)


d_Test= pd.read_csv('dados/test.csv')
colunas = ('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')
#colunas = ('NU_NOTA_CN','NU_NOTA_LC')
df_test = d_Test.loc[:,colunas]
df_test.update(df_test.fillna(-100))
X_Data_Test=df_test
Y_Target_Test = 0
X_r = padronizacao.transform(X_Data_Test)

kf = KFold(n_splits=10)

print('*****')
print('*****')
print('CV')

print('Linear Regression')
lm = LinearRegression()
predictions_n = cross_val_predict(lm, X_p,Y_Target,cv=kf)
fig, ax = plt.subplots()
ax.scatter(Y_Target, predictions_n, edgecolors=(0, 0, 0))
ax.plot([Y_Target.min(), Y_Target.max()], [Y_Target.min(), Y_Target.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print('*****')
print('Random Forest')
print('*****')
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, warm_start=True)
predictions_rf = cross_val_predict(rf, X_p,Y_Target,cv=kf)
#print(cross_validate(rf, X_p,Y_Target,cv=kf))
fig, ax = plt.subplots()
ax.scatter(Y_Target, predictions_rf, edgecolors=(0, 0, 0))
ax.plot([Y_Target.min(), Y_Target.max()], [Y_Target.min(), Y_Target.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print('*****')
print('Padronização')
print('*****')
rf.fit(X_p, Y_Target)
pred_train = rf.predict(X_p)
print(rf.score(X_p,Y_Target))
final = rf.predict(X_r)



#X_train, X_test, Y_train, Y_test = train_test_split(X_p, Y_Target, test_size=0.25, random_state = 5)
#lm = LinearRegression()
#lm.fit(X_train, Y_train)
#pred_train = lm.predict(X_train)
#pred_test = lm.predict(X_test)
#final = lm.predict(X_r)
#print(lm.score(X_train,Y_train))
#
#X_train, X_test, Y_train, Y_test = train_test_split(X_p, Y_Target, test_size=0.25, random_state = 5)
#rf = RandomForestRegressor()
#rf.fit(X_train, Y_train)
#pred_train = rf.predict(X_train)
#pred_test = rf.predict(X_test)
#final = rf.predict(X_r)
#print(rf.score(X_train,Y_train))

lista = final
x = 0
num_elementos_lista = len(lista)

while(x < num_elementos_lista):
    lista[x] = round(lista[x],1)
    if(lista[x] < 0):
        lista[x] = 'NaN'
    elif(lista[x] >1000):
        lista[x] = 1000
    x+=1
final = lista

ds_maiores = d_Test.loc[:,('NU_INSCRICAO','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')]     
ds_maiores['NU_NOTA_MT']=final
cn,ch,lc,red,mt = [2,1,1.5,3,3]
nota_T = cn+ch+lc+red+mt
ds_maiores_M = ds_maiores
ds_maiores_M['NU_NOTA_CN'] = pd.DataFrame(ds_maiores['NU_NOTA_CN'].mul(cn, axis= 0))
ds_maiores_M['NU_NOTA_CH'] = pd.DataFrame(ds_maiores['NU_NOTA_CH'].mul(ch, axis= 0))
ds_maiores_M['NU_NOTA_LC'] = pd.DataFrame(ds_maiores['NU_NOTA_LC'].mul(lc, axis= 0)) 
ds_maiores_M['NU_NOTA_REDACAO'] = pd.DataFrame(ds_maiores['NU_NOTA_REDACAO'].mul(red, axis= 0))  
ds_maiores_M['NU_NOTA_MT'] = pd.DataFrame(ds_maiores['NU_NOTA_MT'].mul(mt, axis= 0))
#ds_maiores_M = ds_maiores_M.fillna(0)
ds_maiores_M['NU_NOTA_MD'] = (ds_maiores_M['NU_NOTA_CN']+ds_maiores_M['NU_NOTA_CH']+ds_maiores_M['NU_NOTA_LC']+ds_maiores_M['NU_NOTA_REDACAO']
+ds_maiores_M['NU_NOTA_MT'])/nota_T
novo = ds_maiores_M.sort_values(by=['NU_NOTA_MD'],ascending=False)
novo = novo.loc[:,('NU_INSCRICAO','NU_NOTA_MT')]
novo['NU_NOTA_MT'] = novo['NU_NOTA_MT'].div(mt,axis = 0)
novo['NU_NOTA_MT'] = round(novo['NU_NOTA_MT'],1)
novo.to_csv("answer.csv",index=False)
