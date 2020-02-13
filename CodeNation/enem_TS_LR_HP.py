import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

d_Train= pd.read_csv('dados/train.csv')
colunas = ('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT')
#colunas = ('NU_NOTA_CN','NU_NOTA_LC','NU_NOTA_MT')
df = d_Train.loc[:,colunas]
df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)

df.update(df.fillna(-236))
X_Data = df.iloc[:,:-1]
Y_Target = df.iloc[:,-1]
padronizacao = StandardScaler().fit(X_Data)
X_p = padronizacao.transform(X_Data)


d_Test= pd.read_csv('dados/test.csv')
colunas = ('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')
#colunas = ('NU_NOTA_CN','NU_NOTA_LC')
df_test = d_Test.loc[:,colunas]

df_test.update(df_test.fillna(-236))

X_Data_Test=df_test
X_r = padronizacao.transform(X_Data_Test)

ols = LinearRegression()
print(ols.get_params().keys())
ols_params = {'fit_intercept':[True, False],
              'normalize':[True, False]}

X_train, X_test, Y_train, Y_test = train_test_split(X_p, Y_Target, test_size=0.25, random_state = 5)
ols.fit(X_p, Y_Target)

pred_train = ols.predict(X_train)
pred_test = ols.predict(X_test)
final = ols.predict(X_r)
print(ols.score(X_test,Y_test))


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
novo = ds_maiores_M.loc[:,('NU_INSCRICAO','NU_NOTA_MT')]
novo['NU_NOTA_MT'] = novo['NU_NOTA_MT'].div(mt,axis = 0)
novo['NU_NOTA_MT'] = round(novo['NU_NOTA_MT'],1)
novo.to_csv("answer.csv",index=False, header=True)
print('Final')
