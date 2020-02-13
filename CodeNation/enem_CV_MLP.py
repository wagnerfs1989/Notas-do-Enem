import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

d_Train= pd.read_csv('dados/train.csv')
colunas = ('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT')
#colunas = ('NU_NOTA_CN','NU_NOTA_LC','NU_NOTA_MT')
df = d_Train.loc[:,colunas]
df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)

df.update(df.fillna(-236))

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

df_test.update(df_test.fillna(-236))

X_Data_Test=df_test
Y_Target_Test = 0
X_r = padronizacao.transform(X_Data_Test)

solver = mlp = MLPRegressor(alpha = 0.001,max_iter=1000,learning_rate_init=0.001,hidden_layer_sizes=(100,10), verbose = False)

kf = KFold(n_splits=10)

# Ajustar e sintonizar o modelo

cross_validate(solver.fit(X_p,Y_Target), X_p,Y_Target,cv=kf)

print(solver.score(X_p,Y_Target))
pred_notas = solver.predict(X_r)
final = pred_notas

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
