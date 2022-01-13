import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

f=open('data.csv')
raw_data=list(csv.reader(f))
raw_data.pop(0)

temp=[]
melted=[]
for i in raw_data:
    tem=float(i[0])
    melt=float(i[1])

    temp.append(tem)
    melted.append(melt)

X=np.reshape(temp,(len(temp),1))
Y=np.reshape(melted,(len(melted),1))

lr=LogisticRegression()
lr.fit(X,Y)
plt.figure()
plt.scatter(X.ravel(),Y,color='black',zorder=20)
def model(x):
    return 1/(1+np.exp(-x))
X_test=np.linspace(0,5000,10000)
melting_chances=model(X_test*lr.coef_+lr.intercept_).ravel()

plt.plot(X_test,melting_chances,color='red',linewidth=3)
plt.axhline(y=0,color='k',linestyle='-')
plt.axhline(y=1,color='k',linestyle='-')
plt.axhline(y=0.5,color='b',linestyle='--')

plt.axvline(x=X_test[6843],color='b',linestyle='--')
plt.xlim(3400,3450)

plt.show()

