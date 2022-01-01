"""this code helps us to predict the no of omicron cases"""  
import pandas as vp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
data=vp.read_csv('cases.csv',sep=',')
data=data[['id','cases']]

x=np.array(data['id']).reshape(-1,1)
y=np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')

pf=PolynomialFeatures(degree=3)
x=pf.fit_transform(x)

model=LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)}%')
z=model.predict(x)
plt.plot(z,'--b')
for i in range(1,30):
    print("\n",7+i ,"th day:\t")
    print(model.predict(pf.fit_transform([[7+i]])))
x1=np.array(list(range(1,7+i))).reshape(-1,1)
y1=model.predict(pf.fit_transform(x1))
plt.plot(y1,'--b')
plt.plot(z,'--r')
plt.show()
