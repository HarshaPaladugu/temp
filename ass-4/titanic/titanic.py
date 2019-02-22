import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("/home/deeepika/Desktop/Demo/train_and_test2.csv")
#print(df)
x = df[['Age','Sex','Pclass']]
#print(x)
y = df['Survived']
#print(y)
w = np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))
print('w is',w)
yhat = np.dot(x,w)
sig = 1.0/(1+np.exp(-yhat))
#error calculation
#d1 = (y-sig)
#d2 = (y-y.mean())
#error = d1.dot(d1)/d2.dot(d2)
error=y*np.log(sig)
print('error is',error)
#accuracy
acc = 1-error
print('accuracy is',acc)
plt.scatter(df['Age'],sig,color='green')
#plt.scatter()
#plt.plot(df['Sex'],sig)
plt.show()
#sig = 1.0/(1+np.exp(-yhat))
age = int(input("enter age"))
sex = int(input("enter sex"))
pclass = int(input("enter passenger class"))
estima = age*w[0]+sex*w[1]+pclass*w[2] 
#print("estimate is",estima)
qua = 1.0/(1+np.exp(-estima))
print("survived status",qua)
if qua>0.5:
    print("Survived")
else:
    print('Died')

    
    
    



