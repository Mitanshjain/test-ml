import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('insurance - insurance.csv')
# print(df.head(3))

# print(df.isna().sum())  

x = df.drop(columns=['charges'])
y = df['charges']

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)


