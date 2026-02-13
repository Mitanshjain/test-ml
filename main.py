import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('insurance - insurance.csv')
# print(df.head(3))

# print(df.isna().sum())  

x = df.drop(columns=['charges'])
y = df['charges']

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y.shape)
# print(y_train.shape)
# print(y_test.shape)


ohe = OneHotEncoder(drop = 'first' , sparse_output = False )
x_train_sex_smoker_region = ohe.fit_transform(x_train[['sex' , 'smoker','region']])
x_test_sex_smoker_region = ohe.fit_transform(x_test[['sex' , 'smoker','region']])
print(x_train_sex_smoker_region.shape)


x_train_age_bmi_children = x_train.drop(columns =['smoker', 'region','sex']).values
x_test_age_bmi_children = x_test.drop(columns =['smoker', 'region','sex']).values
print(x_train_age_bmi_children.shape)


x_train_transformed = np.concatenate((x_train_age_bmi_children ,x_train_sex_smoker_region) , axis = 1)
print(x_train_transformed.shape)


