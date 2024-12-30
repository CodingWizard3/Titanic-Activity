import matplotlib.pyplot as pt
import seaborn as sb
import pandas as pd
import numpy as np

titanic=pd.read_csv(r"titanic.csv")
print(titanic.head())
print(titanic.shape)
print(titanic.isnull().sum())
sb.heatmap(titanic.isnull(),cmap="spring")
pt.show()

#titanic.drop('deck', axis=1, inplace= True)
titanic.dropna(inplace=True)

print(pd.get_dummies(titanic['sex']).head())
sex=pd.get_dummies(titanic['sex'],drop_first=True)
embarked=pd.get_dummies(titanic['embarked'],drop_first=True)
pclass=pd.get_dummies(titanic['pclass'],drop_first=True)

new=pd.concat([titanic,sex,embarked,pclass])
print(new.head())