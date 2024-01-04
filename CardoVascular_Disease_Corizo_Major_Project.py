#!/usr/bin/env python
# coding: utf-8

# # Perform data pre-processing operations.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('cardo_train.csv',sep=";")
df


# In[ ]:


df.describe()


# In[ ]:


# df.dropna()


# In[ ]:


df.columns


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# # data analysis and visualizations draw all the possible plots to provide essential informations and to derive some meaningful insights.

# In[ ]:


import matplotlib.pyplot as plt
x=df['cholesterol']
y=df['cardio']
plt.plot(x,y,color='r')
plt.xlabel('cholesterol')
plt.ylabel('cardio')
plt.title("clolestrol vs cardio")
plt.show()


# In[ ]:


x=df['smoke']
y=df['cardio']
plt.plot(x,y,color='r')
plt.xlabel('smoke')
plt.ylabel('cardio')
plt.title("smoke vs cardio")
plt.show()


# In[ ]:


x=df['alco']
y=df['cardio']
plt.scatter(x, y, color='blue', marker='*')
plt.xlabel('alco')
plt.ylabel('cardio')
plt.title("alco vs cardio")
plt.show()


# In[ ]:


import seaborn as sns
sns.lineplot(x="active",y="cardio",data = df)
plt.xlabel('active')
plt.ylabel('cardio')
plt.title("active vs cardio")
plt.show()


# In[ ]:


import seaborn as sns
sns.histplot(x="gluc",y="cardio",data = df)
plt.xlabel('gluc')
plt.ylabel('cardio')
plt.title("gluc vs cardio")
plt.show()


# In[ ]:


import seaborn as sns
sns.lmplot(x="ap_hi",y="cardio",data = df)
plt.xlabel('ap_hi')
plt.ylabel('cardio')
plt.title("ap_hi vs cardio")
plt.show()


# In[ ]:


import seaborn as sns
sns.scatterplot(x="ap_lo",y="cardio",data = df)
plt.xlabel('ap_lo')
plt.ylabel('cardio')
plt.title("ap_lo vs cardio")
plt.show()


# In[ ]:


import seaborn as sns
sns.barplot(x="active",y="cardio",data = df)
plt.xlabel('active')
plt.ylabel('cardio')
plt.title("active vs cardio")
plt.show()


# # Show your correlation matrix of features according to the datasets.

# In[ ]:


import pandas as pd
import seaborn as sns

data = pd.read_csv('cardo_train.csv',sep=";")

corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True)


# In[ ]:


x=df.iloc[:,0:12]
x


# In[ ]:


y=df.iloc[:,12:]
y


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


predicted=model.predict(x_test)
predicted


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model2= LogisticRegression()


# In[ ]:


model2.fit(x_train,y_train)


# In[ ]:


model2.score(x_test,y_test)


# In[ ]:


predicted=model2.predict(x_test)
predicted


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted)
cm


# In[ ]:


import seaborn as sns
plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("actual")
plt.show()


# # Random forest

# In[ ]:


# random forest
from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()


# In[ ]:


model3.fit(x_train,y_train)


# In[ ]:


model3.score(x_test,y_test)


# In[ ]:


pred=model3.predict(x_test)
pred


# # Decision tree

# In[ ]:


# decision tree

from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion="entropy")
model4.fit(x_train,y_train)


# In[ ]:


model4.score(x_test,y_test)


# In[ ]:


pred=model4.predict(x_test)
pred


# In[ ]:


# from sklearn import tree
# tree.plot_tree(model4)


# # k nearest Neighbors

# In[ ]:


#knn classifier

from sklearn.neighbors import KNeighborsClassifier
model5=KNeighborsClassifier()
model5.fit(x_train,y_train)


# In[ ]:


model5.score(x_test,y_test)


# In[ ]:


pred=model.predict(x_test)
pred


# In[ ]:


plt.scatter(df.active,df.cardio)


# # K Means Clustering

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


df


# In[ ]:


km=KMeans(n_clusters=5)
km.fit(df[["id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","cardio"]])


# In[ ]:


km.cluster_centers_


# In[ ]:


df["cluster_group"]=km.labels_


# In[ ]:


df


# In[ ]:


df["cluster_group"].value_counts()


# In[ ]:


sns.scatterplot(x="gender",y="cardio",data=df,hue="cluster_group")


# # support vector machine

# In[ ]:


#svm

from sklearn.svm import SVC
model6 = SVC()


# In[ ]:


x=df.iloc[:,0:12]
x


# In[ ]:


# import numpy as np

# # Create a column vector y
# y = df.iloc[:,12:0]

# # Convert y to a 1D array using ravel()
# y_1d = np.ravel(y)

# # Check the shape of y_1d
# print(y_1d.shape)  


# In[ ]:


model6.fit(x_train,y_train)


# In[ ]:


model6.score(x_test,y_test)


# In[ ]:


pred=model6.predict(x_test)
pred


# In[ ]:




