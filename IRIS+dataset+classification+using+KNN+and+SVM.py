
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
iris.keys()


# In[2]:


data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
data.head()


# In[3]:


target = pd.DataFrame(iris['target'], columns=['Target'])
target.head()


# In[4]:


total = pd.concat([data, target], axis=1)
total.head()


# In[5]:


#split the dataset into two of features and labels
X = total.drop(['Target'], axis=1)
y = total['Target']


# In[6]:


ax = sns.countplot(y)
plt.show()


# To be honest, count plot was supposed to be what it is showing as there ar equal number of every species of flower

# In[7]:


#Check the discription of the features.
# From this you can find if there are any missing values in the dataset
X.describe()


# In[8]:


data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
data.head()


# In[9]:


#compare every feature with every other feature
sns.pairplot(data, kind = 'scatter')
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 100)

#test_size = 0.20 refers to training data size of 80% and test data size of 20%


# # 1. K-NEAREST NEIGHBORS FOR CLASSIFICATION

# In[11]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 35)   #initially setting k = 35


# In[12]:


knn.fit(X_train, y_train)


# In[13]:


prediction = knn.predict(X_test)


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(prediction, y_test))


# In[15]:


cm = confusion_matrix(prediction, y_test)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


# As can be seen from heat map some of our predictions seems to be wrong.
# We will try to corect these and try to achieve maximum efficiency by optimising k value 

# CHOOSING THE VALUE OF K

# #we will plot the graph for different values k versus the corresponding accuracy score to fund the optimal k value

# In[16]:


acc_score = []

for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    prediction_y = knn.predict(X_test)
    acc_score.append(accuracy_score(y_test, prediction_y))


# In[17]:


plt.figure(figsize= (10,10))
plt.plot(range(1,40), acc_score, marker = 'o', linestyle='dashed', color = 'r')
plt.show()


# As it can be observed from the above graph, for maximum efficiency the optimal value of k value lies in the range of 
# 1 to 3 and or 5 to 12.
# 
# Lets check the accuracy by using different value of k.

# In[18]:


knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)


# In[19]:


y_pred = knn.predict(X_test)


# In[20]:


print("Accuracy score: {}".format(accuracy_score(y_pred, y_test) * 100))


# In[21]:


print(classification_report(y_pred, y_test))


# With k =10 the accuracy changes from 97% to 100% 

# In[22]:


#plotting confusion matrix and heat map
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


# From this heat map it can be concluded that all our predictions were corect

# # 2 . SUPPORT VECTOR MACHINES FOR CLASSIFICATION

# In[23]:


from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', C=10, gamma = 40)


# selecting random values of parameters such as kernel, C, gamma

# In[24]:


svc.fit(X_train, y_train)


# In[25]:


svc_pred = svc.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score

print(accuracy_score(svc_pred, y_test))


# In[27]:


print(classification_report(svc_pred, y_test))


# The accuracy of the model turn out to be 70 %.
# We will try to improve the accuracy by finding the most effecient values for SVM parameters 

# In[28]:


#import GridSearchCV 
from sklearn.model_selection import GridSearchCV

hyperparameters = ({'C': [0.001, 0.01, 10, 100, 1000], 'gamma': [1, 5, 10, 20, 30], 'kernel': ['rbf', 'linear']})


# In[29]:


grid = GridSearchCV(SVC(), hyperparameters, refit=True, verbose=3)


# In[30]:


grid.fit(X_train, y_train)


# In[31]:


grid.best_params_


# In[32]:


svc_grid = SVC(kernel = 'rbf', C=10, gamma=1)


# In[33]:


svc_grid.fit(X_train, y_train)


# In[34]:


svc_predict = svc_grid.predict(X_test)


# In[35]:


print(accuracy_score(svc_predict, y_test))


# In[36]:


conf_mat = confusion_matrix(svc_predict, y_test)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.show()


# So the accuracy of the model increased significantly from 70% to 97% by finding the most suitable values for 
# the SVM parameters
