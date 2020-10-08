#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

Data = pd.read_csv (r"C://Users//priya//OneDrive//Desktop//StudentsPerformance.csv")
Data.head()


# # Exploratory Data Analysis

# In[8]:


print(Data.shape)


# In[9]:


Data.info()


# In[10]:


print(Data.columns)


# In[31]:


# type of each: they are not list
print(type(Data.columns))
print(type(Data.index))


# # Checking Null Values

# In[12]:


Missing = Data.isnull().sum()
Missing


# # Number of unique values

# In[13]:


unique = Data.nunique()
unique


# In[14]:


Data.describe(include=['object'])


# # Statistics

# In[23]:


Data.describe()


# #  Creation of a variable, "avg_score", the average of the 3 marks

# In[32]:


Data['avg_score'] = round(Data[['math score','reading score','writing score']].sum(axis=1)/3,1)
Data.head()


# # "result": 1 if failed 0 if succeed.
# # cut-off at a score >= 70 

# In[36]:


def test(data):
    if data['avg_score'] >= 70:
        return 1
    else:
        return 0

Data['result'] = Data.apply(test, axis=1)
Data.head()


# # percentage of fail & pass

# In[37]:


count_failed = len(Data[Data['result']==0])
count_passed = len(Data[Data['result']==1])

pct_of_passed = count_passed/(count_failed+count_passed)
print("Percentage of passed is", pct_of_passed*100)

pct_of_failed = count_failed/(count_failed+count_passed)
print("Percentage of failed is", pct_of_failed*100)


# #  Univariate Analysis

# # correlation between variable

# In[42]:


Data.corr()


# In[ ]:


#Lunch


# In[43]:


Data["lunch"].value_counts(normalize=True).plot(kind='pie')

plt.axis('equal') 
plt.show() 


# In[44]:


#Race and ethnicity


# In[45]:


Data["race/ethnicity"].value_counts(normalize=True).plot(kind='pie')

plt.axis('equal') 
plt.show() 


# In[46]:


import seaborn as sns

sns.pairplot(Data[['reading score', 'writing score']])


# # The graphs confirm the high correlation between reading and writing scores.

# In[47]:


sns.pairplot(Data[[ 'reading score', 'math score']])


# # Less corelation with math.

# In[49]:


sns.pairplot(Data)


# In[50]:


import seaborn as sns

sns.boxplot(x="test preparation course", y="avg_score", data=Data)


# # Completing the course seems to be beneficial for students who have a better average.

# In[51]:


sns.catplot(x="test preparation course", y="avg_score", hue="race/ethnicity", kind="box", data=Data);


# # Group E has better average than the other group.

# In[52]:


sns.catplot(x="test preparation course", y="avg_score", hue="parental level of education", kind="box", data=Data);


# # We observe that students with parents having an higher education degree are performing better.

# In[53]:


table=Data.pivot_table('avg_score',index='race/ethnicity',columns='result',aggfunc='mean')
table


# In[54]:


table=Data.pivot_table('avg_score',index='parental level of education',columns='result',aggfunc='mean')
table


# In[55]:


grouped_data = Data.groupby(['lunch', 'race/ethnicity'])
grouped_data['avg_score'].describe()


# In[60]:


Data['Gender'].value_counts()


# In[62]:


sns.catplot(x='Gender',kind='count',data=Data,height=4,palette='viridis')
plt.title('Gender')


# In[66]:


Data['Gender'].replace({'male':'0','female':'1'},inplace=True)


# In[67]:


Data['race/ethnicity'].value_counts()


# In[69]:


Data["race/ethnicity"].sort_values()
sns.catplot(x='race/ethnicity',kind='count',data=Data,height=4,palette='viridis',
            order=['group A','group B','group C','group D','group E'])


# In[70]:


Data['race/ethnicity'].replace({'group A':'1','group B':'2', 'group C':'3',
                               'group D':'4','group E':'5'},inplace=True)


# In[71]:


Data['lunch'].value_counts()


# In[72]:


sns.catplot(x='lunch',kind='count',data=Data,height=4,palette='viridis')


# In[73]:


Data['lunch'].replace({'free/reduced':'0','standard':'1'},inplace=True)


# In[74]:


Data['test preparation course'].value_counts()


# In[75]:


sns.catplot(x='test preparation course',kind='count',data=Data,height=4.5,palette='viridis')


# In[76]:


Data['test preparation course'].replace({'none':'0','completed':'1'},inplace=True)


# In[77]:


Data['parental level of education'].value_counts()


# In[78]:


Data['parental level of education'].replace({'some high school':'1','high school':'1',"associate's degree":'2',
                                        'some college':'3',"bachelor's degree":'4',"master's degree":'5'},inplace=True)


# In[79]:


Data.head()


# In[80]:


sns.set(rc={'figure.figsize':(20,6)})
sns.countplot(x='writing score', hue='test preparation course',data=Data, palette='viridis')
plt.title('Writing Score by Test Preparation Course')


# # We can see from this plot that most of the students who got a high score at the writing test, study at the test preparation course. We can understand that the preparation course is helping the students at the writing test

# In[84]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='math score',y='reading score', hue='Gender',data=Data, palette='viridis')
plt.title('Math score VS Readind Score by Gender')


# # We can see that there is a nice correlation between the math score and the reading score. We can also see that female (Green) has a better score at the reading exams

# In[85]:


plt.figure(figsize=(8, 4))
plt.hist(x='writing score',bins=10,data=Data)


# # We can see that the scores distribution is a Normal distrution, where most of the student's score is between 65 to 80

# In[86]:


sns.set(rc={'figure.figsize':(20,6)})
sns.countplot(x='writing score', hue='lunch',data=Data, palette='viridis')


# # We can see that the students who ate lunch, had a better score at the writing test.

# # Linear Regression

# In[99]:


x= Data[['Gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score']]


# In[109]:


y = Data['writing score']


# # Splitting Data

# In[110]:


from sklearn import model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[112]:


from sklearn.linear_model import LinearRegression
kfold = model_selection.KFold(n_splits=10)
lr = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(lr, x, y, cv=kfold, scoring=scoring)
lr.fit(x_train,y_train)
lr_predictions = lr.predict(x_test)
print('Coefficients: \n', lr.coef_)


# In[107]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, lr_predictions))
print('MSE:', metrics.mean_squared_error(y_test, lr_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))


# In[113]:


from sklearn.metrics import r2_score
print("R_square score: ", r2_score(y_test,lr_predictions))


# In[114]:


x= Data[['Gender','race/ethnicity','parental level of education','lunch','test preparation course','writing score','reading score']]


# In[115]:


y=Data['math score']


# In[116]:


from sklearn import model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[117]:


from sklearn.linear_model import LinearRegression
kfold = model_selection.KFold(n_splits=10)
lr = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(lr, x, y, cv=kfold, scoring=scoring)
lr.fit(x_train,y_train)
lr_predictions = lr.predict(x_test)
print('Coefficients: \n', lr.coef_)


# In[118]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, lr_predictions))
print('MSE:', metrics.mean_squared_error(y_test, lr_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))


# In[119]:


from sklearn.metrics import r2_score
print("R_square score: ", r2_score(y_test,lr_predictions))


# In[ ]:




