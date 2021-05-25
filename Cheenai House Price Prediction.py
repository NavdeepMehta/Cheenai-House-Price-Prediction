#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv('chennai_house_price_prediction.csv')


# In[3]:


df


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


temp=pd.DataFrame(index=df.columns)
temp['data_type'] = df.dtypes
temp['null_count'] = df.isnull().sum()
temp['unique_count'] = df.nunique()
temp


# In[8]:


df.describe()


# In[9]:


df.describe(include='all')


# In[10]:


df.drop('PRT_ID',inplace=True,axis=1)


# In[11]:


sns.set_style('whitegrid')


# In[12]:


plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True)


# In[13]:


def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    print(dataset)


# In[14]:


correlation(df,0.80)


# In[15]:


plt.figure(figsize=(15,3))
sns.countplot(df['AREA'])


# In[16]:


df['AREA'].value_counts()


# In[17]:


dic_area={'Chrompt':'Chrompet','Chrmpet':'Chrompet','Chormpet':'Chrompet','TNagar':'T Nagar','Karapakam':'Karapakkam',
          'Ana Nagar':'Anna Nagar','Ann Nagar':'Anna Nagar','Velchery':'Velachery','Adyr':'Adyar','KKNagar':'KK Nagar'}


# In[18]:


df.replace({'AREA':dic_area},inplace=True)


# In[19]:


x=pd.DataFrame(df.groupby('AREA')['SALES_PRICE'].mean().sort_values())

sns.barplot(x.index,x['SALES_PRICE'])


# In[20]:


plt.figure(figsize=(15,3))
sns.countplot(df['SALE_COND'])


# In[21]:


df['SALE_COND'].value_counts()


# In[22]:


dict_sale = {'Adj Land':'AdjLand','Ab Normal':'AbNormal','Partiall':'Partial','PartiaLl':'Partial'}


# In[23]:


df.replace({'SALE_COND':dict_sale},inplace=True)


# In[24]:


x=pd.DataFrame(df.groupby('SALE_COND')['SALES_PRICE'].mean().sort_values())

sns.barplot(x.index,x['SALES_PRICE'])


# In[25]:


sns.countplot(df['BUILDTYPE'])


# In[26]:


df['BUILDTYPE'].value_counts()


# In[27]:


dict_build = {'Other':'Others','Comercial':'Commercial'}


# In[28]:


df.replace({'BUILDTYPE':dict_build},inplace=True)


# In[29]:


x=pd.DataFrame(df.groupby('BUILDTYPE')['SALES_PRICE'].mean().sort_values())

sns.barplot(x.index,x['SALES_PRICE'])


# In[30]:


sns.countplot(df['PARK_FACIL'])


# In[31]:


dict_park={'Noo':'No'}


# In[32]:


df.replace({'PARK_FACIL':dict_park},inplace=True)


# In[33]:


x=pd.DataFrame(df.groupby('PARK_FACIL')['SALES_PRICE'].mean().sort_values().sort_values())

sns.barplot(x.index,x['SALES_PRICE'])


# In[34]:


plt.figure(figsize=(15,3))
sns.countplot(df['UTILITY_AVAIL'])


# In[35]:


dict_utility ={'All Pub':'AllPub'}


# In[36]:


df.replace({'UTILITY_AVAIL':dict_utility},inplace=True)


# In[37]:


x=pd.DataFrame(df.groupby('UTILITY_AVAIL')['SALES_PRICE'].mean().sort_values())
sns.barplot(x.index,x['SALES_PRICE'])


# In[38]:


sns.countplot(df['STREET'])


# In[39]:


dict_street = {'Pavd':'Paved','NoAccess':'No Access'}


# In[40]:


df.replace({'STREET':dict_street},inplace=True)


# In[41]:


x=pd.DataFrame(df.groupby('STREET')['SALES_PRICE'].mean().sort_values())
sns.barplot(x.index,x['SALES_PRICE'])


# In[42]:



sns.countplot(df['MZZONE'])


# In[43]:


x=pd.DataFrame(df.groupby('MZZONE')['SALES_PRICE'].mean().sort_values())
sns.barplot(x.index,x['SALES_PRICE'])


# In[44]:


##sns.pairplot(df.drop(['N_BEDROOM', 'N_BATHROOM'],axis=1),hue='AREA')


# In[45]:


sns.histplot(df['INT_SQFT'],kde=True)


# In[46]:


sns.scatterplot(df['INT_SQFT'],df['SALES_PRICE'],hue=df['AREA'])


# In[47]:


sns.histplot(df['DIST_MAINROAD'],kde=True)


# In[48]:


sns.scatterplot(df['DIST_MAINROAD'],df['SALES_PRICE'],hue=df['AREA'])


# In[49]:


df.columns


# In[50]:


sns.histplot(df['COMMIS'],kde=True)


# In[51]:


df['N_BATHROOM'].value_counts()


# In[52]:


df['N_BEDROOM'].value_counts()


# In[53]:


sns.boxplot(df['N_BATHROOM'],df['N_BEDROOM'])


# In[54]:


sns.scatterplot(df['N_BEDROOM'],df['SALES_PRICE'],hue=df['AREA'])


# In[55]:


sns.scatterplot(df['N_BATHROOM'],df['SALES_PRICE'],hue=df['AREA'])


# In[56]:


sns.scatterplot(df['QS_ROOMS'],df['SALES_PRICE'],hue=df['AREA'])


# In[57]:


sns.scatterplot(df['QS_BEDROOM'],df['SALES_PRICE'],hue=df['AREA'])


# In[58]:


sns.scatterplot(df['QS_BATHROOM'],df['SALES_PRICE'],hue=df['AREA'])


# In[59]:


sns.scatterplot(df['QS_OVERALL'],df['SALES_PRICE'],hue=df['AREA'])


# In[60]:


def Qs_Overall(cols):
    QS_ROOMS=cols[0]
    QS_BATHROOM = cols[1]
    QS_BEDROOM = cols[2]
    QS_OVERALL = cols[3]
    if pd.isnull(QS_OVERALL):
        QS_OVERALL = QS_ROOMS + QS_BATHROOM + QS_BEDROOM
        return QS_OVERALL
    else:
        return QS_OVERALL


# In[61]:


df['QS_OVERALL']=df[['QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM','QS_OVERALL']].apply(Qs_Overall,axis=1)


# In[62]:


df.groupby('N_BEDROOM')['SALES_PRICE'].median()


# In[63]:


x=df[df['N_BEDROOM'].isnull()]


# In[64]:


x


# In[65]:


df['N_BEDROOM'].fillna(value=1.0, limit=1,inplace=True)


# In[66]:


def N_Bathroom(cols):
    N_BEDROOM=cols[0]
    N_BATHROOM = cols[1]
    if pd.isnull(N_BATHROOM):
        if N_BEDROOM == 1.0 or N_BEDROOM == 2.0 :
            return 1.0
        else:
            return 2.0
    else:
        return N_BATHROOM


# In[67]:


df['N_BATHROOM']=df[['N_BEDROOM', 'N_BATHROOM']].apply(N_Bathroom,axis=1)


# In[68]:


sns.boxplot(df['INT_SQFT'])


# In[69]:


sns.boxplot(df['DIST_MAINROAD'])


# In[70]:


sns.boxplot(df['COMMIS'])


# In[71]:


sns.boxplot(df['SALES_PRICE'])


# In[72]:


df.isnull().sum()


# In[73]:


def remove_outlier(dataset,k,col):
    mean = dataset[col].mean()
    global df1      
    std = dataset[col].std()    
    outlier = [i for i in dataset[col] if (i > mean - k * std)]
    outlier = [i for i in outlier if (i < mean + k * std)]       
    df1 = dataset.loc[dataset[col].isin(outlier)]


# In[74]:


remove_outlier(df,3.0,'COMMIS')


# In[75]:


remove_outlier(df1,3.0,'SALES_PRICE')


# In[76]:


sns.boxplot(df1['COMMIS'])


# In[77]:


sns.boxplot(df1['SALES_PRICE'])


# In[78]:


df1.dtypes


# In[79]:


for label,content in df1.items():
    if df1[label].dtypes == 'object':
        df1[label] = pd.Categorical(content).codes+1
    else:
        pass


# In[80]:


for label in df1.columns:
    if df1[label].dtypes == 'int8':
        df1[label] = df1[label].astype(np.int64)
    else:
        pass


# In[81]:


df1.dtypes


# In[82]:


df1


# In[83]:


df1.columns


# In[246]:


df1


# In[247]:


# copy the data
df_max_scaled = df1.drop(['COMMIS','SALES_PRICE'],axis=1)
  
# apply normalization techniques
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
      
# view normalized data
display(df_max_scaled)


# In[84]:


from sklearn.model_selection import train_test_split as tts


# In[98]:


X_train,x_test,Y_train,Y_test=tts(df1.drop(['COMMIS','SALES_PRICE','QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL'],axis=1),df1['SALES_PRICE'],test_size=.20,random_state=10)


# In[111]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)
model.fit(X_train,Y_train)


# In[112]:


from sklearn.metrics import mean_absolute_error as mae


# In[113]:


train_predict =model.predict(X_train)
k = mae(train_predict, Y_train)
print('Training Mean Absolute Error', k )


# In[114]:


test_predict =model.predict(x_test)
k = mae(test_predict, Y_test)
print('Testing Mean Absolute Error', k )


# In[115]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()


# In[116]:


plot_features(X_train.columns,model.feature_importances_)


# In[123]:


X_train,x_test,Y_train,Y_test=tts(df1.drop(['COMMIS','SALES_PRICE','QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL','SALE_COND','UTILITY_AVAIL'],axis=1),df1['SALES_PRICE'],test_size=.30,random_state=10)


# In[124]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)
model.fit(X_train,Y_train)


# In[125]:


train_predict =model.predict(X_train)
k = mae(train_predict, Y_train)
print('Training Mean Absolute Error', k )


# In[126]:


test_predict =model.predict(x_test)
k = mae(test_predict, Y_test)
print('Testing Mean Absolute Error', k )


# In[145]:


from sklearn.model_selection import RandomizedSearchCV
np.random.seed(np.arange(1,100,1))
grid = {
    "n_estimators":np.arange(10,100,10),
    "max_depth":[None,3,5,10],
    "min_samples_split":np.arange(2,20,2),
    "min_samples_leaf":np.arange(1,20,2),
    "max_features": [0.5,1,"sqrt","auto"],
    "max_samples":[900,1800,2700,3600],
    "random_state":np.arange(1,100,1)

}
rs_model = RandomizedSearchCV(
RandomForestRegressor(n_jobs=-1),
                    param_distributions = grid,
                     n_iter=100,
                    cv=5,
                    verbose=True)
rs_model.fit(X_train,Y_train)


# In[146]:


rs_model.best_params_


# In[147]:


y_preds_rs = rs_model.predict(x_test)
mae_hyp = mae(Y_test,y_preds_rs)
mae_hyp


# In[148]:


y_preds_rs = rs_model.predict(X_train)
mae_hyp = mae(Y_train,y_preds_rs)
mae_hyp


# In[150]:


import pickle
pickle.dump(rs_model, open('model.pkl','wb'))


# In[ ]:




