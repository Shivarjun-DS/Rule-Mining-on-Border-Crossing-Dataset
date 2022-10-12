#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt


# In[6]:


#Data = https://data.bts.gov/Research-and-Statistics/Border-Crossing-Entry-Data/keg4-3bc2/data
df = pd.read_csv(r"C:\Users\SHIVARJUN\Downloads\archive (1)\Border_Crossing_Entry_Data.csv", nrows = 9000)
new_col= [c.replace(' ', '_') for c in df.columns]
df.columns = new_col


# In[7]:


df.describe()


# In[20]:


df2 = df


# In[43]:


df['Port_Code'] = df['Port_Code'].map(str)
df['Value'] = df['Value'].map(str)
df['Border'] = df['Border'].map(str)
df['Measure'] = df['Measure'].map(str)
df['State'] = df['State'].map(str)
df['Date'] = df['Date'].map(str)
df['Port_Name'] = df['Port_Name'].map(str)


# In[23]:


#df.drop(['Date'], axis = 1, inplace = True)


# In[35]:


#df.drop(['Date', 'Port_Code', 'Value'], axis = 1, inplace = True)


# In[85]:


df.columns


# In[31]:


new_col


# In[44]:


df_list = df.values.tolist()


# In[8]:


#lis = []
#for i in range(0, 6000):
    #lis.append([str(df.values[i,j]) for j in range(0, 3)])


# In[ ]:





# In[45]:


te = TransactionEncoder()
te_array = te.fit(df_list).transform(df_list)
dff = pd.DataFrame(te_array, columns=te.columns_)


# In[46]:


dff


# In[51]:


def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
    
dff = dff.applymap(encode_units)
dff.head(10)


# In[88]:


from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(dff, min_support = 0.08, use_colnames=True)
frequent_itemsets


# In[89]:


from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_itemsets, metric= 'confidence', min_threshold=0.2)
res


# In[90]:


res2 = res.sort_values('lift', ascending = False)
res2[:60]


# In[19]:


plt.scatter(res['support'], res['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:





# In[14]:


plt.scatter(res['support'], res['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()


# In[ ]:




