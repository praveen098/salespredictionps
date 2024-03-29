#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, metrics

sns.set_style('darkgrid')
pd.options.display.float_format = '{:,.3f}'.format


get_ipython().run_cell_magic('time', '', "\nparser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')\n\ntrain = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], date_parser=parser)\ntest = pd.read_csv('../input/test.csv')\nitems = pd.read_csv('../input/items.csv')\nitem_cats = pd.read_csv('../input/item_categories.csv')\nshops = pd.read_csv('../input/shops.csv')\n\nprint('train:', train.shape, 'test:', test.shape)\nprint('items:', items.shape, 'item_cats:', item_cats.shape, 'shops:', shops.shape)")

train.head()


test.head()


items.head()


item_cats.head()


shops.head()


print(train['date_block_num'].max())


print(train['item_cnt_day'].describe())


train['item_cnt_day'].nlargest(25).values


test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()
print('test only items:', len(test_only))


# ## Preprocessing


subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']
print(train.duplicated(subset=subset).value_counts())
train.drop_duplicates(subset=subset, inplace=True)



test_shops = test.shop_id.unique()
test_items = test.item_id.unique()
train = train[train.shop_id.isin(test_shops)]
train = train[train.item_id.isin(test_items)]

print('train:', train.shape)




from itertools import product

block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num','shop_id'])
shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id','item_id'])
all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')
print(len(all_combi), 34 * len(test_shops) * len(test_items))

# group by monthly
train_base = pd.merge(all_combi, train, on=['date_block_num','shop_id','item_id'], how='left')
train_base['item_cnt_day'].fillna(0, inplace=True)
train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])



train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day':['sum','count']})).reset_index()
train_monthly.columns = ['date_block_num','shop_id','item_id','item_cnt','item_order']
print(train_monthly[['item_cnt','item_order']].describe())

# trim count
train_monthly['item_cnt'].clip(0, 20, inplace=True)

train_monthly.head()





# pickup first category name
item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])
item_cats['item_group'] = pd.Categorical(item_grp).codes
#item_cats = item_cats.join(pd.get_dummies(item_grp, prefix='item_group', drop_first=True))
items = pd.merge(items, item_cats.loc[:,['item_category_id','item_group']], on=['item_category_id'], how='left')

item_grp.unique()



city = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(city).codes

city.unique()


# In[18]:


# By shop,item
grp = train_monthly.groupby(['shop_id', 'item_id'])
train_shop = grp.agg({'item_cnt':['mean','median','std'],'item_order':'mean'}).reset_index()
train_shop.columns = ['shop_id','item_id','cnt_mean_shop','cnt_med_shop','cnt_std_shop','order_mean_shop']
print(train_shop[['cnt_mean_shop','cnt_med_shop','cnt_std_shop']].describe())

train_shop.head()


# In[19]:


# By shop,item_group
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')
grp = train_cat_monthly.groupby(['shop_id', 'item_group'])
train_shop_cat = grp.agg({'item_cnt':['mean']}).reset_index()
train_shop_cat.columns = ['shop_id','item_group','cnt_mean_shop_cat']
print(train_shop_cat.loc[:,['cnt_mean_shop_cat']].describe())

train_shop_cat.head()


# ### Lags

# In[20]:


# By month,shop,item At previous
train_prev = train_monthly.copy()
train_prev['date_block_num'] = train_prev['date_block_num'] + 1
train_prev.columns = ['date_block_num','shop_id','item_id','cnt_prev','order_prev']

for i in [2,12]:
    train_prev_n = train_monthly.copy()
    train_prev_n['date_block_num'] = train_prev_n['date_block_num'] + i
    train_prev_n.columns = ['date_block_num','shop_id','item_id','cnt_prev' + str(i),'order_prev' + str(i)]
    train_prev = pd.merge(train_prev, train_prev_n, on=['date_block_num','shop_id','item_id'], how='left')

train_prev.head()


# In[21]:


# By month,shop,item_group At previous
grp = pd.merge(train_prev, items, on=['item_id'], how='left').groupby(['date_block_num','shop_id','item_group'])
train_cat_prev = grp['cnt_prev'].mean().reset_index()
train_cat_prev = train_cat_prev.rename(columns={'cnt_prev':'cnt_prev_cat'})
print(train_cat_prev.loc[:,['cnt_prev_cat']].describe())

train_cat_prev.head()


# ### Crosstab

# In[22]:


train_piv = train_monthly.pivot_table(index=['shop_id','item_id'], columns=['date_block_num'], values='item_cnt', aggfunc=np.sum, fill_value=0)
train_piv = train_piv.reset_index()
train_piv.head()


# In[23]:


# MACD At previous
col = np.arange(34)
pivT = train_piv[col].T
ema_s = pivT.ewm(span=4).mean().T
ema_m = pivT.ewm(span=12).mean().T
ema_l = pivT.ewm(span=26).mean().T
macd = ema_s - ema_l
sig = macd.ewm(span=9).mean()

ema_list = []
for c in col:
  sub_ema = pd.concat([train_piv.loc[:,['shop_id','item_id']],
      pd.DataFrame(ema_s.loc[:,c]).rename(columns={c:'cnt_ema_s_prev'}),
      pd.DataFrame(ema_m.loc[:,c]).rename(columns={c:'cnt_ema_m_prev'}),
      pd.DataFrame(ema_l.loc[:,c]).rename(columns={c:'cnt_ema_l_prev'}),
      pd.DataFrame(macd.loc[:,c]).rename(columns={c:'cnt_macd_prev'}),
      pd.DataFrame(sig.loc[:,c]).rename(columns={c:'cnt_sig_prev'})], axis=1)
  sub_ema['date_block_num'] = c + 1
  ema_list.append(sub_ema)
    
train_ema_prev = pd.concat(ema_list)
train_ema_prev.head()


# In[24]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

train_monthly.groupby(['date_block_num']).sum().reset_index()['item_cnt'].plot(ax=ax[0])
train_cat_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')
train_cat_monthly.pivot_table(index=['date_block_num'], columns=['item_group'], values='item_cnt', aggfunc=np.sum, fill_value=0).plot(ax=ax[1], legend=False)


# ### Item prices

# In[25]:


# Price mean by month,shop,item
train_price = train_grp['item_price'].mean().reset_index()
price = train_price[~train_price['item_price'].isnull()]

# last price by shop,item
last_price = price.drop_duplicates(subset=['shop_id', 'item_id'], keep='last').drop(['date_block_num'], axis=1)

# null price by shop,item
'''
mean_price = price.groupby(['item_id'])['item_price'].mean().reset_index()
result_price = pd.merge(test, mean_price, on=['item_id'], how='left').drop('ID', axis=1)
pred_price_set = result_price[result_price['item_price'].isnull()]
'''
uitem = price['item_id'].unique()
pred_price_set = test[~test['item_id'].isin(uitem)].drop('ID', axis=1)


# In[26]:


_ = '''
'''
if len(pred_price_set) > 0:
    train_price_set = pd.merge(price, items, on=['item_id'], how='inner')
    pred_price_set = pd.merge(pred_price_set, items, on=['item_id'], how='inner').drop(['item_name'], axis=1)
    reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=-1, max_depth=15, random_state=42)
    reg.fit(train_price_set[pred_price_set.columns], train_price_set['item_price'])
    pred_price_set['item_price'] = reg.predict(pred_price_set)

test_price = pd.concat([last_price, pred_price_set], join='inner')
test_price.head()


# ### Discount rate

# In[27]:


price_max = price.groupby(['item_id']).max()['item_price'].reset_index()
price_max.rename(columns={'item_price':'item_max_price'}, inplace=True)
price_max.head()


# In[28]:


train_price_a = pd.merge(price, price_max, on=['item_id'], how='left')
train_price_a['discount_rate'] = 1 - (train_price_a['item_price'] / train_price_a['item_max_price'])
train_price_a.drop('item_max_price', axis=1, inplace=True)
train_price_a.head()


# In[29]:


test_price_a = pd.merge(test_price, price_max, on=['item_id'], how='left')
test_price_a.loc[test_price_a['item_max_price'].isnull(), 'item_max_price'] = test_price_a['item_price']
test_price_a['discount_rate'] = 1 - (test_price_a['item_price'] / test_price_a['item_max_price'])
test_price_a.drop('item_max_price', axis=1, inplace=True)
test_price_a.head()


