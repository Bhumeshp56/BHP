import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('/Users/bhumeshpanchal/Documents/Bhumesh/Mtech/Mtech_Thesis/ML_codebasics/ML_project/bengaluru_house_prices.csv')

print(df1.groupby('area_type')['area_type'].agg('count'))

df2 = df1.drop(['availability','society','balcony','area_type'],axis='columns')
print(df2.shape)

df3 = df2.dropna()

print(df3.isnull().sum())
print(df3.shape)
# print(df3['size'].unique())

#df3['bhk']=df3['size'].apply    this will directly convert the column of the dataset

df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3['bhk'])

# print(df3.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# df3=df3[~df3['total_sqft'].apply(is_float)].head()

#to convert the range of the values into the single values and also remove the values which having the different units

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4 =df3.copy()

df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)

print(df4.head())
print(df4.loc[30])

df5=df4.copy()

df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.shape)
print(len(df5.location.unique()))

df5.location= df5.location.apply(lambda x: x.strip())

location_stats= df5.groupby('location')['location'].agg('count').sort_values(ascending=True)

# print(location_stats)
location_stats_less_than_10=location_stats[location_stats<=10]
# print(len(location_stats_less_than_10))
# print(location_stats_less_than_10)

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

print(len(df5.location.unique()))

#for outliars

df6=df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

print(df6.price_per_sqft.describe())

def remove_pps_outliars(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m= np.mean(subdf.price_per_sqft)
        st= np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

df7 = remove_pps_outliars(df6)
print(df7.shape)

#here we are checking for the same area same location but for the different bhk the price of the home is nearly same or higher than that of the higher bhk

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
    
# plot_scatter_chart(df7,"Rajaji Nagar")

#function for to remove the above case

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
print(df8.shape)
# plot_scatter_chart(df8,"Hebbal")

#to check the distribution of bhk

plt.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
# plt.show()

#Now we'll go for the bathroom data
#generally 2 bhk having 2 bathroom. let say in good case 2 bhk having max 3 bathroom other than this we'll consider this as an outliar
#so we'll first check how it's histogram shows

# print(df8[df8.bath>df8.bhk+2])

df9 = df8[df8.bath<df8.bhk+2]
print(df9)

df10= df9.drop(['size','price_per_sqft'],axis='columns')

print(df10.shape)

#Tut_5 building machine learning model

#As the machine learning model can not read the text columns we required to make it numeric column so we will use one hot encoding
#drop is used in dummies data to avoid data trapping
dummie=pd.get_dummies(df10.location)

df11=pd.concat([df10,dummie.drop('other',axis='columns')],axis='columns')

df12= df11.drop('location',axis='columns')

print(df12.shape)

X=df12.drop('price',axis='columns')
y=df12.price

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=10)


#simple linear regression model is taken


from sklearn.linear_model import LinearRegression

lr_clf=LinearRegression()

lr_clf.fit(x_train,y_train)
score=lr_clf.score(x_test,y_test)
print(score)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv= ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score =cross_val_score(LinearRegression(),X,y,cv=cv)

print(np.mean(cross_val_score))

#now to find the other best model grid search cv is used

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
            'copy_X' : [True, False],
                'fit_intercept' : [True, False],
                'n_jobs' : [1,2,3],
                'positive' : [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(find_best_model_using_gridsearchcv(X,y))

#prdict price function to predict the price of the home


def predict_price(location,sqft,bath,bhk):

    loc_index = np.where(X.columns == location)[0][0]   #244 types of location that's why this is used

    x= np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]    

# print(predict_price('1st Phase JP Nagar',1000,2,2))

import pickle 

with open('/Users/bhumeshpanchal/Documents/Bhumesh/Mtech/Mtech_Thesis/ML_codebasics/ML_project/benglore_home_prices_model','wb') as f:
    pickle.dump(lr_clf,f)

import json

columns = {
    'data_columns':[col.lower() for col in X.columns]
}

with open('/Users/bhumeshpanchal/Documents/Bhumesh/Mtech/Mtech_Thesis/ML_codebasics/ML_project/columns.json','w') as f:
    f.write(json.dumps(columns))


#python flask server for backend for ui application


