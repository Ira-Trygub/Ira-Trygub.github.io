# Heart Attack Prediction


```python
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
import tensorflow
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import cross_val_score
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```


```python
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
```


```python
df = pd.read_csv("heart.csv")
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



 "previous_peak", "slp", 'vessels', "thall", "output"

About this dataset

age : Age of the patient

sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

vessels ca: number of major vessels (0-3)

chest_pain (cp) : Chest Pain type chest pain type

Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic

blood_pressure (rbp) : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

blood_sugar (fbs) : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

cardio (rest_ecg) : resting electrocardiographic results

Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
max_heart_rate (thalach) : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack

previous_peak (oldpeak)

thall - Thallium Stress Test


```python
df.columns = ["age", "sex", "chest_pain", 'blood_pressure',  'chol', 'blood_sugar',  "cardio", "max_heart_rate", 'exang', "previous_peak", "slp", 'vessels', "thall", "output"]
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>chest_pain</th>
      <th>blood_pressure</th>
      <th>chol</th>
      <th>blood_sugar</th>
      <th>cardio</th>
      <th>max_heart_rate</th>
      <th>exang</th>
      <th>previous_peak</th>
      <th>slp</th>
      <th>vessels</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.366337</td>
      <td>0.683168</td>
      <td>0.966997</td>
      <td>131.623762</td>
      <td>246.264026</td>
      <td>0.148515</td>
      <td>0.528053</td>
      <td>149.646865</td>
      <td>0.326733</td>
      <td>1.039604</td>
      <td>1.399340</td>
      <td>0.729373</td>
      <td>2.313531</td>
      <td>0.544554</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.082101</td>
      <td>0.466011</td>
      <td>1.032052</td>
      <td>17.538143</td>
      <td>51.830751</td>
      <td>0.356198</td>
      <td>0.525860</td>
      <td>22.905161</td>
      <td>0.469794</td>
      <td>1.161075</td>
      <td>0.616226</td>
      <td>1.022606</td>
      <td>0.612277</td>
      <td>0.498835</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>211.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>133.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>130.000000</td>
      <td>240.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>153.000000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>140.000000</td>
      <td>274.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>166.000000</td>
      <td>1.000000</td>
      <td>1.600000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>564.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>202.000000</td>
      <td>1.000000</td>
      <td>6.200000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             303 non-null    int64  
     1   sex             303 non-null    int64  
     2   chest_pain      303 non-null    int64  
     3   blood_pressure  303 non-null    int64  
     4   chol            303 non-null    int64  
     5   blood_sugar     303 non-null    int64  
     6   cardio          303 non-null    int64  
     7   max_heart_rate  303 non-null    int64  
     8   exang           303 non-null    int64  
     9   previous_peak   303 non-null    float64
     10  slp             303 non-null    int64  
     11  vessels         303 non-null    int64  
     12  thall           303 non-null    int64  
     13  output          303 non-null    int64  
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB
    

Let's check wether set has some gaps


```python
df.isnull().sum()
```




    age               0
    sex               0
    chest_pain        0
    blood_pressure    0
    chol              0
    blood_sugar       0
    cardio            0
    max_heart_rate    0
    exang             0
    previous_peak     0
    slp               0
    vessels           0
    thall             0
    output            0
    dtype: int64




```python
fig, ax = plt.subplots(figsize=(13,13))
sns.heatmap(df.corr(), ax = ax)
```




    <AxesSubplot:>




    
![png](output_10_1.png)
    



```python
def visual(col):
    #print("Visualization of {}".format(col))
    fig, axes=plt.subplots(1, 2 , figsize=(12,6))
    sns.set_context( 'paper', font_scale = 1.4)
    sns.histplot(ax=axes[0], x= df[col])
    sns.boxplot(ax=axes[1], x=df[col])
```


```python
for col in df[["age",  'blood_pressure',  'chol',  "max_heart_rate",  "previous_peak"]]:
    visual(col)
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



```python
def visual2(col):
    fig = plt.figure(figsize =( 8,5))
    sns.countplot(x=df[col], data=df) 
    plt.xlabel(col, fontsize=18)

```


```python
for col in df[[ "sex", "chest_pain", 'blood_sugar',  "cardio", 'exang', "slp", 'vessels', "thall", "output"]]:
    visual2(col)
```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    



    
![png](output_14_3.png)
    



    
![png](output_14_4.png)
    



    
![png](output_14_5.png)
    



    
![png](output_14_6.png)
    



    
![png](output_14_7.png)
    



    
![png](output_14_8.png)
    



```python
df_pie = pd.crosstab(columns=df["sex"],index=df["output"],normalize="columns")
df_pie.columns=["female", "male"]
df_pie.index=["",""]
df_pie.plot.pie(subplots=True, legend=False, autopct='%1.1f%%', figsize=(11, 6), colors= ["lime", "red"])
label = ["lower chance", "higher chance"]
plt.legend(label,  loc='best', bbox_to_anchor=(1,0.2))
plt.title("Chance of heart attack for Females and Males", fontsize=16)
```




    Text(0.5, 1.0, 'Chance of heart attack for Females and Males')




    
![png](output_15_1.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split( df.drop(['output'], axis = 1), df['output'], test_size=0.2, random_state=42)
```

<!-- importance = pd.DataFrame(f_classif(X_train, y_train)[0])
fig = plt.figure(figsize =( 14,8))
importance.plot.bar()
plt.xticks(( 0,1, 2, 3,4,5,6,7,8,9,10,11,12), ("age", "sex", "chest_pain", 'blood_pressure',  'chol', 'blood_sugar',  "cardio", "max_heart_rate", 'exang', "previous_peak", "slp", 'vessels', "thall"), rotation=70)
plt.xlabel('Importance pf features', fontsize=18) -->


```python
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train) # Skalierung wird berechnet
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train) # Skalierung wird angewandt
X_test = scaler.transform(X_test) # Skalierung wird angewandt

```

Function for training, testing and estimating models


```python
accuracy_set = pd.DataFrame()

```


```python
def trait_regressor(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = str(accuracy_score(y_test, y_pred))
    app =[(type(model).__name__), model.get_params(), acc]
    print(type(model).__name__ +' Accuracy:', acc)
    return (model, app)

```


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                              LogisticRegression())
 
lr_model= res_tuple[0]

accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=False)

```

    LogisticRegression Accuracy: 0.8524590163934426
    


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                            RandomForestClassifier())
rf_1 = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    RandomForestClassifier Accuracy: 0.8360655737704918
    


```python
# param_grid = {
#     'n_estimators': [100, 200, 500, 1000, 5000],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [1, 3, 4, 6, 10]
# }

# gb_2 = trait_regressor(X_train, y_train, X_test, y_test,
#                        GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, n_jobs = 6)
#                        )[0]
# print(gb_2.best_params_)
```


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                         RandomForestClassifier(max_depth= 1, max_features = 'log2', n_estimators= 200))
gb_def = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    RandomForestClassifier Accuracy: 0.8852459016393442
    


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                            DecisionTreeClassifier())
ftree_def  = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    DecisionTreeClassifier Accuracy: 0.819672131147541
    


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                         GradientBoostingClassifier())
gb_def = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    GradientBoostingClassifier Accuracy: 0.7868852459016393
    


```python
# param_grid = {
#     'n_estimators': [200, 500, 1000, 5000],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [1, 3, 4, 6, 10]
# }

# gb_2 = trait_regressor(X_train, y_train, X_test, y_test,
#                        GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = param_grid, n_jobs = 6)
#                        )[0]
# print(gb_2.best_params_)
```


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test,
                         GradientBoostingClassifier(max_depth= 4, max_features = 'log2', n_estimators= 200))
gb_def = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    GradientBoostingClassifier Accuracy: 0.8524590163934426
    


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test, SVC(kernel='linear', C=1, random_state=42))
svc = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    SVC Accuracy: 0.8688524590163934
    


```python
# param_grid = {
#     "C":np.arange(1,10,1),'gamma':[0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}

# svc_2 = trait_regressor(X_train, y_train, X_test, y_test,
#                        GridSearchCV(estimator = SVC(), param_grid = param_grid, n_jobs = 6)
#                        )[0]
# print(svc_2.best_params_)
```


```python
res_tuple = trait_regressor( X_train, y_train, X_test, y_test, SVC(kernel='linear', C=8, random_state=42, gamma=  0.01))
svc = res_tuple[0]
accuracy_set = accuracy_set.append([res_tuple[1]],  ignore_index=True)
```

    SVC Accuracy: 0.8852459016393442
    


```python
accuracy_set.columns = ["model", "parameters", "accuracy"] 
accuracy_set

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>parameters</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>{'C': 1.0, 'class_weight': None, 'dual': False...</td>
      <td>0.8524590163934426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>0.8360655737704918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'class_w...</td>
      <td>0.8852459016393442</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DecisionTreeClassifier</td>
      <td>{'ccp_alpha': 0.0, 'class_weight': None, 'crit...</td>
      <td>0.819672131147541</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GradientBoostingClassifier</td>
      <td>{'ccp_alpha': 0.0, 'criterion': 'friedman_mse'...</td>
      <td>0.7868852459016393</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GradientBoostingClassifier</td>
      <td>{'ccp_alpha': 0.0, 'criterion': 'friedman_mse'...</td>
      <td>0.8524590163934426</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVC</td>
      <td>{'C': 1, 'break_ties': False, 'cache_size': 20...</td>
      <td>0.8688524590163934</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SVC</td>
      <td>{'C': 8, 'break_ties': False, 'cache_size': 20...</td>
      <td>0.8852459016393442</td>
    </tr>
  </tbody>
</table>
</div>



# Let's try fing dome clusters 


```python
# k-means with some arbitrary k
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(df)
# kmeans.labels_
```




    KMeans(max_iter=50, n_clusters=4)




```python
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8,10]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df)
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)
```




    [<matplotlib.lines.Line2D at 0x1075b6fc640>]




    
![png](output_36_1.png)
    



```python
# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
```

    For n_clusters=2, the silhouette score is 0.3893747942796262
    For n_clusters=3, the silhouette score is 0.28738672830258105
    For n_clusters=4, the silhouette score is 0.2768729353128985
    For n_clusters=5, the silhouette score is 0.2781194558314485
    For n_clusters=6, the silhouette score is 0.27310621696379334
    For n_clusters=7, the silhouette score is 0.25736108416671466
    For n_clusters=8, the silhouette score is 0.23874504025021254
    


```python
# Final model with k=2
kmeans = KMeans(n_clusters=2, max_iter=50)
kmeans.fit(df)
```




    KMeans(max_iter=50, n_clusters=2)




```python
# assign the label
df['Cluster_Id'] = kmeans.labels_
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>chest_pain</th>
      <th>blood_pressure</th>
      <th>chol</th>
      <th>blood_sugar</th>
      <th>cardio</th>
      <th>max_heart_rate</th>
      <th>exang</th>
      <th>previous_peak</th>
      <th>slp</th>
      <th>vessels</th>
      <th>thall</th>
      <th>output</th>
      <th>Cluster_Id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pie = pd.crosstab(columns=df["Cluster_Id"],index=df["output"],normalize="columns")
df_pie.columns=["Cluster_Id 0", "Cluster_Id 1"]
df_pie.index=["",""]
df_pie.plot.pie(subplots=True, legend=False, autopct='%1.1f%%', figsize=(11, 6))
label = ["lower chance", "higher chance"]
plt.legend(label,  loc='best', bbox_to_anchor=(1,0.2))
plt.title("Visualizing of clustering", fontsize=16)
```




    Text(0.5, 1.0, 'Visualizing of clustering')




    
![png](output_40_1.png)
    



```python

```


```python
X_train.shape
```




    (242, 13)




```python
model_seq = Sequential()


model_seq.add(Dense(units=128,  input_shape = (242, 13),
                    kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-4),
                    activity_regularizer=regularizers.l2(1e-5),
                    #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
                    activation='relu'))  #kernel_regularizer

# layer = tf.keras.layers.Dropout(0.2) made Accuracy smaller

model_seq.add(Dense(units=64,  input_shape = (242, 13),
#                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                     activity_regularizer=regularizers.l2(1e-5),
                    #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
                    activation='relu'))  #kernel_regularizer

# model_seq.add(Dense(units=32,  input_shape = (242, 13),
# #                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
# #                     activity_regularizer=regularizers.l2(1e-5),
#                     #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
#                     activation='relu'))

# model_seq.add(Dense(units=16,  input_shape = (768,8),
#                     activation='relu'))



model_seq.add(Dense(units=1, activation='sigmoid'))
```


```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```


```python
model_seq.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
```


```python
history = model_seq.fit(X_train, y_train, epochs=80, batch_size=32,  validation_split=0.33)
```

    Epoch 1/80
    WARNING:tensorflow:Model was constructed with shape (None, 242, 13) for input KerasTensor(type_spec=TensorSpec(shape=(None, 242, 13), dtype=tf.float32, name='dense_17_input'), name='dense_17_input', description="created by layer 'dense_17_input'"), but it was called on an input with incompatible shape (None, 13).
    WARNING:tensorflow:Model was constructed with shape (None, 242, 13) for input KerasTensor(type_spec=TensorSpec(shape=(None, 242, 13), dtype=tf.float32, name='dense_17_input'), name='dense_17_input', description="created by layer 'dense_17_input'"), but it was called on an input with incompatible shape (None, 13).
    1/6 [====>.........................] - ETA: 8s - loss: 0.6938 - accuracy: 0.6250WARNING:tensorflow:Model was constructed with shape (None, 242, 13) for input KerasTensor(type_spec=TensorSpec(shape=(None, 242, 13), dtype=tf.float32, name='dense_17_input'), name='dense_17_input', description="created by layer 'dense_17_input'"), but it was called on an input with incompatible shape (None, 13).
    6/6 [==============================] - 2s 87ms/step - loss: 0.7394 - accuracy: 0.4198 - val_loss: 0.7457 - val_accuracy: 0.4625
    Epoch 2/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.7251 - accuracy: 0.4444 - val_loss: 0.7343 - val_accuracy: 0.4750
    Epoch 3/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.7136 - accuracy: 0.4568 - val_loss: 0.7224 - val_accuracy: 0.4875
    Epoch 4/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.7010 - accuracy: 0.5000 - val_loss: 0.7117 - val_accuracy: 0.5125
    Epoch 5/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.6911 - accuracy: 0.5309 - val_loss: 0.7010 - val_accuracy: 0.5375
    Epoch 6/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.6801 - accuracy: 0.5370 - val_loss: 0.6914 - val_accuracy: 0.5500
    Epoch 7/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.6706 - accuracy: 0.5741 - val_loss: 0.6824 - val_accuracy: 0.5750
    Epoch 8/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.6617 - accuracy: 0.6049 - val_loss: 0.6733 - val_accuracy: 0.5875
    Epoch 9/80
    6/6 [==============================] - 0s 25ms/step - loss: 0.6522 - accuracy: 0.6481 - val_loss: 0.6648 - val_accuracy: 0.5875
    Epoch 10/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.6432 - accuracy: 0.6667 - val_loss: 0.6565 - val_accuracy: 0.6125
    Epoch 11/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.6351 - accuracy: 0.6728 - val_loss: 0.6493 - val_accuracy: 0.6250
    Epoch 12/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.6270 - accuracy: 0.6852 - val_loss: 0.6418 - val_accuracy: 0.6625
    Epoch 13/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.6190 - accuracy: 0.6790 - val_loss: 0.6342 - val_accuracy: 0.6750
    Epoch 14/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.6112 - accuracy: 0.6975 - val_loss: 0.6269 - val_accuracy: 0.6875
    Epoch 15/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.6036 - accuracy: 0.7222 - val_loss: 0.6198 - val_accuracy: 0.7000
    Epoch 16/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.5964 - accuracy: 0.7346 - val_loss: 0.6132 - val_accuracy: 0.6875
    Epoch 17/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.5891 - accuracy: 0.7407 - val_loss: 0.6068 - val_accuracy: 0.7000
    Epoch 18/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.5824 - accuracy: 0.7531 - val_loss: 0.6011 - val_accuracy: 0.7250
    Epoch 19/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.5757 - accuracy: 0.7531 - val_loss: 0.5952 - val_accuracy: 0.7375
    Epoch 20/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.5692 - accuracy: 0.7531 - val_loss: 0.5888 - val_accuracy: 0.7625
    Epoch 21/80
    6/6 [==============================] - 0s 14ms/step - loss: 0.5624 - accuracy: 0.7531 - val_loss: 0.5826 - val_accuracy: 0.7625
    Epoch 22/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.5562 - accuracy: 0.7593 - val_loss: 0.5767 - val_accuracy: 0.7625
    Epoch 23/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.5499 - accuracy: 0.7593 - val_loss: 0.5708 - val_accuracy: 0.7750
    Epoch 24/80
    6/6 [==============================] - 0s 9ms/step - loss: 0.5438 - accuracy: 0.7654 - val_loss: 0.5649 - val_accuracy: 0.7750
    Epoch 25/80
    6/6 [==============================] - 0s 9ms/step - loss: 0.5376 - accuracy: 0.7716 - val_loss: 0.5597 - val_accuracy: 0.7750
    Epoch 26/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.5323 - accuracy: 0.7778 - val_loss: 0.5542 - val_accuracy: 0.7750
    Epoch 27/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.5263 - accuracy: 0.7778 - val_loss: 0.5491 - val_accuracy: 0.7875
    Epoch 28/80
    6/6 [==============================] - 0s 14ms/step - loss: 0.5207 - accuracy: 0.7840 - val_loss: 0.5444 - val_accuracy: 0.7875
    Epoch 29/80
    6/6 [==============================] - 0s 14ms/step - loss: 0.5155 - accuracy: 0.7901 - val_loss: 0.5400 - val_accuracy: 0.7875
    Epoch 30/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.5103 - accuracy: 0.8025 - val_loss: 0.5350 - val_accuracy: 0.7875
    Epoch 31/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.5049 - accuracy: 0.8086 - val_loss: 0.5301 - val_accuracy: 0.7875
    Epoch 32/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4994 - accuracy: 0.8086 - val_loss: 0.5256 - val_accuracy: 0.7750
    Epoch 33/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4943 - accuracy: 0.8086 - val_loss: 0.5214 - val_accuracy: 0.7750
    Epoch 34/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4894 - accuracy: 0.8086 - val_loss: 0.5169 - val_accuracy: 0.7750
    Epoch 35/80
    6/6 [==============================] - 0s 14ms/step - loss: 0.4849 - accuracy: 0.8086 - val_loss: 0.5127 - val_accuracy: 0.7750
    Epoch 36/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4802 - accuracy: 0.8148 - val_loss: 0.5087 - val_accuracy: 0.7750
    Epoch 37/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4758 - accuracy: 0.8210 - val_loss: 0.5052 - val_accuracy: 0.7750
    Epoch 38/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4712 - accuracy: 0.8210 - val_loss: 0.5014 - val_accuracy: 0.7750
    Epoch 39/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4670 - accuracy: 0.8210 - val_loss: 0.4976 - val_accuracy: 0.7750
    Epoch 40/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4627 - accuracy: 0.8210 - val_loss: 0.4935 - val_accuracy: 0.7750
    Epoch 41/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4584 - accuracy: 0.8210 - val_loss: 0.4895 - val_accuracy: 0.7875
    Epoch 42/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4541 - accuracy: 0.8210 - val_loss: 0.4856 - val_accuracy: 0.7875
    Epoch 43/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4499 - accuracy: 0.8272 - val_loss: 0.4821 - val_accuracy: 0.7875
    Epoch 44/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4459 - accuracy: 0.8272 - val_loss: 0.4785 - val_accuracy: 0.7875
    Epoch 45/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4422 - accuracy: 0.8272 - val_loss: 0.4749 - val_accuracy: 0.7875
    Epoch 46/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4382 - accuracy: 0.8272 - val_loss: 0.4713 - val_accuracy: 0.7875
    Epoch 47/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4343 - accuracy: 0.8272 - val_loss: 0.4676 - val_accuracy: 0.7875
    Epoch 48/80
    6/6 [==============================] - 0s 13ms/step - loss: 0.4307 - accuracy: 0.8272 - val_loss: 0.4643 - val_accuracy: 0.7875
    Epoch 49/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4272 - accuracy: 0.8272 - val_loss: 0.4613 - val_accuracy: 0.7875
    Epoch 50/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4238 - accuracy: 0.8272 - val_loss: 0.4585 - val_accuracy: 0.7875
    Epoch 51/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.4206 - accuracy: 0.8333 - val_loss: 0.4554 - val_accuracy: 0.7875
    Epoch 52/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4174 - accuracy: 0.8333 - val_loss: 0.4523 - val_accuracy: 0.7875
    Epoch 53/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.4141 - accuracy: 0.8333 - val_loss: 0.4492 - val_accuracy: 0.7875
    Epoch 54/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4108 - accuracy: 0.8333 - val_loss: 0.4461 - val_accuracy: 0.7875
    Epoch 55/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.4078 - accuracy: 0.8457 - val_loss: 0.4436 - val_accuracy: 0.7875
    Epoch 56/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.4050 - accuracy: 0.8519 - val_loss: 0.4413 - val_accuracy: 0.7875
    Epoch 57/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.4020 - accuracy: 0.8457 - val_loss: 0.4394 - val_accuracy: 0.8000
    Epoch 58/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3998 - accuracy: 0.8395 - val_loss: 0.4373 - val_accuracy: 0.8000
    Epoch 59/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3973 - accuracy: 0.8395 - val_loss: 0.4351 - val_accuracy: 0.8000
    Epoch 60/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3945 - accuracy: 0.8395 - val_loss: 0.4325 - val_accuracy: 0.8000
    Epoch 61/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3917 - accuracy: 0.8395 - val_loss: 0.4295 - val_accuracy: 0.8000
    Epoch 62/80
    6/6 [==============================] - 0s 14ms/step - loss: 0.3889 - accuracy: 0.8457 - val_loss: 0.4269 - val_accuracy: 0.8000
    Epoch 63/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3865 - accuracy: 0.8519 - val_loss: 0.4245 - val_accuracy: 0.8000
    Epoch 64/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3837 - accuracy: 0.8580 - val_loss: 0.4223 - val_accuracy: 0.8000
    Epoch 65/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3814 - accuracy: 0.8580 - val_loss: 0.4205 - val_accuracy: 0.8000
    Epoch 66/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3790 - accuracy: 0.8580 - val_loss: 0.4185 - val_accuracy: 0.8000
    Epoch 67/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3766 - accuracy: 0.8642 - val_loss: 0.4163 - val_accuracy: 0.8000
    Epoch 68/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3742 - accuracy: 0.8642 - val_loss: 0.4145 - val_accuracy: 0.8000
    Epoch 69/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3719 - accuracy: 0.8642 - val_loss: 0.4125 - val_accuracy: 0.8000
    Epoch 70/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3697 - accuracy: 0.8642 - val_loss: 0.4107 - val_accuracy: 0.8125
    Epoch 71/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3673 - accuracy: 0.8642 - val_loss: 0.4090 - val_accuracy: 0.8125
    Epoch 72/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3655 - accuracy: 0.8642 - val_loss: 0.4074 - val_accuracy: 0.8125
    Epoch 73/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3632 - accuracy: 0.8642 - val_loss: 0.4059 - val_accuracy: 0.8125
    Epoch 74/80
    6/6 [==============================] - 0s 9ms/step - loss: 0.3614 - accuracy: 0.8642 - val_loss: 0.4051 - val_accuracy: 0.8125
    Epoch 75/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3598 - accuracy: 0.8704 - val_loss: 0.4038 - val_accuracy: 0.8125
    Epoch 76/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3580 - accuracy: 0.8704 - val_loss: 0.4027 - val_accuracy: 0.8125
    Epoch 77/80
    6/6 [==============================] - 0s 12ms/step - loss: 0.3567 - accuracy: 0.8704 - val_loss: 0.4022 - val_accuracy: 0.8000
    Epoch 78/80
    6/6 [==============================] - 0s 9ms/step - loss: 0.3551 - accuracy: 0.8704 - val_loss: 0.4013 - val_accuracy: 0.8000
    Epoch 79/80
    6/6 [==============================] - 0s 11ms/step - loss: 0.3535 - accuracy: 0.8704 - val_loss: 0.4001 - val_accuracy: 0.8000
    Epoch 80/80
    6/6 [==============================] - 0s 10ms/step - loss: 0.3518 - accuracy: 0.8642 - val_loss: 0.3988 - val_accuracy: 0.8000
    


```python
plt.style.use(["seaborn-muted"])
fig = plt.figure(figsize = (12, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
```


    
![png](output_47_0.png)
    



```python
plt.style.use(["seaborn-muted"])
fig = plt.figure(figsize = (12, 8))
plt.plot(history.history["accuracy"])

plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
```


    
![png](output_48_0.png)
    



```python
loss_and_metrics = model_seq.evaluate( X_test, y_test, batch_size=32)
```

    2/2 [==============================] - 0s 4ms/step - loss: 0.3742 - accuracy: 0.8525
    


```python

```
