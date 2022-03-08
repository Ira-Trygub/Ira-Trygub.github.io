---
layout: post
title:  "Diabetes Prediction"
date:   2021-04-07 18:12:12 +0100
categories: fp
---
Praktikum "Data Science"
  
  SoSe 2021
  
  Aufgabe 03
  
  Iryna Trygub
  
  Nico Fritz

# Data Analysis for Diabetes Prediction


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
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import cross_val_score
from functools import partial
```


```python
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
```


```python
df = pd.read_csv("diabetes.csv")
df.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize = (13,13))
sns.heatmap(df.corr(), annot = True)
```




    <AxesSubplot:>




    
![png](output_5_1.png)
    



```python
sns.jointplot(x =df['Glucose'], y =df['BMI'], hue=df['Outcome'] )
```




    <seaborn.axisgrid.JointGrid at 0x2b85a9e15b0>




    
![png](output_6_1.png)
    



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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
sns.countplot(x=df['Outcome'], data=df)  
perc_diab = df['Outcome'].sum() / df['Outcome'].count() * 100
print(f'{perc_diab}% of the people tested have diabetes')
```

    34.89583333333333% of the people tested have diabetes
    


    
![png](output_9_1.png)
    


The target variable values are well balanced.


```python
def visual(col):
    #print("Visualization of {}".format(col))
    fig, axes=plt.subplots(1, 2 , figsize=(12,6))
    sns.set_context( 'paper', font_scale = 1.4)
    sns.histplot(ax=axes[0], x= df[col])
    sns.boxplot(ax=axes[1], x=df[col])
```


```python
for col in df.drop('Outcome', axis = 1):
    visual(col)
```


    
![png](/img2/output_12_0.png)
    



    
![png](/img2/output_12_1.png)
    



    
![png](/img2/output_12_2.png)
    



    
![png](/img2/output_12_3.png)
    



    
![png](/img2/output_12_4.png)
    



    
![png](/img2/output_12_5.png)
    



    
![png](/img2/output_12_6.png)
    



    
![png](/img2/output_12_7.png)
    


There are a lot of unreal 0-values in such columns as "BMI", "Glucose", "BloodPressure", "SkinThickness", "Insulin"

"BMI" corrilates with "SkinThickness"


```python

```


```python
sns.jointplot(x =df['BMI'], y =df["SkinThickness"], kind = "reg" )
```




    <seaborn.axisgrid.JointGrid at 0x2b86ff4da60>




    
![png](/img2/output_16_1.png)
    


The line of regression between "BMI" and "SkinThickness" is deviated because of big amount of nulls in  "SkinThickness".



```python
for index, row in df.iterrows():
    if row["BMI"]==0 and row["SkinThickness"]==0 or row["Insulin"]==0 and row["Glucose"]==0:
        row[['Glucose', 'BloodPressure', 'SkinThickness',  'Insulin', 'BMI']] = row[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
        row.fillna(row.median(), inplace=True)
      
```


```python
A = np.vstack([df["BMI"][df["SkinThickness"]!=0], np.ones(len(df["BMI"][df["SkinThickness"]!=0]))]).T

```


```python
m, c = np.linalg.lstsq(A, df["SkinThickness"][df["SkinThickness"]!=0], rcond=None)[0]
print(m, c)
```

    0.9267600758180833 -1.2203281928193395
    


```python
df["SkinThickness"][df["SkinThickness"] == 0] =  df["BMI"][df["SkinThickness"] == 0] *m +c
```

    <ipython-input-3939-56c0c389ea76>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["SkinThickness"][df["SkinThickness"] == 0] =  df["BMI"][df["SkinThickness"] == 0] *m +c
    


```python
sns.jointplot(x =df['BMI'], y =df["SkinThickness"], kind = "reg" )
```




    <seaborn.axisgrid.JointGrid at 0x2b87a4a20d0>




    
![png](/img2/output_22_1.png)
    


Let's extract function to use it for other columns:


```python
def fillnulls(col1,*args):
    l = len(df[col1][df[col1] !=0 ])
    A = df[col1][df[col1] != 0]
    for col in args:
        A = np.vstack([A, df[col][df[col1] != 0]])
    A = np.delete(A, (0), axis=0)
    A = np.vstack([A,  np.ones(len(df[col1][df[col1] !=0 ]))]).T
    regr_list = np.linalg.lstsq(A, df[col1][df[col1]!=0], rcond=None)[0]
    sum = 0
    for i, col in enumerate(args):
        sum = sum + df[col][df[col1] == 0] * regr_list[i]  
    df[col1][df[col1] == 0] = sum + regr_list[-1] 
  
```

As we can see, the coefficient c can take on negative values. Since this is possible only for small values of x, in this case we replace the result of calculations with the values of the lower quantile.


```python
def check_negative( col, row):
    checked = row[col]
    if checked <= 0:
        res = df[col].quantile(0.25)
    else:
        res = checked
    return res
    
```


```python
df["SkinThickness"] = df.apply(partial(check_negative, "SkinThickness" ), axis = 1)
```

Let's calculate the Insulin value from Glucose, BloodPressure and Glucose values from BMI and Age, BMI from BloodPressure and Glucose on the same way.


```python
fillnulls("Insulin", "Glucose" )
df["Insulin"] = df.apply(partial(check_negative, "Insulin" ), axis = 1)

```

    <ipython-input-3941-8802713345ef>:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[col1][df[col1] == 0] = sum + regr_list[-1]
    


```python
fillnulls("BloodPressure", "BMI", "Age" )
df["BloodPressure"] = df.apply(partial(check_negative, "BloodPressure" ), axis = 1)

```

    <ipython-input-3941-8802713345ef>:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[col1][df[col1] == 0] = sum + regr_list[-1]
    


```python
sns.jointplot(x =df["Glucose"], y =df["Insulin"], kind = "reg" )
```




    <seaborn.axisgrid.JointGrid at 0x2b866580550>




    
![png](/img2/output_31_1.png)
    



```python
fillnulls("Glucose", "BMI", "Age" )
df["Glucose"] = df.apply(partial(check_negative, "Glucose" ), axis = 1)

```

    <ipython-input-3941-8802713345ef>:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[col1][df[col1] == 0] = sum + regr_list[-1]
    


```python
fillnulls("BMI", "BloodPressure", "Glucose" )
df["BMI"] =df.apply(partial(check_negative, "BMI" ), axis = 1)

```

    <ipython-input-3941-8802713345ef>:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df[col1][df[col1] == 0] = sum + regr_list[-1]
    


```python
for col in df.drop('Outcome', axis = 1):
    visual(col)
```


    
![png](/img2/output_34_0.png)
    



    
![png](/img2/output_34_1.png)
    



    
![png](/img2/output_34_2.png)
    



    
![png](/img2/output_34_3.png)
    



    
![png](/img2/output_34_4.png)
    



    
![png](/img2/output_34_5.png)
    



    
![png](/img2/output_34_6.png)
    



    
![png](/img2/output_34_7.png)
    



```python
df["Insulin"].sort_values()
```




    146     12.255936
    537     12.255936
    445     14.000000
    617     15.000000
    760     16.000000
              ...    
    409    579.000000
    584    600.000000
    247    680.000000
    228    744.000000
    13     846.000000
    Name: Insulin, Length: 768, dtype: float64




```python
df.shape
```




    (768, 9)




```python
X_train, X_test, y_train, y_test = train_test_split( df.drop(['Outcome'], axis = 1), df['Outcome'], test_size=0.2, random_state=42)
```


```python

```


```python
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train) # Skalierung wird berechnet
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train) # Skalierung wird angewandt
X_test = scaler.transform(X_test) # Skalierung wird angewandt

```


```python
X_train = X_train.reshape((614, 8))
```


```python
print('Dimensions of Array', X_train.ndim)
print('Shape of Array', X_train.shape)
print('Size of the first dimension', len(X_train))
print('Total number of elements', X_train.size)
```

    Dimensions of Array 2
    Shape of Array (614, 8)
    Size of the first dimension 614
    Total number of elements 4912
    


```python
y_train = y_train.values.reshape((-1,1))
```


```python
print('Dimensions of Array', y_train.ndim)
print('Shape of Array', y_train.shape)
print('Size of the first dimension', len(y_train))
print('Total number of elements', y_train.size)
```

    Dimensions of Array 2
    Shape of Array (614, 1)
    Size of the first dimension 614
    Total number of elements 614
    


```python

```

## SWM


```python
svc = SVC(kernel="linear", probability=True, random_state=0)
```


```python
model_svc = svc.fit(X_train, y_train)
probs= model_svc.predict(X_test)
print('Accuracy:'+str(accuracy_score(y_test, probs)))
```

    Accuracy:0.7532467532467533
    

    c:\users\trigu\appdata\local\programs\python\python39\lib\site-packages\sklearn\utils\validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    

## Firstly wir build a simple Sequential model


```python
model_seq = Sequential()


model_seq.add(Dense(units=128,  input_shape = (768,8),
                    kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-4),
                    activity_regularizer=regularizers.l2(1e-5),
                    #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
                    activation='relu'))  #kernel_regularizer

# layer = tf.keras.layers.Dropout(0.2) made Accuracy smaller

model_seq.add(Dense(units=64,  input_shape = (768,8),
#                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                     activity_regularizer=regularizers.l2(1e-5),
                    #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
                    activation='relu'))  #kernel_regularizer

model_seq.add(Dense(units=32,  input_shape = (768,8),
#                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                     activity_regularizer=regularizers.l2(1e-5),
                    #bias_regularizer=regularizers.l2(1e-4),  bias regulizer made Accuracy plot more noisy
                    activation='relu'))

# model_seq.add(Dense(units=16,  input_shape = (768,8),
#                     activation='relu'))



model_seq.add(Dense(units=1, activation='sigmoid'))
```

Let's add decay of the learning rate


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

cv = KFold(n_splits = 10, shuffle=True, random_state=42)
splits = cv.split(X_train)
for train_fold_index, predict_fold_index in splits:
    X_fold_train = X_train[train_fold_index]
    X_fold_predict = X_train[predict_fold_index]
    y_fold_train = y_train[train_fold_index]
    history = model_seq.fit(X_fold_train, y_fold_train, epochs=10, batch_size=128,  validation_split=0.33)


```python
history = model_seq.fit(X_train, y_train, epochs=30, batch_size=256,  validation_split=0.33)
```

    Epoch 1/30
    WARNING:tensorflow:Model was constructed with shape (None, 768, 8) for input KerasTensor(type_spec=TensorSpec(shape=(None, 768, 8), dtype=tf.float32, name='dense_396_input'), name='dense_396_input', description="created by layer 'dense_396_input'"), but it was called on an input with incompatible shape (None, 8).
    WARNING:tensorflow:Model was constructed with shape (None, 768, 8) for input KerasTensor(type_spec=TensorSpec(shape=(None, 768, 8), dtype=tf.float32, name='dense_396_input'), name='dense_396_input', description="created by layer 'dense_396_input'"), but it was called on an input with incompatible shape (None, 8).
    1/2 [==============>...............] - ETA: 1s - loss: 0.6766 - accuracy: 0.6367WARNING:tensorflow:Model was constructed with shape (None, 768, 8) for input KerasTensor(type_spec=TensorSpec(shape=(None, 768, 8), dtype=tf.float32, name='dense_396_input'), name='dense_396_input', description="created by layer 'dense_396_input'"), but it was called on an input with incompatible shape (None, 8).
    2/2 [==============================] - 2s 340ms/step - loss: 0.6766 - accuracy: 0.6423 - val_loss: 0.6728 - val_accuracy: 0.6847
    Epoch 2/30
    2/2 [==============================] - 0s 34ms/step - loss: 0.6735 - accuracy: 0.6618 - val_loss: 0.6704 - val_accuracy: 0.6847
    Epoch 3/30
    2/2 [==============================] - 0s 42ms/step - loss: 0.6704 - accuracy: 0.6813 - val_loss: 0.6679 - val_accuracy: 0.6946
    Epoch 4/30
    2/2 [==============================] - 0s 35ms/step - loss: 0.6673 - accuracy: 0.6910 - val_loss: 0.6655 - val_accuracy: 0.7143
    Epoch 5/30
    2/2 [==============================] - 0s 34ms/step - loss: 0.6644 - accuracy: 0.6910 - val_loss: 0.6631 - val_accuracy: 0.7241
    Epoch 6/30
    2/2 [==============================] - 0s 37ms/step - loss: 0.6614 - accuracy: 0.6910 - val_loss: 0.6607 - val_accuracy: 0.7241
    Epoch 7/30
    2/2 [==============================] - 0s 190ms/step - loss: 0.6585 - accuracy: 0.6959 - val_loss: 0.6583 - val_accuracy: 0.7241
    Epoch 8/30
    2/2 [==============================] - 0s 36ms/step - loss: 0.6556 - accuracy: 0.7007 - val_loss: 0.6560 - val_accuracy: 0.7241
    Epoch 9/30
    2/2 [==============================] - 0s 41ms/step - loss: 0.6525 - accuracy: 0.7080 - val_loss: 0.6536 - val_accuracy: 0.7241
    Epoch 10/30
    2/2 [==============================] - 0s 33ms/step - loss: 0.6496 - accuracy: 0.7129 - val_loss: 0.6513 - val_accuracy: 0.7241
    Epoch 11/30
    2/2 [==============================] - 0s 43ms/step - loss: 0.6469 - accuracy: 0.7178 - val_loss: 0.6489 - val_accuracy: 0.7192
    Epoch 12/30
    2/2 [==============================] - 0s 37ms/step - loss: 0.6440 - accuracy: 0.7251 - val_loss: 0.6466 - val_accuracy: 0.7192
    Epoch 13/30
    2/2 [==============================] - 0s 34ms/step - loss: 0.6410 - accuracy: 0.7324 - val_loss: 0.6442 - val_accuracy: 0.7192
    Epoch 14/30
    2/2 [==============================] - 0s 40ms/step - loss: 0.6382 - accuracy: 0.7348 - val_loss: 0.6419 - val_accuracy: 0.7192
    Epoch 15/30
    2/2 [==============================] - 0s 29ms/step - loss: 0.6353 - accuracy: 0.7397 - val_loss: 0.6395 - val_accuracy: 0.7291
    Epoch 16/30
    2/2 [==============================] - 0s 35ms/step - loss: 0.6326 - accuracy: 0.7445 - val_loss: 0.6372 - val_accuracy: 0.7291
    Epoch 17/30
    2/2 [==============================] - 0s 29ms/step - loss: 0.6297 - accuracy: 0.7470 - val_loss: 0.6348 - val_accuracy: 0.7291
    Epoch 18/30
    2/2 [==============================] - 0s 33ms/step - loss: 0.6268 - accuracy: 0.7543 - val_loss: 0.6325 - val_accuracy: 0.7340
    Epoch 19/30
    2/2 [==============================] - 0s 31ms/step - loss: 0.6240 - accuracy: 0.7567 - val_loss: 0.6301 - val_accuracy: 0.7389
    Epoch 20/30
    2/2 [==============================] - 0s 33ms/step - loss: 0.6213 - accuracy: 0.7567 - val_loss: 0.6277 - val_accuracy: 0.7389
    Epoch 21/30
    2/2 [==============================] - 0s 31ms/step - loss: 0.6183 - accuracy: 0.7616 - val_loss: 0.6253 - val_accuracy: 0.7340
    Epoch 22/30
    2/2 [==============================] - 0s 34ms/step - loss: 0.6155 - accuracy: 0.7616 - val_loss: 0.6229 - val_accuracy: 0.7340
    Epoch 23/30
    2/2 [==============================] - 0s 32ms/step - loss: 0.6127 - accuracy: 0.7591 - val_loss: 0.6205 - val_accuracy: 0.7389
    Epoch 24/30
    2/2 [==============================] - 0s 33ms/step - loss: 0.6097 - accuracy: 0.7664 - val_loss: 0.6181 - val_accuracy: 0.7389
    Epoch 25/30
    2/2 [==============================] - 0s 38ms/step - loss: 0.6069 - accuracy: 0.7713 - val_loss: 0.6157 - val_accuracy: 0.7389
    Epoch 26/30
    2/2 [==============================] - 0s 30ms/step - loss: 0.6041 - accuracy: 0.7713 - val_loss: 0.6132 - val_accuracy: 0.7340
    Epoch 27/30
    2/2 [==============================] - 0s 32ms/step - loss: 0.6011 - accuracy: 0.7713 - val_loss: 0.6108 - val_accuracy: 0.7389
    Epoch 28/30
    2/2 [==============================] - 0s 34ms/step - loss: 0.5983 - accuracy: 0.7713 - val_loss: 0.6084 - val_accuracy: 0.7389
    Epoch 29/30
    2/2 [==============================] - 0s 31ms/step - loss: 0.5953 - accuracy: 0.7737 - val_loss: 0.6059 - val_accuracy: 0.7389
    Epoch 30/30
    2/2 [==============================] - 0s 37ms/step - loss: 0.5924 - accuracy: 0.7762 - val_loss: 0.6035 - val_accuracy: 0.7389
    


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


    
![png](/img2/output_55_0.png)
    



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


    
![png](/img2/output_56_0.png)
    



```python
loss_and_metrics = model_seq.evaluate( X_test, y_test, batch_size=32)
```

    WARNING:tensorflow:Model was constructed with shape (None, 768, 8) for input KerasTensor(type_spec=TensorSpec(shape=(None, 768, 8), dtype=tf.float32, name='dense_396_input'), name='dense_396_input', description="created by layer 'dense_396_input'"), but it was called on an input with incompatible shape (None, 8).
    5/5 [==============================] - 0s 3ms/step - loss: 0.6021 - accuracy: 0.7792
    

4/4 [==============================] - 0s 2ms/step - loss: 0.4931 - accuracy: 0.7315

4/4 [==============================] - 0s 2ms/step - loss: 0.5648 - accuracy: 0.7407

5/5 [==============================] - 0s 2ms/step - loss: 0.5524 - accuracy: 0.7532

5/5 [==============================] - 0s 2ms/step - loss: 0.5270 - accuracy: 0.7662

5/5 [==============================] - 0s 2ms/step - loss: 0.5008 - accuracy: 0.7727

5/5 [==============================] - 0s 2ms/step - loss: 0.4917 - accuracy: 0.7857

5/5 [==============================] - 0s 2ms/step - loss: 0.4883 - accuracy: 0.7922


```python

```


```python

```


```python

```


```python
#keras.utils.plot_model(model_func, "my_first_model.png")
```


```python

```
