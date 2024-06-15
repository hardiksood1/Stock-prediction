#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[2]:


df = pd.read_csv(r"C:\Users\hp\Desktop\codealpha internship\stock prediction\AAP.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


data = df['Close'].values.reshape(-1, 1)


# In[6]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# In[7]:


look_back = 60
X, y = [], []
for i in range(len(data) - look_back):
    X.append(scaled_data[i:i+look_back, 0])
    y.append(scaled_data[i+look_back, 0])
X, y = np.array(X), np.array(y)


# In[8]:


X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# In[12]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# In[13]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)


# In[14]:


predicted_prices = model.predict(X[-look_back:])
predicted_prices = scaler.inverse_transform(predicted_prices)


# In[15]:


plt.plot(df['Close'].values, label='Actual Prices')
plt.plot(np.arange(len(data)-look_back, len(data)), predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




