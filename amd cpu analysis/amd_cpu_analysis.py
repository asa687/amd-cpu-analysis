"""
dataset from AMD

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("AMD CPU spreadsheet(cleaned with no nulls).csv", encoding = 'latin1') 
## this drops null values from the dataset  
print(data.info())   

coreNum = data["# of CPU Cores"].sum()
threadNum = data["# of Threads"].sum() 
baseClock = data["Base Clock(GHz)"].sum() 
 

plt.figure(figsize=(10, 8)) 
plt.style.use('fivethirtyeight')
plt.title("Distribution of prices") 
sns.distplot(data['1kU Pricing(USD)']) 
## plt.xticks(np.arange(min(data['1kU Pricing(USD)']), max(data['1kU Pricing(USD)'])+1, 1250))
plt.show()   

plt.figure(figsize=(10, 8)) 
plt.style.use('fivethirtyeight')
plt.title("Distribution of threads") 
sns.distplot(data['# of Threads']) 
plt.xticks(np.arange(0, max(data['# of Threads'])+1, 10))
plt.show()     

plt.figure(figsize=(10, 8)) 
plt.style.use('fivethirtyeight')
plt.title("Distribution of cores") 
sns.distplot(data['# of CPU Cores']) 
plt.xticks(np.arange(0, max(data['# of CPU Cores'])+1, 4))
plt.show() 

## model for a hypothetical cpu 
x = np.array(data[['# of CPU Cores', '# of Threads', 'Base Clock(GHz)', 'L1 Cache (KB)', 'L2 Cache(MB)', 'L3 Cache(MB)']])
y = np.array(data["1kU Pricing(USD)"])  
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)  
features = np.array([[8, 16, 4, 120, 11.5, 24.5]])
prediction = model.predict(features)