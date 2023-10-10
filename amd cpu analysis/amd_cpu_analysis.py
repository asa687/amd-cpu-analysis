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
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("AMD CPU spreadsheet(cleaned with no nulls).csv", encoding = 'latin1')   
data2 = pd.read_csv("AMD CPU spreadsheet(cleaned 2).csv", encoding = 'latin1')  
cleanedData2 = data2.dropna(subset=['1kU Pricing(USD)'], inplace=False)
## this drops null values from the dataset  
print(data.info())   

coreNum = data["# of CPU Cores"].sum()
threadNum = data["# of Threads"].sum() 
baseClock = data["Base Clock(GHz)"].sum() 
 

plt.figure(figsize=(10, 8)) 
plt.style.use('fivethirtyeight')
plt.title("Distribution of prices") 
sns.distplot(data['1kU Pricing(USD)']) 
plt.xticks(np.arange(0, max(data['1kU Pricing(USD)'])+1, 625))
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
##removed caches
x = np.array(cleanedData2[['# of CPU Cores', '# of Threads', 'Base Clock(GHz)']])
y = np.array(cleanedData2["1kU Pricing(USD)"])  
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)   
print("Enter the number of cores: ")  
cores = int(input())
print("Enter the number of threads: ") 
threads = int(input())
print("Enter the base clock in Ghz: ")  
baseClock = float(input())

features = np.array([[cores, threads, baseClock]])
prediction = model.predict(features)   
print(prediction)  


clf = MLPClassifier(solver='lbfgs',max_iter=1600, alpha=1e-5, hidden_layer_sizes=(8, 6, 2), random_state=42, batch_size = 80) 
clf.fit(cleanedData2[['# of CPU Cores', '# of Threads', 'Base Clock(GHz)']].values, cleanedData2["1kU Pricing(USD)"].values)   
MLPClassifier(solver='lbfgs',max_iter=1600 ,alpha=1e-5, hidden_layer_sizes=(8,6 , 2), random_state=42, batch_size = 80)
prediction2 = clf.predict(features)  
print(prediction2)
 
