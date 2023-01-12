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

data = pd.read_csv("AMD CPU spreadsheet(cleaned).csv", encoding = 'latin1') 
## this drops null values from the dataset   
print(data.info())   

