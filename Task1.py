import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.describe())

#Cleaning data
df=df.drop(columns=["Cabin"])
df["Age"]=df["Age"].fillna(df["Age"].median())

most_common=df["Embarked"].value_counts().idxmax()
df["Embarked"]=df["Embarked"].fillna(most_common)

#Rechecking
print("After cleaning")
print(df.info())
print(df.describe())
print(df.isnull().sum())

#encoding using one hot
df=pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True)

#checking
print(df.head())
print(df.columns)

#Visualize outliers using boxplots
import seaborn as sns
import matplotlib.pyplot as plt
numerical_cols=["Age","Fare","SibSp","Parch"]
plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

#Removing outliners
Q1=df[numerical_cols].quantile(0.25)
Q3=df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

#Standardizing
from sklearn.preprocessing import StandardScaler
numerical_cols=["Age","Fare","SibSp","Parch"]
scaler=StandardScaler()
df[numerical_cols]=scaler.fit_transform(df[numerical_cols])

#checking
print(df[numerical_cols].describe())
