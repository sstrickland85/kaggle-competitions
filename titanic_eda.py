#CS 235 \ Titanic Challenge \ Stephen Strickland
#This script imports and concatenates the original datasets (train & test), cleans the
#data, resplits into train and test, and saves each as a csv

#import libraries
import numpy as np
import pandas as pd

#import the train and test datasets then combine them 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = pd.concat([train, test], ignore_index=True, sort=False)


#returns summary info about the data frame; illuminates there are missing 
#values for Age, Cabin, and Embarked
#df.info()

df.head()	#shows the first 5 data points

#check for missing values on all features
df.isnull().sum()

#embarked only had 2 missing values; 3 unique values for embarked [S, C, Q]
#S was the most frequent and thus filled for the NaN values
df['Embarked'].value_counts()
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#cabin has 1014 missing values; I choose to create a new variable 'Unknown' and fill
df['Cabin'] = df['Cabin'].fillna("Unknown")

#fare only had one missing value; right skewed distribution; choose to use the median 
#due to the outlier
df['Fare'].fillna(df['Fare'].median(), inplace=True)

#returns the average age per sex
df.groupby(['Sex']).aggregate({'Age':np.mean}).reset_index()

#age was missing 263 values; choose to use the average age per sex
df['Age'].fillna(df.groupby('Sex')['Age'].transform("mean"), inplace=True)


#based on my worldly view I am dropping the following columns because it is unlikely
#they had any impact on the survival chance or they are captured by another variable
#for example the fare could be indicative of survival but is captured by Pclass
df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

#encoding the sex feature to integers
gender = {'male' : 1, 'female' : 2}
df.Sex = [gender[item] for item in df.Sex]

#create age groups and encode to integers
df['Age_Band'] = pd.cut(df['Age'], 4, labels=['1', '2', '3', '4'])

#dropping the original age column
df.drop('Age', axis=1, inplace=True)

#combine the SibSp and Parch into one feature called Fam_Size
df['Fam_Size'] = df['SibSp'] + df['Parch']

#drop the SibSp and Parch columns
df.drop(['SibSp', 'Parch'], axis = 1, inplace=True)

#moving the Survived column to the end to the dataframe
df = df[['Pclass', 'Sex', 'Age_Band', 'Fam_Size', 'Survived']]


#split the dataset back into train and test files; save files for modeling phase
train_clean = df.iloc[:891, :]
test_clean = df.iloc[891:, :]

#formatting the Survived column (all the NaN rows) in the test_clean dataframe
#the model won't take null values
test_clean = test_clean[['Pclass', 'Sex', 'Age_Band', 'Fam_Size']]

train_clean.to_csv("train_clean.csv", index=False)
test_clean.to_csv("test_clean.csv", index=False)

print(train_clean.head())
print(test_clean.head())