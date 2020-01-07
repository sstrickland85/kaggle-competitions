from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

train = pd.read_csv('train_clean.csv')
test = pd.read_csv('test_clean.csv')

#split the train dataset into features (x) and labels (y)
x = train.iloc[:, :-1]
y = train.iloc[:, -1]

#split the seen data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,
random_state = 42)

#######################################
#This code block trains and predicts for a Decision Tree using information gain
#train the model [Decision Tree]
tree = DecisionTreeClassifier(criterion = 'entropy')
tree.fit(x_train, y_train)

#predict on the unseen data
y_tree_pred = tree.predict(test).astype(int)

#print the results to a csv
tree_results = pd.DataFrame(data = {"PassengerId" : range(892, 1310), 
"Survived" : y_tree_pred})
tree_results.to_csv("titanic_predictions_tree.csv", index=False)


#######################################
#this code block trains and predicts for a Logistic Regression classifier
#train the model [Logistic Regression]
logR = LogisticRegression(verbose=True, max_iter=10).fit(x_train, y_train)

#predict on unseen data
y_log_pred = logR.predict(test).astype(int)


#print the results to a csv
log_results = pd.DataFrame(data = {"PassengerId" : range(892, 1310), 
"Survived" : y_log_pred})
log_results.to_csv("titanic_predictions_log.csv", index=False)

#######################################
#printing the accuracy of each model
print("The decision tree model accuracy is: ", tree.score(x_test, y_test)*100, "%")
print("The logistic regression model accuracy is: ", logR.score(x_test, y_test)*100, "%")