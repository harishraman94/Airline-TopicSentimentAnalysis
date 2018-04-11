import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('resource\\Tweets.csv')
y = data.negativereason
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4)

print (X_train.shape,X_test.shape,y_train.shape,y_test.shape,type(X_train))

X_train.to_csv('resource\\train.csv')
X_test.to_csv('output\\test.csv')