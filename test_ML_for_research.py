
import pandas as pd
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 20)
# way of getting wider output in the console

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv (r"F:\Machine Learning Shit My Own\StudentsPerformance.csv")
df=df[["race/ethnicity","math score", "reading score","writing score"]]
df.fillna(-99999, inplace=True)
X_features = df[["math score","reading score"]]
y_label = df["writing score"]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy)

math_score = 30
for reading_score in range(40,60):
    predicted_writing_score = classifier.predict([[math_score, reading_score]])
    writing_score = predicted_writing_score[0]
    print("Reading score: ", reading_score, "\t Writing score: ", int(writing_score))



