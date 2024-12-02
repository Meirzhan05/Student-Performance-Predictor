import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


df = pd.read_csv('./student-mat.csv', delimiter=';')

# columns_to_drop = ['address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 
#                    'Fjob', 'traveltime', 'reason', 'sex', 'romantic', 'paid', 'schoolsup', 'famsup', 'activities', 
#                    'nursery', 'higher', 'internet', 'guardian', 'school']

# df.drop(columns=columns_to_drop, inplace=True)
df = df[["G1", "G2", "G3", 'studytime', 'failures', 'absences', 'health']]
predict = "G3"

X = np.array(df.drop(predict, axis=1))
y = np.array(df[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y)


model = LinearRegression()
model.fit(X_train, y_train)
print(predict)
# print(mean_squared_error(y_test, prediction))
acc = model.score(X_test, y_test)
print(acc * 100)

prediction = model.predict(X_test)
for x in range(len(prediction)):
    print(prediction[x], X_test[x], y_test[x])