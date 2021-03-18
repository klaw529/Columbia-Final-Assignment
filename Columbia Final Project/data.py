import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

scaler = MinMaxScaler(copy=False)
data = pd.read_csv("day.csv")
numeric_data = ["season","yr","mnth",'holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered']


scaler.fit(data[numeric_data])
normalizedData = scaler.transform(data[numeric_data])

x = normalizedData
y = data['cnt']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_predict=  regressor.predict(X_test)
print(y_predict)


