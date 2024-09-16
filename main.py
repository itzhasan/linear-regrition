from sklearn.linear_model import LinearRegression

reg = LinearRegression()
x = [[1],[2],[3],[4],[5],[6]]
y = [2 , 2.5 , 4.5 , 3 ,5 , 4.7]
reg.fit(x,y)

print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)

prediction = reg.predict([[5]])
print("Prediction for x=5:", prediction[0])

multiple_predictions = reg.predict([[5], [6], [7]])
print("Multiple predictions:", multiple_predictions)
