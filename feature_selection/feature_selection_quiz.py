import sklearn.linear_model.Lasso
features, labels = GetMyData()
regression = Lasso()
regression.fit(features, labels)
regression.predict([2, 4])

