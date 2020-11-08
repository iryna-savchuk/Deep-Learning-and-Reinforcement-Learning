# Import Scikit-Learn Model
from sklearn.neural_network import MLPClassifier

# Specify an activation function
mlp = MLPClassifier(hidden_layer_sizes = (5, 2), activation = 'logistic')

# Fit and predict data (similar to approach for other sklearn models)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)