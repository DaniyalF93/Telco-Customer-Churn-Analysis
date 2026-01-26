from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class TrainData:

    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)   # 🔥 training happens here

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Logistic Regression Accuracy: {accuracy:.4f}")

        return model, accuracy
