import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ReadCleanAndSplitData:

    def __init__(self, datafile_path):
        self.datafile_path = datafile_path

    def display_info(self, file: pd):
        # Display basic info
        print("SHAPE :", file.shape)
        categorical_cols = file.select_dtypes(include=['object']).columns
        clean_summary = pd.DataFrame({
            "Missing Values": file.isnull().sum(),
            "Duplicates": file.duplicated().sum(),
            "Data Types": file.dtypes,
            "Categorical columns": file[categorical_cols].nunique(),
        })
        print(clean_summary)

    def clean_data(self, df):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        df.drop("customerID", axis=1, inplace=True)

        cat_cols = df.select_dtypes(include=["object"]).columns
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df_encoded

    def train_data(self, df_encoded):
        X = df_encoded.drop("Churn", axis=1)
        y = df_encoded["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 7. Scale features (important for Logistic Regression & NB)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def read_data(self):
        df = pd.read_csv(self.datafile_path)
        df_encoded = self.clean_data(df)
        self.train_data(df_encoded)
        # display_info(df)
