import pandas as pd


def read_and_display_info(file: pd):
    # Display basic info
    print("SHAPE :", df.shape)
    categorical_cols = df.select_dtypes(include=['object']).columns
    clean_summary = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Duplicates": df.duplicated().sum(),
        "Data Types": df.dtypes,
        "Categorical columns": df[categorical_cols].nunique(),
    })
    print(clean_summary)


df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

read_and_display_info(df)
