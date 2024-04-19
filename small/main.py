import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load datasets
def load_data():
    train_2020 = pd.read_csv('2020DecPA.csv')
    train_2021 = pd.read_csv('2021DecPA.csv')
    train_2022 = pd.read_csv('2022DecPA.csv')
    test_2023 = pd.read_csv('2023DecPA.csv')
    return train_2020, train_2021, train_2022, test_2023

# Combine train datasets
def combine_datasets(datasets):
    return pd.concat(datasets, ignore_index=True)

# Handling missing data
def handle_missing_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[['DEP_DELAY', 'DEP_DEL15', 'DISTANCE']] = imputer.fit_transform(df[['DEP_DELAY', 'DEP_DEL15', 'DISTANCE']])
    df.dropna(inplace=True)  # Assuming dropping rows where ArrDel15 is NaN
    return df

# Preprocess and train model
def preprocess_and_train(train_df, test_df):
    # Handling missing data
    train_df = handle_missing_data(train_df)
    test_df = handle_missing_data(test_df)
    
    # Features and target variable
    X_train = train_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_train = train_df['ARR_DEL15']
    X_test = test_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_test = test_df['ARR_DEL15']
    
    # Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

# Main execution function
def main():
    train_2020, train_2021, train_2022, test_2023 = load_data()
    train_df = combine_datasets([train_2020, train_2021, train_2022])
    preprocess_and_train(train_df, test_2023)

# Run main
if __name__ == "__main__":
    main()
