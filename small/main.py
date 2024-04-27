import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier  # Importing BalancedRandomForest

from xgboost import XGBClassifier

# Load datasets
def load_data():
    train_2018 = pd.read_csv('2018DecPA.csv')
    train_2019 = pd.read_csv('2019DecPA.csv')
    train_2020 = pd.read_csv('2020DecPA.csv')
    train_2021 = pd.read_csv('2021DecPA.csv')
    train_2022 = pd.read_csv('2022DecPA.csv')
    test_2023 = pd.read_csv('2023DecPA.csv')
    return train_2018, train_2019, train_2020, train_2021, train_2022, test_2023

# Combine train datasets
def combine_datasets(datasets):
    return pd.concat(datasets, ignore_index=True)

# Handling missing data
def handle_missing_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[['DEP_DELAY', 'DEP_DEL15']] = imputer.fit_transform(df[['DEP_DELAY', 'DEP_DEL15']])
    df.dropna(inplace=True)  # Assuming dropping rows where ArrDel15 is NaN
    return df

# Remove outliers in DEP_DELAY
def remove_outliers(df):
    # Remove DEP_DELAY outliers beyond 1500
    df = df[df['DEP_DELAY'] <= 1500]
    return df

'''
# Preprocess and train model
def preprocess_and_train(train_df, test_df):
    # Handling missing data
    train_df = handle_missing_data(train_df)
    test_df = handle_missing_data(test_df)

    # Remove outliers
    train_df = remove_outliers(train_df)
    test_df = remove_outliers(test_df)
    
    # Features and target variable
    X_train = train_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_train = train_df['ARR_DEL15']
    X_test = test_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_test = test_df['ARR_DEL15']
    
    # Decision Tree Classifier
    #clf = DecisionTreeClassifier()
    # Random Forest Classifier
    #clf = RandomForestClassifier()
    # Balanced Random Forest Classifier
    #clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    # XGBoost Classifier
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    # Save the figure
    plt.savefig('confusion_matrix.png')
    plt.close()  # Close the figure to free up memory
'''

# Preprocess and train model
def preprocess_and_train(train_df, test_df):
    # Handling missing data
    train_df = handle_missing_data(train_df)
    test_df = handle_missing_data(test_df)

    # Remove outliers
    train_df = remove_outliers(train_df)
    test_df = remove_outliers(test_df)
    
    # Features and target variable
    X_train = train_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_train = train_df['ARR_DEL15']
    X_test = test_df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']]
    y_test = test_df['ARR_DEL15']
    
    # Decision Tree Classifier
    #clf = DecisionTreeClassifier()
    # Random Forest Classifier
    #clf = RandomForestClassifier()
    # Balanced Random Forest Classifier
    #clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    # XGBoost Classifier
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    # Predict probabilities for ROC curve
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Predictions and evaluation
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
# Main execution function
def main():
    train_2018, train_2019, train_2020, train_2021, train_2022, test_2023 = load_data()
    train_df = combine_datasets([train_2018, train_2019, train_2020, train_2021, train_2022])
    preprocess_and_train(train_df, test_2023)

# Run main
if __name__ == "__main__":
    main()
