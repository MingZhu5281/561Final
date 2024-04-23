import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load and prepare datasets
def load_data():
    train = pd.concat([pd.read_csv(f'202{year}DecPA.csv') for year in range(9, 3)])  # Assuming datasets from 2019 to 2022
    test = pd.read_csv('2023DecPA.csv')
    return train, test

def preprocess_data(train, test):
    # Handling missing data
    imputer = SimpleImputer(strategy='mean')
    features = ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_WAC', 'DEST_WAC', 'DEP_DELAY', 'DEP_DEL15', 'DISTANCE']
    
    train[features] = imputer.fit_transform(train[features])
    test[features] = imputer.transform(test[features])
    
    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train = train['ARR_DEL15']
    y_test = test['ARR_DEL15']
    
    return X_train, X_test, y_train, y_test

# Build the neural network model
def build_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to run the workflow
def main():
    train_df, test_df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
