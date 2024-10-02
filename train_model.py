from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy, X_train, X_test, y_train, y_test
