
############
#step 3 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, X_val, y_val, kernel='linear', C=1.0):
    # Reshape data for sklearn (n_samples, n_features)
    X_train_sk = X_train.T
    y_train_sk = y_train.ravel()
    X_val_sk = X_val.T
    y_val_sk = y_val.ravel()
    
    # Create and train SVM
    svm = SVC(kernel=kernel, C=C, random_state=42)
    svm.fit(X_train_sk, y_train_sk)
    
    # Evaluate
    train_pred = svm.predict(X_train_sk)
    val_pred = svm.predict(X_val_sk)
    
    train_acc = accuracy_score(y_train_sk, train_pred) * 100
    val_acc = accuracy_score(y_val_sk, val_pred) * 100
    
    print(f"SVM ({kernel} kernel) Training Accuracy: {train_acc:.2f}%")
    print(f"SVM ({kernel} kernel) Validation Accuracy: {val_acc:.2f}%")
    
    return svm

# Train SVM
svm_model = train_svm(X_train, y_train, X_val, y_val)

# Evaluate on test set
test_preds_svm = svm_model.predict(X_test.T)
test_acc_svm = accuracy_score(test_y.ravel(), test_preds_svm) * 100
print(f"SVM Test Accuracy: {test_acc_svm:.2f}%")


