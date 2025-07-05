import numpy as np 
import h5py 
#import log_reg_code 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#step 1 
def load_and_prepare_data(validation_ratio=0.1, random_state=43):
    # Load raw data from .h5 files
    train_dataset = h5py.File('train_happy.h5', "r")
    test_dataset = h5py.File('test_happy.h5', "r")

    train_x_orig = np.array(train_dataset["train_set_x"][:])     # shape: (600, 64, 64, 3)
    train_y_orig = np.array(train_dataset["train_set_y"][:])     # shape: (600,)
    test_x_orig = np.array(test_dataset["test_set_x"][:])        # shape: (150, 64, 64, 3)
    test_y_orig = np.array(test_dataset["test_set_y"][:])        # shape: (150,)
    classes = np.array(test_dataset["list_classes"][:])

    # Normalize and flatten
    train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1) / 255.  # (600, 12288)
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1) / 255.     # (150, 12288)

    # Convert labels to shape (600,) to split
    train_y_flat = train_y_orig.reshape(-1,)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_x_flat, train_y_flat,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=train_y_flat
    )

    # Transpose to shape (features, examples)
    X_train = X_train.T     # (12288, N_train)
    X_val = X_val.T         # (12288, N_val)
    X_test = test_x_flat.T  # (12288, 150)

    y_train = y_train.reshape(1, -1)  # (1, N_train)
    y_val = y_val.reshape(1, -1)      # (1, N_val)
    test_y = test_y_orig.reshape(1, -1)  # (1, 150)

    return (X_train, y_train), (X_val, y_val), (X_test, test_y), classes


(X_train, y_train), (X_val, y_val), (X_test, test_y), classes = load_and_prepare_data()

print(f"Train set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {test_y.shape}")


#part 2 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def compute_cost(X, y, w, b):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(y * np.log(A + 1e-15) + (1-y) * np.log(1-A + 1e-15))
    return cost

def gradient_descent(X_train, y_train, X_val, y_val, X_test, test_y, w, b, learning_rate, num_iterations):
    m_train = X_train.shape[1]
    m_val = X_val.shape[1]
    m_test = X_test.shape[1]
    
    train_costs = []
    val_costs = []
    test_costs = []
    
    for i in range(num_iterations):
        # Forward propagation (training)
        A_train = sigmoid(np.dot(w.T, X_train) + b)
        
        # Backward propagation (training)
        dw = (1/m_train) * np.dot(X_train, (A_train - y_train).T)
        db = (1/m_train) * np.sum(A_train - y_train)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Compute costs for all datasets
        train_cost = compute_cost(X_train, y_train, w, b)
        val_cost = compute_cost(X_val, y_val, w, b)
        test_cost = compute_cost(X_test, test_y, w, b)
        
        train_costs.append(train_cost)
        val_costs.append(val_cost)
        test_costs.append(test_cost)
        
        # Print cost every 100 iterations
       # if i % 100 == 0:
       #     print(f"Iteration {i}: Train Cost = {train_cost:.4f}, Val Cost = {val_cost:.4f}, Test Cost = {test_cost:.4f}")
    
    return w, b, train_costs, val_costs, test_costs


def predict(X, w, b):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100
# Modified train_logistic_regression to include all costs
def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, test_y, learning_rate=0.001, num_iterations=2000,plot_costs=True):
    # Initialize parameters
    w, b = initialize_parameters(X_train.shape[0])
    
    # Train the model
    #print("Training logistic regression model...")
    w, b, train_costs, val_costs, test_costs = gradient_descent(
        X_train, y_train, X_val, y_val, X_test, test_y, w, b, learning_rate, num_iterations
    )
    
    if plot_costs==True:
    # Plot all three costs
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(train_costs)
        plt.ylabel('Training Cost')
        plt.xlabel('Iterations')
        plt.title('Training Cost vs Iterations')
        
        plt.subplot(3, 1, 2)
        plt.plot(val_costs)
        plt.ylabel('Validation Cost')
        plt.xlabel('Iterations')
        plt.title('Validation Cost vs Iterations')
        
        plt.subplot(3, 1, 3)
        plt.plot(test_costs)
        plt.ylabel('Testing Cost')
        plt.xlabel('Iterations')
        plt.title('Testing Cost vs Iterations')
        
        plt.tight_layout()
        plt.show()
    
    # Evaluate on training, validation, and test sets
    train_preds = predict(X_train, w, b)
    val_preds = predict(X_val, w, b)
    test_preds = predict(X_test, w, b)
    
    train_acc = accuracy(y_train, train_preds)
    val_acc = accuracy(y_val, val_preds)
    test_acc = accuracy(test_y, test_preds)
    
    if plot_costs==True:
       
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    return w, b, train_costs, val_costs, test_costs
# Train logistic regression

#w, b, train_costs, val_costs, test_costs = train_logistic_regression(
#    X_train, y_train, X_val, y_val, X_test, test_y
#)
# Evaluate on test set
#test_preds = predict(X_test, w, b)
#test_acc = accuracy(test_y, test_preds)
#print(f"Logistic Regression Test Accuracy: {test_acc:.2f}%")



print("part 1 done")

def test_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y,num_iterations=1000):
    learning_rates = [0.0001, 0.001, 0.005,0.01,0.02,0.05, 0.1,0.2,0.3,0.5, 1.0,10]
    test_accuracies = []
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        w, b, _, _, _ = train_logistic_regression(
            X_train, y_train, X_val, y_val, X_test, test_y,
            learning_rate=lr, num_iterations=num_iterations, plot_costs=False
        )
        test_preds = predict(X_test, w, b)
        test_acc = accuracy(test_y, test_preds)
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, test_accuracies, 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Logistic Regression: Test Accuracy vs Learning Rate')
    plt.grid()
    plt.show()

# Run the analysis
#test_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y,num_iterations=1000)



def test_accuracy_vs_training_size(X_train, y_train, X_val, y_val, X_test, test_y):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Fractions of training data to use
    test_accuracies = []
    
    for size in train_sizes:
        n_samples = int(size * X_train.shape[1])
        print(f"\nTraining with {n_samples} samples ({size*100:.0f}% of data)")
        
        # Use subset of training data
        X_subset = X_train[:, :n_samples]
        y_subset = y_train[:, :n_samples]
        
        w, b, _, _, _ = train_logistic_regression(
            X_subset, y_subset, X_val, y_val, X_test, test_y,
            learning_rate=0.01, num_iterations=1000
        )
        test_preds = predict(X_test, w, b)
        test_acc = accuracy(test_y, test_preds)
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot([int(s * X_train.shape[1]) for s in train_sizes], test_accuracies, 'o-')
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Logistic Regression: Test Accuracy vs Training Set Size')
    plt.grid()
    plt.show()

# Run the analysis
#test_accuracy_vs_training_size(X_train, y_train, X_val, y_val, X_test, test_y)





#SVM started 



from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss
import matplotlib.pyplot as plt

# SVM Implementation
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

# Train initial SVM model
svm_model = train_svm(X_train, y_train, X_val, y_val, kernel='linear')

# Evaluate on test set
test_preds_svm = svm_model.predict(X_test.T)
test_acc_svm = accuracy_score(test_y.ravel(), test_preds_svm) * 100
print(f"SVM Test Accuracy: {test_acc_svm:.2f}%")

# 4a: Plot loss with iterations (using SGDClassifier with hinge loss)
def train_svm_with_loss_plots(X_train, y_train, X_val, y_val, X_test, test_y, C=1.0, max_iter=1000):
    X_train_sk = X_train.T
    y_train_sk = y_train.ravel()
    X_val_sk = X_val.T
    y_val_sk = y_val.ravel()
    X_test_sk = X_test.T
    test_y_sk = test_y.ravel()
    
    # Use SGDClassifier with hinge loss (linear SVM)
    svm = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1/(C * len(y_train_sk)),
        max_iter=max_iter,
        tol=1e-3,
        random_state=42,
        learning_rate='optimal',
        verbose=0
    )
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    # Partial fit to track progress
    for epoch in range(max_iter):
        svm.partial_fit(X_train_sk, y_train_sk, classes=np.unique(y_train_sk))
        
        # Compute hinge loss
        train_loss = hinge_loss(y_train_sk, svm.decision_function(X_train_sk))
        val_loss = hinge_loss(y_val_sk, svm.decision_function(X_val_sk))
        test_loss = hinge_loss(test_y_sk, svm.decision_function(X_test_sk))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        
        if epoch % 100 == 0:
            print(f"Iteration {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Plot losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(train_losses)
    plt.ylabel('Training Loss (Hinge)')
    plt.xlabel('Iterations')
    plt.title('SVM Training Loss vs Iterations')
    
    plt.subplot(3, 1, 2)
    plt.plot(val_losses)
    plt.ylabel('Validation Loss (Hinge)')
    plt.xlabel('Iterations')
    plt.title('SVM Validation Loss vs Iterations')
    
    plt.subplot(3, 1, 3)
    plt.plot(test_losses)
    plt.ylabel('Testing Loss (Hinge)')
    plt.xlabel('Iterations')
    plt.title('SVM Testing Loss vs Iterations')
    
    plt.tight_layout()
    plt.show()
    
    return svm, train_losses, val_losses, test_losses

# Train SVM with loss tracking
svm_model, svm_train_loss, svm_val_loss, svm_test_loss = train_svm_with_loss_plots(
    X_train, y_train, X_val, y_val, X_test, test_y, max_iter=1000
)

# 4b: Test Accuracy vs Learning Rate for SVM
def svm_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y):
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    test_accuracies = []
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        svm = SGDClassifier(
            loss='hinge',
            alpha=0.0001,  # Small constant for demonstration
            learning_rate='constant',
            eta0=lr,
            max_iter=1000,
            random_state=42
        )
        svm.fit(X_train.T, y_train.ravel())
        test_preds = svm.predict(X_test.T)
        test_acc = accuracy_score(test_y.ravel(), test_preds) * 100
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, test_accuracies, 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('SVM: Test Accuracy vs Learning Rate')
    plt.grid()
    plt.show()

svm_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y)

# 4c: Test Accuracy vs Training Set Size for SVM
def svm_accuracy_vs_training_size(X_train, y_train, X_val, y_val, X_test, test_y):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # Fractions of training data
    test_accuracies = []
    
    for size in train_sizes:
        n_samples = int(size * X_train.shape[1])
        print(f"\nTraining with {n_samples} samples ({size*100:.0f}% of data)")
        
        X_subset = X_train[:, :n_samples].T
        y_subset = y_train[:, :n_samples].ravel()
        
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_subset, y_subset)
        
        test_preds = svm.predict(X_test.T)
        test_acc = accuracy_score(test_y.ravel(), test_preds) * 100
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot([int(s * X_train.shape[1]) for s in train_sizes], test_accuracies, 'o-')
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('SVM: Test Accuracy vs Training Set Size')
    plt.grid()
    plt.show()

svm_accuracy_vs_training_size(X_train, y_train, X_val, y_val, X_test, test_y)