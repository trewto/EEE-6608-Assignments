import numpy as np 
import h5py 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#step 1 
def load_and_prepare_data(validation_ratio=0.1, random_state=30):
    # Load raw data from .h5 files
    train_dataset = h5py.File('train_happy.h5', "r")
    test_dataset = h5py.File('test_happy.h5', "r")

    train_x_orig = np.array(train_dataset["train_set_x"][:])    
    train_y_orig = np.array(train_dataset["train_set_y"][:])     
    test_x_orig = np.array(test_dataset["test_set_x"][:])        
    test_y_orig = np.array(test_dataset["test_set_y"][:])        
    classes = np.array(test_dataset["list_classes"][:])

    # Normalize and flatten
    train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1) / 255. 
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1) / 255.   

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
    X_train = X_train.T     
    X_val = X_val.T         
    X_test = test_x_flat.T  

    y_train = y_train.reshape(1, -1)  # (1, N_train)
    y_val = y_val.reshape(1, -1)      # (1, N_val)
    test_y = test_y_orig.reshape(1, -1)  # (1, 150)

    return (X_train, y_train), (X_val, y_val), (X_test, test_y), classes


(X_train, y_train), (X_val, y_val), (X_test, test_y), classes = load_and_prepare_data()

print(f"Train set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {test_y.shape}")


#Logistic regreession started

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

        plt.figure(figsize=(12, 6))

        plt.plot(train_costs, 'b', label='Training Cost')
        plt.plot(val_costs, 'r', label='Validation Cost')
       # plt.plot(test_costs, 'g', label='Testing Cost')
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        #plt.title('Training, Validation and Testing Costs vs Iterations')
        plt.title('Training, Validation Costs vs Iterations')
        plt.legend()
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

w, b, train_costs, val_costs, test_costs = train_logistic_regression(
    X_train, y_train, X_val, y_val, X_test, test_y, 0.001,2000
)
# Evaluate on test set
#test_preds = predict(X_test, w, b)
#test_acc = accuracy(test_y, test_preds)
#print(f"Logistic Regression Test Accuracy: {test_acc:.2f}%")



print("part 1 done")

def test_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y,num_iterations=2000):
    learning_rates = [0.0001, 0.001, 0.01, 0.1,1.0,10]
 
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
test_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y,num_iterations=2000)



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
            learning_rate=0.01, num_iterations=2000, plot_costs=False
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
test_accuracy_vs_training_size(X_train, y_train, X_val, y_val, X_test, test_y)





#SVM started 
########################


from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss
import matplotlib.pyplot as plt
import numpy as np




# SVM Training Function (similar to train_logistic_regression)
def train_svm(X_train, y_train, X_val, y_val, X_test, test_y, 
              learning_rate=0.01, C=1.0, max_iter=2000, plot_costs=True):
    # Reshape data for sklearn (n_samples, n_features)
    X_train_sk = X_train.T
    y_train_sk = y_train.ravel()
    X_val_sk = X_val.T
    y_val_sk = y_val.ravel()
    X_test_sk = X_test.T
    test_y_sk = test_y.ravel()
    
    # Use SGDClassifier for loss tracking
    svm = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1/(C * len(y_train_sk)),
        learning_rate='constant',
        eta0=learning_rate,
        max_iter=max_iter,
        random_state=36,
        verbose=0
    )
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    # Partial fit to track progress
    for epoch in range(max_iter):
        if epoch % 100 == 0:
         print(f"Epoch {epoch+1}/{max_iter}")
        svm.partial_fit(X_train_sk, y_train_sk, classes=np.unique(y_train_sk))
        
        # Compute hinge loss
        train_loss = hinge_loss(y_train_sk, svm.decision_function(X_train_sk))
        val_loss = hinge_loss(y_val_sk, svm.decision_function(X_val_sk))
        test_loss = hinge_loss(test_y_sk, svm.decision_function(X_test_sk))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
    
    if plot_costs:
        # Combined plot
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, 'b', label='Training Loss')
        plt.plot(val_losses, 'r', label='Validation Loss')
        plt.ylabel('Hinge Loss')
        plt.xlabel('Iterations')
        plt.title('SVM Training/Validation Loss vs Iterations')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    
    # Evaluate accuracy
    train_pred = svm.predict(X_train_sk)
    val_pred = svm.predict(X_val_sk)
    test_pred = svm.predict(X_test_sk)
    
    train_acc = accuracy_score(y_train_sk, train_pred) * 100
    val_acc = accuracy_score(y_val_sk, val_pred) * 100
    test_acc = accuracy_score(test_y_sk, test_pred) * 100
    
    if plot_costs:
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    return svm, train_losses, val_losses, test_losses

# SVM Learning Rate Analysis
def svm_accuracy_vs_learning_rate(X_train, y_train, X_val, y_val, X_test, test_y, max_iter=500):
    learning_rates = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
    test_accuracies = []
    
    for lr in learning_rates:
        print(f"\nTraining SVM with learning rate: {lr}")
        svm, _, _, _ = train_svm(
            X_train, y_train, X_val, y_val, X_test, test_y,
            learning_rate=lr, max_iter=max_iter, plot_costs=False
        )
        
        test_pred = svm.predict(X_test.T)
        test_acc = accuracy_score(test_y.ravel(), test_pred) * 100
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
def svm_accuracy_vs_training_size_with_lr(X_train, y_train, X_val, y_val, X_test, test_y, learning_rate=0.01, max_iter=1000):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    test_accuracies = []
    
    # Calculate total number of samples
    total_samples = X_train.shape[1]
    
    for size in train_sizes:
        n_samples = int(size * total_samples)
        print(f"\nTraining with {n_samples} samples ({size*100:.0f}% of data)")
        
        X_subset = X_train[:, :n_samples].T
        y_subset = y_train[:, :n_samples].ravel()
        
        # Using SGDClassifier to control learning rate
        svm = SGDClassifier(
            loss='hinge',  
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=max_iter,
            random_state=30
        )
        svm.fit(X_subset, y_subset)
        
        test_pred = svm.predict(X_test.T)
        test_acc = accuracy_score(test_y.ravel(), test_pred) * 100
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Create custom x-axis labels showing both % and absolute number
    x_labels = [f"{size*100:.0f}%\n({int(size*total_samples)})" for size in train_sizes]
    
    plt.figure(figsize=(12, 6))  # Slightly wider figure for better label spacing
    plt.plot([int(s * total_samples) for s in train_sizes], test_accuracies, 'o-')
    
    # Set custom x-ticks and labels
    plt.xticks(
        ticks=[int(s * total_samples) for s in train_sizes],
        labels=x_labels
    )
    
    plt.xlabel('Training Set Size (% of total Original Training Samples)\n(Number of samples)')  # Multi-line label
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'SVM: Test Accuracy vs Training Size (LR={learning_rate})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    for i, acc in enumerate(test_accuracies):
        plt.text(
            x=int(train_sizes[i] * total_samples),
            y=acc + 0.5,  # Small offset above the point
            s=f"{acc:.1f}%",
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()  # Ensure labels don't get cut off
    plt.show()


print("\n=== Basic SVM Training ===")


svm_model, train_loss, val_loss, test_loss = train_svm(
    X_train, y_train, X_val, y_val, X_test, test_y,
    learning_rate=0.01, max_iter=2000
)

svm_model, train_loss, val_loss, test_loss = train_svm(
    X_train, y_train, X_val, y_val, X_test, test_y,
    learning_rate=0.0001, max_iter=2000
)



svm_model, train_loss, val_loss, test_loss = train_svm(
    X_train, y_train, X_val, y_val, X_test, test_y,
    learning_rate=0.000001, max_iter=2000
)





# 2. Learning rate analysis
print("\n=== Learning Rate Analysis ===")
svm_accuracy_vs_learning_rate(
    X_train, y_train, X_val, y_val, X_test, test_y
)

# 3. Training size analysis
print("\n=== Training Size Analysis ===")
svm_accuracy_vs_training_size_with_lr(
    X_train, y_train, X_val, y_val, X_test, test_y,0.0001, max_iter=2000
)
