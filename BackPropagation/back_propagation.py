# Concrete example of backpropagation step by step. Generated with Clude AI.
# This is a simplified example with small dimensions and concrete values for clarity.

import numpy as np

print("=" * 70)
print("BACKPROPAGATION EXAMPLE WITH CONCRETE VALUES")
print("=" * 70)

# Network architecture:
# Input: 2 values
# Hidden layer: 2 units (ReLU activation)
# Output layer: 2 units (Softmax activation)
# Loss: Cross-entropy

print("\n--- NETWORK SETUP ---")
print("Input → Hidden (2 units, ReLU) → Output (2 units, Softmax) → Loss")

# Initialize weights and biases (small values for clarity)
np.random.seed(42)
W1 = np.array([[0.5, 0.3],   # Weight matrix: input (2) → hidden (2)
               [0.2, 0.4]])
b1 = np.array([0.1, 0.2])     # Bias for hidden layer

W2 = np.array([[0.6, 0.7],   # Weight matrix: hidden (2) → output (2)
               [0.8, 0.9]])
b2 = np.array([0.3, 0.4])     # Bias for output layer

print(f"\nW1 (input → hidden):\n{W1}")
print(f"b1: {b1}")
print(f"\nW2 (hidden → output):\n{W2}")
print(f"b2: {b2}")

# Input and true label
X = np.array([0.8, 0.6])      # Input: 2 features
y_true = 0                     # True class: 0

print(f"\n--- FORWARD PASS ---")
print(f"Input X: {X}")
print(f"True label: {y_true}")

# Layer 1: Input → Hidden (with ReLU)
print("\n1. Hidden Layer Computation:")
z1 = X @ W1 + b1  # Linear combination
print(f"   z1 = X @ W1 + b1")
print(f"   z1 = {X} @ {W1[0]} (first unit)")
print(f"   z1[0] = {X[0]}*{W1[0,0]} + {X[1]}*{W1[1,0]} + {b1[0]} = {z1[0]:.4f}")
print(f"   z1[1] = {X[0]}*{W1[0,1]} + {X[1]}*{W1[1,1]} + {b1[1]} = {z1[1]:.4f}")
print(f"   z1 = {z1}")

# ReLU activation
a1 = np.maximum(0, z1)
print(f"\n   a1 = ReLU(z1) = max(0, z1)")
print(f"   a1 = {a1}")

# Layer 2: Hidden → Output (with Softmax)
print("\n2. Output Layer Computation:")
z2 = a1 @ W2 + b2
print(f"   z2 = a1 @ W2 + b2")
print(f"   z2[0] = {a1[0]:.4f}*{W2[0,0]} + {a1[1]:.4f}*{W2[1,0]} + {b2[0]} = {z2[0]:.4f}")
print(f"   z2[1] = {a1[0]:.4f}*{W2[0,1]} + {a1[1]:.4f}*{W2[1,1]} + {b2[1]} = {z2[1]:.4f}")
print(f"   z2 = {z2}")

# Softmax activation
exp_z2 = np.exp(z2)
a2 = exp_z2 / np.sum(exp_z2)
print(f"\n   a2 = Softmax(z2)")
print(f"   exp(z2) = {exp_z2}")
print(f"   a2 = exp(z2) / sum(exp(z2))")
print(f"   a2 = {a2}")
print(f"\n   Prediction probabilities: Class 0: {a2[0]:.4f}, Class 1: {a2[1]:.4f}")

# Loss (Cross-entropy)
loss = -np.log(a2[y_true])
print(f"\n3. Loss Calculation:")
print(f"   Loss = -log(a2[{y_true}]) = -log({a2[y_true]:.4f}) = {loss:.4f}")

# BACKWARD PASS
print(f"\n--- BACKWARD PASS (Computing Gradients) ---")

# Gradient of loss with respect to output (softmax + cross-entropy derivative)
print("\n1. Output Layer Gradients:")
dz2 = a2.copy()
dz2[y_true] -= 1  # Derivative of softmax + cross-entropy
print(f"   dL/dz2 = a2 - one_hot(y_true)")
print(f"   dL/dz2 = {dz2}")
print(f"   (This says: predicted {a2[y_true]:.4f} but should be 1.0 for class {y_true})")

# Gradients for W2 and b2
dW2 = a1.reshape(-1, 1) @ dz2.reshape(1, -1)
db2 = dz2
print(f"\n   dL/dW2 = a1^T @ dL/dz2")
print(f"   dL/dW2:\n{dW2}")
print(f"\n   dL/db2 = dL/dz2 = {db2}")

# Backpropagate to hidden layer
print("\n2. Hidden Layer Gradients:")
da1 = dz2 @ W2.T
print(f"   dL/da1 = dL/dz2 @ W2^T")
print(f"   dL/da1 = {da1}")

# ReLU derivative (gradient is 0 where input was negative, 1 where positive)
dz1 = da1 * (z1 > 0)
print(f"\n   dL/dz1 = dL/da1 * ReLU'(z1)")
print(f"   ReLU'(z1) = {(z1 > 0).astype(int)} (1 where z1>0, else 0)")
print(f"   dL/dz1 = {dz1}")

# Gradients for W1 and b1
dW1 = X.reshape(-1, 1) @ dz1.reshape(1, -1)
db1 = dz1
print(f"\n   dL/dW1 = X^T @ dL/dz1")
print(f"   dL/dW1:\n{dW1}")
print(f"\n   dL/db1 = dL/dz1 = {db1}")

# WEIGHT UPDATE
print(f"\n--- WEIGHT UPDATE ---")
learning_rate = 0.1
print(f"Learning rate: {learning_rate}")

W2_new = W2 - learning_rate * dW2
b2_new = b2 - learning_rate * db2
W1_new = W1 - learning_rate * dW1
b1_new = b1 - learning_rate * db1

print(f"\nW2_new = W2 - lr * dW2")
print(f"Old W2:\n{W2}")
print(f"New W2:\n{W2_new}")
print(f"Change:\n{W2 - W2_new}")

print(f"\nW1_new = W1 - lr * dW1")
print(f"Old W1:\n{W1}")
print(f"New W1:\n{W1_new}")
print(f"Change:\n{W1 - W1_new}")

print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("1. Each weight has its own gradient showing how it affects the loss")
print("2. Gradients flow backward: output → hidden → input")
print("3. Chain rule connects gradients through layers")
print("4. Weights are updated proportionally to their gradients")
print(f"5. After update, the model should predict class {y_true} slightly better")