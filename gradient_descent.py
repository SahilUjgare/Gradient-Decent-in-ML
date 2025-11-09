import numpy as np
import matplotlib.pyplot as plt


# Generate synthetic data (y = 2x + 1 + noise)
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.2


def gradient_descent(X, y, lr=0.1, iterations=100):
    """
    Perform Gradient Descent for simple linear regression (y = mX + b).
    
    Parameters:
        X (ndarray): Input feature(s)
        y (ndarray): Target values
        lr (float): Learning rate
        iterations (int): Number of iterations
        
    Returns:
        m, b (floats): Learned slope and intercept
        history (list): Cost function values at each iteration
    """
    m, b = 0, 0  # initial guesses
    n = len(X)
    history = []

    for i in range(iterations):
        y_pred = m * X + b
        error = y_pred - y
        cost = (1 / n) * np.sum(error ** 2)
        history.append(cost)

        # Compute gradients
        dm = (2 / n) * np.sum(X * error)
        db = (2 / n) * np.sum(error)

        # Update parameters
        m -= lr * dm
        b -= lr * db

        # Print progress every 10 steps
        if i % 10 == 0:
            print(f"Iteration {i}: Cost={cost:.4f}, m={m:.4f}, b={b:.4f}")

    return m, b, history


# Run gradient descent
m, b, history = gradient_descent(X, y, lr=0.5, iterations=100)

# Plot the results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + b, color='red', label='Best Fit Line')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Plot the cost function
plt.plot(history)
plt.title("Cost Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.show()
