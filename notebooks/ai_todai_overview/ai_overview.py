import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display
import os
import re

OUT_DIR = "reports/ai_todai_overview"
os.makedirs(OUT_DIR, exist_ok=True)

def save_plot(name):
    filename = re.sub(r'[^A-Za-z0-9_]+', '_', name.lower()).strip('_') + ".png"
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight')
    plt.close()

class LinearTransform:
    def __init__(self, w, b):
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)

    def forward(self, x):
        return np.dot(self.w, x) + self.b


class VisualizationMixin:
    def plot_vector_transformation(self, x, y, title):
        plt.figure(figsize=(6, 6))
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Input x')
        plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, color='red', label='Transformed Yield (Wx+b)')
        
        max_limit = max(abs(np.concatenate([x, y]))) + 2
        plt.xlim(-max_limit, max_limit)
        plt.ylim(-max_limit, max_limit)
        plt.grid()
        plt.legend()
        plt.title(f'Geometric Intuition: {title}')
        save_plot(f'Geometric Intuition: {title}')

    def plot_decision_boundary(self, X, y_true, w, b, title):
        colors = ['red' if y == 1 else 'blue' for y in y_true]
        plt.figure(figsize=(6,5))
        plt.scatter(X[:,0], X[:,1], c=colors, edgecolor='k', zorder=5)
        
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        x1_vals = np.array([x_min, x_max])
        b_val = np.atleast_1d(b)[0]
        
        if w[1] != 0:
            x2_vals = (-w[0]*x1_vals - b_val) / w[1]
            plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
            if w[1] > 0:
                plt.fill_between(x1_vals, x2_vals, y_max+5, color='red', alpha=0.1)
                plt.fill_between(x1_vals, y_min-5, x2_vals, color='blue', alpha=0.1)
            else:
                plt.fill_between(x1_vals, x2_vals, y_max+5, color='blue', alpha=0.1)
                plt.fill_between(x1_vals, y_min-5, x2_vals, color='red', alpha=0.1)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(title)
        save_plot(title)
        
    def plot_weight_tracking(self, epochs_list, w_hist, b_hist, X, y_true):
        fig, axes = plt.subplots(1, len(epochs_list), figsize=(15, 4))
        for ax, (ep_w, ep_b, ep_num) in zip(axes, zip(w_hist, b_hist, epochs_list)):
            colors = ['red' if y == 1 else 'blue' for y in y_true]
            ax.scatter(X[:,0], X[:,1], c=colors, edgecolor='k', zorder=5)
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
            x1_vals = np.array([x_min, x_max])
            b_val = np.atleast_1d(ep_b)[0]
            
            if ep_w[1] != 0:
                x2_vals = (-ep_w[0]*x1_vals - b_val) / ep_w[1]
                ax.plot(x1_vals, x2_vals, 'k--', label='Boundary')
                if ep_w[1] > 0:
                    ax.fill_between(x1_vals, x2_vals, y_max+5, color='red', alpha=0.1)
                    ax.fill_between(x1_vals, y_min-5, x2_vals, color='blue', alpha=0.1)
                else:
                    ax.fill_between(x1_vals, x2_vals, y_max+5, color='blue', alpha=0.1)
                    ax.fill_between(x1_vals, y_min-5, x2_vals, color='red', alpha=0.1)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Epoch {ep_num}")
        plt.tight_layout()
        save_plot("weight_tracking_per_epoch")
        
    def plot_loss_curve(self, epochs, losses, title):
        plt.figure(figsize=(8, 4))
        plt.plot(range(epochs), losses, color='purple', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        save_plot(title)


w_sym = sp.Matrix([[1.5, 0.5], [-0.5, 1.0]])
x_sym = sp.Matrix([2.0, 1.0])
b_sym = sp.Matrix([1.0, -1.0])

print("W matrix:")
display(w_sym)
print("x input:")
display(x_sym)
print("b bias:")
display(b_sym)
print("W * x + b Result:")
display(w_sym * x_sym + b_sym)


class VectorVisualizer(LinearTransform, VisualizationMixin):
    def __init__(self, w_matrix=[[1.5, 0.5], [-0.5, 1.0]], b_vector=[1.0, -1.0], x_input=[2.0, 1.0], title="Example 1"):
        super().__init__(w=w_matrix, b=b_vector)
        self.x_input = np.array(x_input)
        self.title = title

    def run(self):
        y = self.forward(self.x_input)
        self.plot_vector_transformation(self.x_input, y, self.title)

# Example 1: Expanding Vector
visualizer_1 = VectorVisualizer(w_matrix=[[1.5, 0.5], [-0.5, 1.0]], b_vector=[1.0, -1.0], x_input=[2.0, 1.0], title="Example 1: Expanding Vector")
visualizer_1.run()

# Example 2: Shrinking & Shifting Vector
visualizer_2 = VectorVisualizer(w_matrix=[[0.5, -0.2], [0.1, 0.5]], b_vector=[0.0, 2.0], x_input=[2.0, 1.0], title="Example 2: Shrinking & Shifting Vector")
visualizer_2.run()


# =====================================================================
# 1. PERCEPTRON (1958)
# =====================================================================
# The Perceptron is the genesis of artificial neural networks.
# Mathematically, it operates precisely like a standard Linear Transformation
# (Wx + b) followed by a discrete step function threshold (z >= 0).
# Geometric Intuition: A single perceptron tries to draw a straight line
# (decision boundary) perfectly splitting two classes of data in space.
class PerceptronDemo(LinearTransform, VisualizationMixin):
    def __init__(self, x, y_true, lr=0.1, epochs=10, title='Perceptron Visual Decision Boundary'):
        X_arr = np.atleast_2d(x)
        num_features = X_arr.shape[1]
        np.random.seed(42)
        w_init = np.random.randn(num_features) * 0.1
        b_init = 0.0
        super().__init__(w=w_init, b=b_init)
        self.X = X_arr
        self.y_true = np.array(y_true)
        self.lr = lr
        self.epochs = epochs
        self.title = title

    def run(self):
        for _ in range(self.epochs):
            for i, x_i in enumerate(self.X):
                #print(x_i)
                # Calculate perceptron prediction
                z = self.forward(x_i)  # Linear Transform: Wx + b
                #print(z)
                y_pred = 1 if z >= 0 else 0  # Step Function Threshold
                error = self.y_true[i] - y_pred  # Error: 0 if correct, non-zero if wrong
                
                # Perceptron learning rule: update weights if prediction is wrong
                if error != 0:
                    self.w += self.lr * error * x_i  # Update weight vector
                    self.b += self.lr * error  # Update bias scalar
                    
        self.plot_decision_boundary(self.X, self.y_true, self.w, self.b, self.title)


# --- Example 1: Learning the AND Logic Gate ---
# Unlike passing explicit boundary weights, we start with random weights.
# The Perceptron uses the Perceptron Learning Rule to iteratively adjust 
# weights using discrete error corrections against target labels.
X_and = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_and = np.array([0, 0, 0, 1])
perceptron_1 = PerceptronDemo(x=X_and, y_true=y_and, lr=0.1, epochs=10, title='Perceptron: Learned AND Gate')
perceptron_1.run()

# --- Example 2: Pass/Fail classification ---
# Training a perceptron on a linearly separable physical dataset forecasting 
# whether a student passes based on hours studied vs hours continuously slept.
np.random.seed(42)
N = 40
study_hours = np.random.uniform(0, 10, N)
sleep_hours = np.random.uniform(3, 9, N)
X_students = np.column_stack([study_hours, sleep_hours])
scores = study_hours + 0.5 * sleep_hours
y_students = np.where(scores > 8.0, 1, 0)

perceptron_2 = PerceptronDemo(x=X_students, y_true=y_students, lr=0.1, epochs=20, title='Perceptron: Learned Exam Pass/Fail')
perceptron_2.run()


# =====================================================================
# Epoch Weight Tracker (Visualizing Parameter Updates)
# =====================================================================
# This tracks how the weights (w) and bias (b) jump around across epochs 
# using the discrete 'instant error' corrections of the Perceptron.
class EpochWeightTracker(PerceptronDemo):
    def __init__(self, x, y_true, lr=0.1, epochs=10):
        super().__init__(x, y_true, lr, epochs, title="Parameter Values After Each Epoch")
        # Start with intentionally 'bad' weights so we can watch it learn
        self.w = np.array([2.0, -3.0]) 
        self.b = np.array(2.0)
        self.epoch_w_history = []
        self.epoch_b_history = []

    def run_with_tracking(self):
        for epoch in range(self.epochs):
            for i, x_i in enumerate(self.X):
                z = self.forward(x_i)
                y_pred = 1 if z >= 0 else 0
                error = self.y_true[i] - y_pred
                
                if error != 0:
                    self.w += self.lr * error * x_i
                    self.b += self.lr * error
            
            # Save state at the end of the epoch
            self.epoch_w_history.append(self.w.copy())
            self.epoch_b_history.append(float(np.squeeze(self.b)))
        
        self.plot_results()

    def plot_results(self):
        w_hist = np.array(self.epoch_w_history)
        b_hist = np.array(self.epoch_b_history)
        epochs = np.arange(1, self.epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Parameter Trajectories per Epoch
        ax1.plot(epochs, w_hist[:, 0], label='$w_1$', marker='o', color='tab:blue')
        ax1.plot(epochs, w_hist[:, 1], label='$w_2$', marker='o', color='tab:orange')
        ax1.plot(epochs, b_hist, label='bias ($b$)', linestyle=':', marker='s', color='tab:green')
        ax1.set_title("Parameter Values After Each Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Value")
        ax1.set_xticks(epochs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Decision Boundaries per Epoch
        colors = ['red' if y == 1 else 'blue' for y in self.y_true]
        ax2.scatter(self.X[:,0], self.X[:,1], c=colors, edgecolor='k')
        
        # Calculate axis limits based on data
        x_min, x_max = self.X[:,0].min()-1, self.X[:,0].max()+1
        y_min, y_max = self.X[:,1].min()-1, self.X[:,1].max()+1
        x_vals = np.array([x_min, x_max])
        
        cmap = plt.get_cmap('viridis')
        line_styles = ['-', '--', '-.', ':']
        
        for i in range(self.epochs):
            w_epoch = w_hist[i]
            b_epoch = b_hist[i]
            
            if w_epoch[1] != 0:
                y_vals = (-w_epoch[0] * x_vals - b_epoch) / w_epoch[1]
                color = cmap(i / max(1, self.epochs - 1))
                style = line_styles[i % len(line_styles)]
                
                ax2.plot(x_vals, y_vals, color=color, linestyle=style, 
                         linewidth=2, label=f'Epoch {i+1}', alpha=0.8)

        ax2.set_title("Decision Boundaries Per Epoch")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_plot(self.title)

# --- Execution ---

np.random.seed(42)

# 1. Create two clusters closer together
class_0 = np.random.randn(15, 2) * 0.8 + np.array([1, 1])
y_0 = np.zeros(15)
class_1 = np.random.randn(15, 2) * 0.8 + np.array([-1, -1])
y_1 = np.ones(15)

# 2. Combine and SHUFFLE the training data
X_dummy = np.vstack((class_0, class_1))
y_dummy = np.concatenate((y_0, y_1))

shuffle_idx = np.random.permutation(len(y_dummy))
X_dummy = X_dummy[shuffle_idx]
y_dummy = y_dummy[shuffle_idx]

# 3. Run tracking with lower learning rate to see smaller steps
print("Running with Learning Rate: 0.05")
tracker1 = EpochWeightTracker(X_dummy, y_dummy, lr=0.05, epochs=10)
tracker1.run_with_tracking()


# =====================================================================
# 2. BACKPROPAGATION (1986)
# =====================================================================
# Backpropagation relies on gradient descent. Instead of instantly jumping 
# weights when a prediction is wrong (like the perceptron), it calculates 
# the continuous Mean Squared Error over the dataset. 
# It takes small, proportional steps down the loss gradient, allowing 
# it to smoothly contour to complex, noisy data without thrashing.
class BackpropagationDemo(LinearTransform, VisualizationMixin):
    def __init__(self, x, y_true, lr=0.01, epochs=50, title='Backprop Training'):
        X_arr = np.atleast_2d(x)
        num_features = X_arr.shape[1]
        np.random.seed(42)
        w_init = np.random.randn(num_features) * 0.1
        b_init = 0.0
        super().__init__(w=w_init, b=b_init)
        self.X = X_arr
        self.y_true = np.array(y_true)
        self.lr = lr
        self.epochs = epochs
        self.title = title

    def run(self):
        losses = []
        N = len(self.X)
        for ep in range(self.epochs):
            # Forward pass
            predictions = np.array([self.forward(x_i) for x_i in self.X]).flatten()  # Get all predictions
            
            # Mean Squared Error Loss
            error = predictions - self.y_true  # Continuous raw error array
            mse = np.mean(error**2)  # Mean Squared Error computation
            losses.append(mse)
            
            # Compute gradient of MSE loss with respect to predictions
            grad_w = (2/N) * np.dot(error, self.X)  # Gradient of weights
            grad_b = (2/N) * np.sum(error)  # Gradient of bias
            
            # Gradient descent weight updates
            self.w -= self.lr * grad_w  # Smooth continuous descent for weights
            self.b -= self.lr * grad_b  # Smooth continuous descent for bias
            
        self.plot_loss_curve(self.epochs, losses, self.title)


# --- Example 1: House Prices (Continuous Regression) ---
# Backprop handles continuous target values to accurately find a line of best fit.
np.random.seed(42)
N = 100
rooms = np.random.normal(3, 1, N)
age = np.random.uniform(1, 40, N)
X_houses = np.column_stack([rooms, age])
# True target: Price = 10 * rooms - 0.5 * age + 50
y_houses = 10.0 * rooms - 0.5 * age + 50.0 + np.random.randn(N)*2.0

backprop_1 = BackpropagationDemo(x=X_houses, y_true=y_houses, lr=0.001, epochs=50, title='Backprop: House Prices')
backprop_1.run()

# --- Example 2: Car Depreciation ---
age_car = np.random.uniform(0, 20, N)
miles_car = np.random.uniform(0, 150, N)
X_cars = np.column_stack([age_car, miles_car])
# True target: Price = 30 - 1 * age - 0.1 * miles
y_cars = 30.0 - 1.0 * age_car - 0.1 * miles_car + np.random.randn(N)*0.5 

backprop_2 = BackpropagationDemo(x=X_cars, y_true=y_cars, lr=0.0001, epochs=100, title='Backprop: Car Depreciation')
backprop_2.run()


# =====================================================================
# Backpropagation Tracker (Visualizing Smooth Gradient Descent)
# =====================================================================
# Visualizes the continuous, smooth updates of the weights relative to the
# mean squared error over epochs.
class BackpropagationTracker(BackpropagationDemo):
    def __init__(self, x, y_true, lr=0.001, epochs=50, title='Backpropagation Demo'):
        super().__init__(x, y_true, lr, epochs, title)
        # Tracking lists
        self.loss_hist = []
        self.w_hist = []
        self.b_hist = []
        self.pred_hist = [] # Track predictions at specific epochs
        self.x = self.X  # Alias used in run()

    def run(self):
        N = self.x.shape[0]
        
        # Save initial state predictions
        self.pred_hist.append((0, self.forward(self.x.T)))
        
        for epoch in range(self.epochs):
            y_pred = self.forward(self.x.T)
            
            # Calculate and store loss
            loss = 0.5 * np.mean((y_pred - self.y_true)**2)
            self.loss_hist.append(loss)
            
            # Save parameter states
            self.w_hist.append(self.w.copy())
            self.b_hist.append(self.b)
            
            # Calculate gradients
            grad_y = (y_pred - self.y_true) / N
            grad_w = np.dot(self.x.T, grad_y)
            grad_b = np.sum(grad_y)
            
            # Update parameters
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            # Save predictions at mid and final points for visualization
            if epoch == self.epochs // 4 or epoch == self.epochs - 1:
                self.pred_hist.append((epoch + 1, self.forward(self.x.T)))
            
        self.plot_results()

    def plot_results(self):
        epochs_arr = np.arange(1, self.epochs + 1)
        w_arr = np.array(self.w_hist)
        b_arr = np.array(self.b_hist)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: MSE Loss over time
        ax1.plot(epochs_arr, self.loss_hist, color='purple', linewidth=2)
        ax1.set_title("Mean Squared Error over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Parameter Trajectories
        ax2.plot(epochs_arr, w_arr[:, 0], label='Weight 1 (Rooms)', color='tab:blue')
        ax2.plot(epochs_arr, w_arr[:, 1], label='Weight 2 (Age)', color='tab:orange')
        ax2.plot(epochs_arr, b_arr, label='Bias', color='tab:green', linestyle=':')
        ax2.set_title("Parameter Updates over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Parameter Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: True vs Predicted Values
        cmap = plt.get_cmap('viridis')
        
        # Perfect prediction diagonal line
        min_val = min(self.y_true.min(), min(p[1].min() for p in self.pred_hist))
        max_val = max(self.y_true.max(), max(p[1].max() for p in self.pred_hist))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction ($y=x$)', alpha=0.5)

        # Plot predictions for selected epochs
        for i, (ep, preds) in enumerate(self.pred_hist):
            color = cmap(i / (len(self.pred_hist) - 1))
            label = 'Initial' if ep == 0 else f'Epoch {ep}'
            ax3.scatter(self.y_true, preds, color=color, label=label, alpha=0.6, edgecolors='none')

        ax3.set_title("True vs Predicted Prices")
        ax3.set_xlabel("True Price")
        ax3.set_ylabel("Predicted Price")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(self.title, fontsize=16, y=1.05)
        plt.tight_layout()
        save_plot(self.title)

# --- Execution ---

np.random.seed(42)
N = 100
rooms = np.random.normal(3, 1, N)
age = np.random.uniform(1, 40, N)
X_houses = np.column_stack([rooms, age])

# True target: Price = 10 * rooms - 0.5 * age + 50
y_houses = 10.0 * rooms - 0.5 * age + 50.0 + np.random.randn(N)*2.0

# Running the tracker
tracker = BackpropagationTracker(x=X_houses, y_true=y_houses, lr=0.002, epochs=50, title='Backprop: House Prices (Tracking)')
tracker.run()


# =====================================================================
# 3. ALGORITHM GEOMETRY: PERCEPTRON VS. BACKPROPAGATION
# =====================================================================
# Key Geometric Intuition Distinction:
# - Perceptron (Discrete Error Correction): If it makes a mistake, it instantly 
#   jumps the weights. Fast, but on noisy/unseparable data, it will never 
#   converge—it thrashes back and forth forever trying to fix impossible errors.
# - Backpropagation (Smooth Gradient Descent): Calculates the error surface for 
#   the whole batch. Moves weights smoothly down the slope. It can find the 
#   optimal "best fit" even when the data cannot be perfectly separated.
class PerceptronVsBackprop:
    def __init__(self, x, y_true, lr_perc=0.1, lr_bp=0.01, epochs=20):
        self.X = np.atleast_2d(x)
        self.y_true = np.array(y_true)
        self.epochs = epochs
        self.lr_perc = lr_perc
        self.lr_bp = lr_bp
        
        num_features = self.X.shape[1]
        np.random.seed(42)
        # Use exact same starting weights for a fair side-by-side comparison
        self.w_init = np.random.randn(num_features) * 0.1
        self.b_init = 0.0

    def run_comparison(self):
        N = len(self.X)
        
        # 1. Perceptron Training (Discrete Error Correction)
        w_p, b_p = self.w_init.copy(), self.b_init
        p_w_hist = [w_p.copy()]
        
        for _ in range(self.epochs):
            for i, x_i in enumerate(self.X):
                # Perceptron forward pass (Threshold step function)
                z = np.dot(w_p, x_i) + b_p
                y_pred = 1 if z >= 0 else 0
                error = self.y_true[i] - y_pred
                if error != 0:
                    # Instant discrete step towards the exact error
                    w_p += self.lr_perc * error * x_i  # JUMPS the weight
                    b_p += self.lr_perc * error
            p_w_hist.append(w_p.copy())
            
        # 2. Backpropagation / Gradient Descent (Continuous MSE Loss)
        w_b, b_b = self.w_init.copy(), self.b_init
        b_w_hist = [w_b.copy()]
        
        for _ in range(self.epochs):
            # Forward pass across all samples
            preds = np.array([np.dot(w_b, x_i) + b_b for x_i in self.X])
            error = preds - self.y_true
            
            # Smooth gradient computation over the entire batch
            grad_w = (2/N) * np.dot(error, self.X)
            grad_b = (2/N) * np.sum(error)
            
            # Continuous proportional gradient update
            w_b -= self.lr_bp * grad_w  # SMOOTHLY slopes the weight
            b_b -= self.lr_bp * grad_b
            
            b_w_hist.append(w_b.copy())
            
        self.plot_comparison(np.array(p_w_hist), np.array(b_w_hist))

    def plot_comparison(self, p_hist, b_hist):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = np.arange(len(p_hist))
        
        # Feature 1 Weight trajectory
        ax1.plot(epochs, p_hist[:, 0], 'r-o', label='Perceptron $w_1$ (Erratic Jumps)', zorder=5)
        ax1.plot(epochs, b_hist[:, 0], 'b-s', label='Backprop $w_1$ (Smooth Gradients)', zorder=4)
        ax1.set_title("Weight 1 Geometry: Jumps vs. Smooth Descent")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weight Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature 2 Weight trajectory
        ax2.plot(epochs, p_hist[:, 1], 'r-o', label='Perceptron $w_2$')
        ax2.plot(epochs, b_hist[:, 1], 'b-s', label='Backprop $w_2$')
        ax2.set_title("Weight 2 Geometry: Jumps vs. Smooth Descent")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Weight Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Algorithm Geometry: Discrete Error Correction (Perceptron) vs. Smooth Gradient Optimization (Backprop)", fontsize=14)
        plt.tight_layout()
        save_plot("Algorithm_Geometry_Perceptron_vs_Backprop")

# --- Execution ---
np.random.seed(42)

# Setup a 'messy' dataset where classes are overlapping slightly.
# This prevents Perceptron from ever perfectly solving it (causing thrashing) vs. Backprop finding optimal best fit.
X_comp = np.random.randn(50, 2)
target_logic = X_comp[:, 0] + X_comp[:, 1] + np.random.randn(50) * 0.8
y_comp = np.where(target_logic > 0, 1, 0)

comparison = PerceptronVsBackprop(X_comp, y_comp, lr_perc=0.1, lr_bp=0.1, epochs=30)
comparison.run_comparison()
# =====================================================================
# 4. CONVOLUTIONAL NEURAL NETWORKS (2012)
# =====================================================================
# Instead of plugging every single pixel into a dense line of transformations, 
# Convolutional Neural Networks (CNNs) scan patches dynamically using much smaller, 
# shared 'Filters'. This captures specific spatial structures (edges, corners) globally 
# and proved that Deep Learning scaling via GPUs could crush traditional image tasks (see: AlexNet, 2012).
class CNNDemo:
    def __init__(self, title="CNN: Edge Detection on MNIST"):
        from sklearn.datasets import load_digits
        digits = load_digits()
        self.image = digits.images[0] / 16.0
        self.filter = np.array([
            [1, -1],
            [1, -1]
        ])
        self.title = title

    def run(self):
        h, w = self.image.shape
        fh, fw = self.filter.shape
        output = np.zeros((h - fh + 1, w - fw + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                patch = self.image[i:i+fh, j:j+fw]
                output[i, j] = np.sum(patch * self.filter)
                
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(self.image, cmap='Blues')
        axes[0].set_title('Original 8x8 Image')
        axes[1].imshow(self.filter, cmap='Oranges')
        axes[1].set_title('2x2 Edge Filter')
        axes[2].imshow(output, cmap='Greens')
        axes[2].set_title('7x7 Convoluted Feature Map')
        plt.suptitle(self.title, fontsize=14)
        plt.tight_layout()
        save_plot(self.title)

cnn_demo = CNNDemo()
cnn_demo.run()


# =====================================================================
# 5. RECURRENT NEURAL NETWORKS / LSTMs (2014-2016)
# =====================================================================
# To handle sequenced arrays (like spoken sentences), Recurrent Neural Networks (RNNs) 
# were designed with literal memory banks. A hidden representation matrix calculates 
# its new output dynamically based on BOTH the current token and what was processed 
# dynamically in the previous tick of time.
class RNNDemo:
    def __init__(self, title="RNN: Flight Passengers Forecasting"):
        import seaborn as sns
        flights = sns.load_dataset('flights')
        passengers = flights['passengers'].values.astype(float)
        norm_p = (passengers - np.min(passengers)) / (np.max(passengers) - np.min(passengers))
        self.sequence = norm_p[:60]
        
        self.transform_x = LinearTransform(w=[[0.8], [-0.5]], b=[0.0, 0.0])
        self.transform_h = LinearTransform(w=[[0.7, 0.1], [-0.1, 0.6]], b=[0.1, -0.1])
        self.title = title

    def run(self):
        hidden_states = []
        hidden = np.array([0.0, 0.0])
        for x_scalar in self.sequence:
            x_t = np.array([x_scalar])
            z = self.transform_x.forward(x_t) + self.transform_h.forward(hidden)
            hidden = np.tanh(z)
            hidden_states.append(hidden)
            
        hidden_states = np.array(hidden_states)
        plt.figure(figsize=(9, 5))
        plt.plot(self.sequence, label='Normalized Passengers (Input)', color='black', linestyle='--', linewidth=2)
        plt.plot(hidden_states[:, 0], label='Hidden Node 1 (Learned Trend)', marker='s', alpha=0.7)
        plt.plot(hidden_states[:, 1], label='Hidden Node 2 (Learned Seasonality)', marker='^', alpha=0.7)
        plt.title(self.title)
        plt.xlabel('Time Sequence (Months)')
        plt.ylabel('Signal Intensity')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_plot(self.title)

rnn_demo = RNNDemo()
rnn_demo.run()


# =====================================================================
# 6. TRANSFORMERS (2017)
# =====================================================================
# Transformers proved RNN-style memory was a sequential gridlock bottleneck! 
# Their breakthrough lies in mapping sequence tokens indiscriminately in parallel 
# via 'Attention'. They evaluate every piece of an input against every other piece, 
# calculating a massive relational proximity score grid dynamically against Query, Key, and Value components.
class TransformerDemo:
    def __init__(self, title="Transformer: Self-Attention Map"):
        self.x = np.array([
            [1.0, 0.0, 0.1], 
            [0.1, 0.9, 0.1], 
            [0.2, 0.1, 0.8], 
            [0.9, 0.2, 0.0]  
        ])
        self.q_proj = LinearTransform(w=np.eye(3) * 0.5, b=[0]*3)
        self.k_proj = LinearTransform(w=np.eye(3) * 0.5, b=[0]*3)
        self.title = title

    def run(self):
        Q = np.array([self.q_proj.forward(row) for row in self.x])
        K = np.array([self.k_proj.forward(row) for row in self.x])
        
        scores = np.dot(Q, K.T) / np.sqrt(3)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attn_w = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        words = ["The", "cat", "sat", "down"]
        
        plt.figure(figsize=(6,5))
        plt.imshow(attn_w, cmap='magma')
        for i in range(len(words)):
            for j in range(len(words)):
                plt.text(j, i, f'{attn_w[i, j]:.2f}', ha="center", va="center", color="black" if attn_w[i,j] > 0.5 else "white")
        plt.colorbar(label='Self-Attention Relational Weight')
        plt.xticks(range(len(words)), words)
        plt.yticks(range(len(words)), words)
        plt.title(self.title)
        plt.ylabel('Token Querying Context')
        plt.xlabel('Token Being Attended To')
        save_plot(self.title)

transformer_demo = TransformerDemo()
transformer_demo.run()
