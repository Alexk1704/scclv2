import numpy as np
import matplotlib.pyplot as plt

# # Define the parameter space boundaries
# x = np.linspace(-1.5, 1.5, 200)
# y = np.linspace(-1.5, 1.5, 200)
# X, Y = np.meshgrid(x, y)

# # Objective functions for Task 1 and Task 2 (for simplicity, using circular contours)
# task1_perf = np.sqrt(X**2 + Y**2)  # Performance for Task 1 (centered at (A))
# task2_perf = np.sqrt((X - 1)**2 + (Y - 1)**2)  # Performance for Task 2 (centered at (B))

# # Draw contour for task performance
# plt.contour(X, Y, task1_perf, levels=10, cmap='Blues', alpha=0.5)
# plt.contour(X, Y, task2_perf, levels=10, cmap='Reds', alpha=0.5)

# # Points A (Task 1) and B (Task 2)
# A = np.array([0, 0])  # Optimal point for Task 1
# B = np.array([1, 1])  # Optimal point for Task 2

# # Initial point O
# O = np.array([-1, -1])

# # Trajectory of learning
# trajectory_task1 = [O, A]
# trajectory_task2 = [A, B]

# # Plot
# plt.plot(*zip(*trajectory_task1), marker='o', color='blue', label='Learning Task 1', linestyle='--')
# plt.plot(*zip(*trajectory_task2), marker='o', color='red', label='Learning Task 2', linestyle='--')

# # Mark points A and B
# plt.scatter(*A, color='blue')
# plt.text(A[0], A[1], ' Task 1 Optimal (A)', fontsize=10, verticalalignment='bottom', horizontalalignment='right')

# plt.scatter(*B, color='red')
# plt.text(B[0], B[1], ' Task 2 Optimal (B)', fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# # Formatting
# plt.xlabel('Parameter Dimension 1')
# plt.ylabel('Parameter Dimension 2')
# plt.title('Catastrophic Forgetting Schematic')
# plt.axhline(0, color='black', lw=0.5, ls='--')
# plt.axvline(0, color='black', lw=0.5, ls='--')
# plt.grid()
# plt.legend()
# plt.axis('equal')
# plt.show()

'''
Optimal set of weights is represtend by a single point A & B.
Each task has an optimal parameter configuration (or weight set) that allows the model to perform well on that task.
'''

'''
    Contours indicate the performance of the tasks in the parameter space. The blue contours represent the performance for Task 1, while the red contours signify Task 2.
    A and B are the optimal parameters for each task.
    Trajectory shows the learning path. As the model optimizes for Task 2 (moving towards point B), performance on Task 1 significantly degrades (the model "forgets").

This visualization gives a clear conceptual representation of catastrophic forgetting in a two-task scenario. You can adapt the contours or performance metrics to illustrate different characteristics of the tasks, but the basic structure will convey the idea effectively.
'''

# Simulated epochs
epochs = np.arange(1, 21)

# Simulated accuracy for Task 1
task1_accuracy_before = np.clip(1.0 - 0.05 * (epochs - 1), 0, 1)  # Decreases slightly as it learns Task 2
task1_accuracy_after = task1_accuracy_before * 0.9  # Assume a drop due to catastrophic forgetting

# Simulated accuracy for Task 2
task2_accuracy = np.clip(0.1 * epochs, 0, 1)  # Increases as the model learns Task 2

# Plotting
plt.figure(figsize=(10, 6))

# Task 1 Performance
plt.plot(epochs, task1_accuracy_before, label='Task 1 Accuracy (Before Task 2)', marker='o', color='blue')
plt.plot(epochs, task1_accuracy_after, label='Task 1 Accuracy (After Task 2)', linestyle='--', color='blue')
plt.axvline(x=10, color='gray', linestyle='--', label='Transition to Task 2')

# Task 2 Performance
plt.plot(epochs, task2_accuracy, label='Task 2 Accuracy', marker='o', color='red')

# Labels and legend
plt.title('Performance of Two Tasks During Sequential Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(epochs)
plt.grid()
plt.legend()
plt.show()

'''
    We simulate a total of 20 epochs of training.
    The accuracy for Task 1 declines slightly over the first 10 epochs (as indicated by task1_accuracy_before), then drops off significantly after transitioning to Task 2 (represented by task1_accuracy_after).
    The accuracy for Task 2 starts low and increases over time as the model learns this new task.
    There's a vertical line at epoch 10 to indicate the transition point where training on Task 2 begins.

    The blue line depicts Task 1's accuracy before and after training on Task 2.
    The red line shows Task 2's accuracy, which improves over time.
    The drop in Task 1's accuracy after learning Task 2 visually demonstrates catastrophic forgetting.
'''