import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# in seconds

m1m2_training = np.round(214.30114579200745*50,2)
m1m2patch_training  = np.round(345.60576581954956*50,2)
multi_m1m2_training = np.round(225.797465801239*50,2)
svm=  np.round(0.7578825950622559,2)
rf=  np.round(10.0809528827667236,2)


# Experiment names and their corresponding elapsed times
experiments = ['M1+M2(E2)', 'M1+M2(E5)', 'Multi-M1+M2(E2)', 'SVM', 'RF']
elapsed_times = [m1m2_training, m1m2patch_training, multi_m1m2_training, svm, rf]  # Replace these with your actual elapsed times
log_elapsed_times = [np.log10(time) for time in elapsed_times]
# Create a bar chart
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
bars = plt.bar(experiments, log_elapsed_times, color='skyblue')

# Add labels and title
# plt.xlabel('Experiments')
# plt.ylabel(' Logarithm of Elapsed Time (seconds)',fontsize=12)
# plt.title('Comparison of Elapsed Times for Different Experiments')
plt.ylim(-0.5,5)
# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure labels fit in the figure
plt.axhline(color = 'black',)
for bar, time in zip(bars, elapsed_times):
    if time ==  np.round(0.7578825950622559,2):
        print('here')
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, 0.1, str(time), ha='center', fontsize=12)
    else:
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 0.1, str(time), ha='center', fontsize=12)


# plt.show()
plt.yticks([]) 
plt.savefig('timeplots.png')
# plt.figure(figsize=(8, 6))
# plt.scatter(experiments, log_elapsed_times, c='skyblue', s=100, alpha=0.7)
# plt.ylabel('Elapsed Time (seconds)')
# plt.title('Scatter Plot of Elapsed Times for Different Experiments')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

