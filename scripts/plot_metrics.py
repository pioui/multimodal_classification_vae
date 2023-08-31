
import matplotlib.pyplot as plt
import numpy as np
from trento_config import images_dir

models_names = ("E1", "E2", "E3", "E4")
metrics_values = {
    'Accuracy': (84.86, 92.48, 84.68, 90.70),
    'F1' : (75.63, 92.43, 75.05, 90.01)
}


x = np.arange(len(models_names))  # the label locations
width = 0.4  # the width of the bars
multiplier = 0

colors = {
'Accuracy': '#7b3294', 
'Precision': '#a6dba0',
'Recall': '#008837',
'F1': '#c2a5cf' 
}

fig, ax = plt.subplots(layout='constrained', dpi=100)

for attribute, measurement in metrics_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color = colors[attribute] )
    ax.bar_label(rects, padding=4, fontsize=8)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score %')
ax.set_xlabel('Encoders')
ax.set_xticks(x + width/2, models_names)
ax.legend(loc='lower right', ncols=4)
ax.set_ylim(0, 100)

plt.savefig(f"{images_dir}trento_acc_f1.png")



metrics_values = {
    'Precision': (72.19, 92.17, 71.49, 88.83),
    'Recall': (80.76, 92.92, 80.50, 91.93),
}

multiplier = 0

fig, ax = plt.subplots(layout='constrained', dpi=100)

for attribute, measurement in metrics_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color = colors[attribute] )
    ax.bar_label(rects, padding=4, fontsize=8)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score %')
ax.set_xlabel('Encoders')
ax.set_xticks(x + width/2, models_names)
ax.legend(loc='lower right', ncols=4)
ax.set_ylim(0, 100)

plt.savefig(f"{images_dir}trento_pre_rec.png")
