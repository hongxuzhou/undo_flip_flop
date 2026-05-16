import matplotlib.pyplot as plt
import numpy as np

# Data extract from Table 1 
tasks = ['Standard Flip-Flop', 'UNDO Flip-Flop']
models = ['1-Layer', '2-Layer']

# Correct ratio [%]
# Standard: 1L(ID:100, OOD:96.25), 2L(ID:100, OOD:99.05)
# UNDO: 1L(ID:95.95, OOD:79.10), 2L(ID:98.55, OOD:87.35)
data = {
    'Standard': {
        'ID': [100.0, 100.0], 
        'OOD': [96.25, 99.05]
    },
    'UNDO': {
        'ID': [95.95, 98.55],
        'OOD': [79.10, 87.35]
    }
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
plt.subplots_adjust(wspace=0.15)

x = np.arange(len(models))
width = 0.35

def plot_task(ax, task_name, title):
    id_vals = data[task_name]['ID']
    ood_vals = data[task_name]['OOD']
    
    rects1 = ax.bar(x - width/2, id_vals, width, label='In-Dist (ID)', color='#4C72B0', edgecolor='black', alpha=0.8)
    rects2 = ax.bar(x + width/2, ood_vals, width, label='Out-of-Dist (OOD)', color='#DD8452', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(70, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # decline degree (Delta)
    for i in range(len(models)):
        diff = ood_vals[i] - id_vals[i]
        ax.annotate(f'{diff:.1f}%', 
                    xy=(x[i] + width/2, ood_vals[i]),
                    xytext=(0, -15), 
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', 
                    fontweight='bold', fontsize=9)

plot_task(ax1, 'Standard', 'Monotonic (Standard)')
plot_task(ax2, 'UNDO', 'Non-Monotonic (UNDO)')

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.show()