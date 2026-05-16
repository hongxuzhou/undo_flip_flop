import matplotlib.pyplot as plt

# Define the data
labels_inner = ['Correct (41.1%)', 'Incorrect (58.9%)']
sizes_inner = [41.1, 58.9]

# Assume 38.04% is the relative percentation w.r.t Correct 
# Right for the Wrong Reason = 41.1 * 0.3804 = 15.63%
labels_outer = ['Truly Correct', 'Right by Heuristic', 'Systematic Error', 'Noise']
sizes_outer = [25.47, 15.63, 35.0, 23.9] 

fig, ax = plt.subplots(figsize=(6, 6))
# Inner Circle
ax.pie(sizes_inner, labels=labels_inner, radius=0.7, labeldistance=0.4)
# Outer Circle
ax.pie(sizes_outer, labels=labels_outer, radius=1.0, wedgeprops=dict(width=0.3, edgecolor='w'))

plt.title("Behavioural Budget under Aggressive UNDO Pressure")
plt.show()