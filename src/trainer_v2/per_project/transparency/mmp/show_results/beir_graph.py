import matplotlib.pyplot as plt

# Data
methods = ['BM25', 'BM25T', 'CrossEncoder']
datasets = ['SciFact', 'HotpotQA', 'TREC-COVID', 'DBPedia', 'NFCorpus', 'NQ', 'FiQA-2018', 'SCIDOCS', 'Quora', 'ArguAna', 'Touché-2020', 'ViHealthQA']
scores = {
    'BM25': [0.678, 0.633, 0.583, 0.325, 0.319, 0.307, 0.245, 0.150, 0.775, 0.407, 0.499, 0.217],
    'BM25T': [0.678, 0.641, 0.602, 0.350, 0.348, 0.332, 0.248, 0.148, 0.738, 0.359, 0.337, 0.173],
    'CrossEncoder': [0.688, 0.725, 0.733, 0.447, 0.369, 0.462, 0.341, 0.163, 0.823, 0.311, 0.272, 0.168]
}

comp_methods = ["BM25T", "CrossEncoder"]
diffs = {}
for method in comp_methods:
    row = []
    for i in range(len(datasets)):
        row.append(scores[method][i] - scores["BM25"][i])
    diffs[method] = row


# Create a new figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the lines
for method in comp_methods:
    ax.plot(datasets, diffs[method], marker='o', label=method)

# Set the x-axis tick labels
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=45, ha='right')

# Set the y-axis limits

# Add labels and title
ax.set_xlabel('Datasets')
ax.set_ylabel('Scores')
ax.set_title('Comparison of BM25, BM25T, and CrossEncoder')

# Add a legend
ax.legend()

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()