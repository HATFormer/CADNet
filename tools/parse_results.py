import pandas as pd
from io import StringIO

# Data string
data = """
Class|IoU|Acc|Fscore|Precision|Recall
A|74.04|83.89|85.09|86.32|83.89
B|72.59|82.14|84.12|86.19|82.14
C|72.61|81.11|84.14|87.4|81.11
D|71.29|82.42|83.24|84.08|82.42
E|66.39|77.85|79.8|81.84|77.85
F|72.46|83.21|84.03|84.87|83.21
"""

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data.strip()), sep='|').set_index('Class').apply(pd.to_numeric)

# Calculate means
mean_iou = df['IoU'].mean()
mean_acc = df['Acc'].mean()
mean_fscore = df['Fscore'].mean()

# Prepare the final DataFrame
final_data = {
    'IoUA': df.at['A', 'IoU'], 'IoUB': df.at['B', 'IoU'], 'IoUC': df.at['C', 'IoU'],
    'IoUD': df.at['D', 'IoU'], 'IoUE': df.at['E', 'IoU'], 'IoUF': df.at['F', 'IoU'],
    'AccA': df.at['A', 'Acc'], 'AccB': df.at['B', 'Acc'], 'AccC': df.at['C', 'Acc'],
    'AccD': df.at['D', 'Acc'], 'AccE': df.at['E', 'Acc'], 'AccF': df.at['F', 'Acc'],
    'FscoreA': df.at['A', 'Fscore'], 'FscoreB': df.at['B', 'Fscore'], 'FscoreC': df.at['C', 'Fscore'],
    'FscoreD': df.at['D', 'Fscore'], 'FscoreE': df.at['E', 'Fscore'], 'FscoreF': df.at['F', 'Fscore'],
    'mean of IoU': mean_iou, 'mean of Acc': mean_acc, 'mean of Fscore': mean_fscore
}

final_df = pd.DataFrame([final_data])
print(final_df.head())  # Displaying the final DataFrame
