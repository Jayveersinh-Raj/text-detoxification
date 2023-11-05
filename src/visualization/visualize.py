import matplotlib.pyplot as plt

# Placeholder values for ROUGE scores, BERT embedding similarity, and true toxic label count
rouge1_score = 0.592
rouge2_score = 0.357
rougeL_score = 0.572
bert_similarity = 0.94
toxic_label_count = 0.503

# Labels for the categories
categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT Similarity', 'Toxic Label Count%']

# Values for each category
values = [rouge1_score, rouge2_score, rougeL_score, bert_similarity, toxic_label_count]

# Plotting the values
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['blue', 'green', 'orange', 'red', 'purple'])
plt.ylabel('Score/Count')
plt.title('Visualization of ROUGE scores, BERT Similarity, and Toxic Label Count')
plt.ylim(0, 1)  # Setting y-axis limits for better visualization of scores

# The path to save the plot
save_path = '../../reports/figures/bloom_final_plot.png'

# Save the plot to the specified path
plt.savefig(save_path)

# Display the plot (optional)
plt.show()
