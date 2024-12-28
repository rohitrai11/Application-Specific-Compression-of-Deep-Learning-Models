import pickle
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForQuestionAnswering

pickle_file = open('path to saved matrix in pkl format', 'rb')
avg_hidden_state_outputs = pickle.load(pickle_file)

for i in range(len(avg_hidden_state_outputs)):
    for j in range(len(avg_hidden_state_outputs)):
        avg_hidden_state_outputs[i][j] = round(avg_hidden_state_outputs[i][j],2)


data = avg_hidden_state_outputs

annot = True

x_axis_labels = ['embed','enc1','enc2','enc3','enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10', 'enc11', 'enc12']
y_axis_labels = ['embed','enc1','enc2','enc3','enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10', 'enc11', 'enc12']

# Create a heatmap using Seaborn
sns.heatmap(data, annot=annot, xticklabels=x_axis_labels, yticklabels=y_axis_labels)  # You can choose different colormaps
plt.show()

thres = 0.9 - 0.01
ind_layer_to_del = set()  # Store indices of layers to delete.

# Loop over each layer 'i'
for i in range(len(avg_hidden_state_outputs)):
    # Skip if the layer is already marked for deletion.
    if i in ind_layer_to_del:
        continue

    # Compare layer 'i' with all subsequent layers 'k'
    for k in range(i + 1, len(avg_hidden_state_outputs)):
        if k in ind_layer_to_del:  # Skip deleted layers.
            continue

        # If similarity is above threshold, mark 'k' for deletion.
        if avg_hidden_state_outputs[i][k] >= thres:
            ind_layer_to_del.add(k)  # Mark layer 'k' for deletion.
            print(avg_hidden_state_outputs[i][k])

# Print the sorted list of deleted layers.
print("Layers to delete:", sorted(ind_layer_to_del))

ind_layer_to_del = [i-1 for i in ind_layer_to_del]

print("Index of Layers to delete:", sorted(ind_layer_to_del))

model = BertForQuestionAnswering.from_pretrained('path_to_fine_tuned_model')

enc = model.bert.encoder.layer
enc_list = list(enc)

for i in sorted(ind_layer_to_del, reverse=True):
    del enc_list[i]

new_enc = nn.ModuleList(enc_list)
model.bert.encoder.layer = new_enc
model.bert.config.num_hidden_layers -= len(ind_layer_to_del)

model.save_pretrained('give path to save compressed model')