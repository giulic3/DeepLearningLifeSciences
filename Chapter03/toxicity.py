# Predict toxicity of molecules

import numpy as np
import deepchem as dc

# Load and featurize dataset from MoleculeNet
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()

# Each task is an enzymatic assay which measures whether the
# molecule in the dataset bind with the biological target in question
# the target is an enzyme which is believed to be linked with toxic responses
print(tox21_tasks)

# Contains multiple datasets
print(tox21_datasets)

# Class that perform transformations on dataset e.g. class balancing
print(transformers)

# Split into train, test and validation set
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Each sample (6246) has 1024 features
print(np.shape(train_dataset.X))
# Each sample (784 has 12 labels
print(np.shape(train_dataset.y))

model = dc.models.MultitaskClassifier(
    n_tasks=12,
    n_features=1024,  # Num of input features for each sample
    layer_sizes=[1000]  # Single hidden layer with 1000 units
)

# Train model for 10 epochs - 1 epoch is a pass of gradient descent over
# the entire dataset
model.fit(train_dataset, nb_epoch=10)
# Compute ROC AUC score across all tasks
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

# Evaluate model
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print(train_scores)
print(test_scores)