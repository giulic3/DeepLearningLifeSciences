# Train a neural network to predict the solubility of molecules.

import deepchem as dc
from rdkit import Chem

# First load the data
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# Create and train the model.

# n_task=1 means that there is only one output value, the solubility
# dropout rate of 20% is to avoid overfitting
model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)
model.fit(train_dataset, nb_epoch=100)

# Evaluate it.
# Use Pearson correlation coefficient
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print("Training set score")
print(model.evaluate(train_dataset, [metric], transformers))
print("Test set score")
print(model.evaluate(test_dataset, [metric], transformers))

# Use it to predict the solubility of some molecules.
smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C',
          'CCOC(=O)CC',
          'CSc1nc(NC(C)C)nc(NC(C)C)n1',
          'CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1',
          'Cc1cc2ccccc2cc1C']

mols = [Chem.MolFromSmiles(s) for s in smiles]
featurizer = dc.feat.ConvMolFeaturizer()
x = featurizer.featurize(mols)
predicted_solubility = model.predict_on_batch(x)

for m, s in zip(smiles, predicted_solubility):
    print()
    print('Molecule:', m)
    print('Predicted solubility:', s) # log (mols / liters)


"""
Training set score
computed_metrics: [0.958007644541195]
{'pearson_r2_score': 0.958007644541195}
Test set score
computed_metrics: [0.8480430913582656]
{'pearson_r2_score': 0.8480430913582656}

Molecule: COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C
Predicted solubility: [-0.19097732]

Molecule: CCOC(=O)CC
Predicted solubility: [1.2680938]

Molecule: CSc1nc(NC(C)C)nc(NC(C)C)n1
Predicted solubility: [-0.11521358]

Molecule: CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1
Predicted solubility: [0.00258102]

Molecule: Cc1cc2ccccc2cc1C
Predicted solubility: [-0.29237247]
"""

# See also https://github.com/deepchem/deepchem/blob/master/examples/tutorials/03_Modeling_Solubility.ipynbd