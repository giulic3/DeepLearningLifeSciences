import deepchem as dc
import tensorflow as tf
import deepchem.models.tensorgraph.layers as layers

from tensorflow.examples.tutorials.mnist import input_data

# Read dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Transfor dataset into format readable by deepchem
train_dataset = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
test_dataset = dc.data.NumpyDataset(mnist.test.images, mnist.test.labels)

model = dc.models.TensorGraph(model_dir='mnist')

# Images in MNIST are 28x28, flattened 784
# None means that the input can be of any dimension - we can use it as variable batch size
feature = layers.Feature(shape=(None, 784))
# 0..9 digits
label = layers.Label(shape=(None, 10))
# Reshape flattened layer to matrix to use it with convolution
make_image = layers.Reshape(shape=(None, 28, 28), in_layers=feature)

conv2d_1 = layers.Conv2D(num_outputs=32, activation_fn=tf.nn.relu, in_layers=make_image)
conv2d_2 = layers.Conv2D(num_outputs=64, activation_fn=tf.nn.relu, in_layers=conv2d_1)

flatten = layers.Flatten(in_layers=conv2d_2)
dense1 = layers.Dense(out_channels=1024, activation_fn=tf.nn.relu, in_layers=flatten)
dense2 = layers.Dense(out_channels=10, activation_fn=None, in_layers=dense1)

# Computes the loss for every sample
smce = layers.SoftMaxCrossEntropy(in_layers=[label, dense2])
# Average all the losses
loss = layers.ReduceMean(in_layers=smce)
model.set_loss(loss)

# Convert the output from logits to probs
output = layers.SoftMax(in_layers=dense2)
model.add_output(output)

model.fit(train_dataset, nb_epoch=1) # nb_epoch=10

# Use as metric accuracy: the fraction of labels that are correctly predicted
metric = dc.metrics.Metric(dc.metrics.accuracy_score)

train_scores = model.evaluate(train_dataset, [metric])
test_scores = model.evaluate(test_dataset, [metric])

