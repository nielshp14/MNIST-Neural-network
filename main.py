import tensorflow as tf
import numpy as np
import matplotlib as plt
import tensorflow_datasets as tf_Data

mnistDataset = tf.keras.datasets.mnist
(trainingData,trainingLabels), (testData, testLabels) = mnistDataset.load_data()
trainingData, testData = trainingData/255, testData/255

print("hello world")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape = (28,28)),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)]
)

lossFunktion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
  optimizer='adam',
  loss = lossFunktion,
  metrics=['accuracy']
)

model.fit(trainingData, trainingLabels, epochs=2)

result = model.evaluate(trainingData, trainingLabels)
print('loss = ', result[0], " accuracy = ", result[1])
print("We give the model a " + str(testLabels[0]))

predictions = model(testData[:1]).numpy()
predictionProcentage = tf.nn.softmax(predictions).numpy()
bestPrediction = np.argmax(predictionProcentage[0])
print("The model thought it was this: \n", bestPrediction)
from matplotlib import pyplot
for i in range(1):  
  pyplot.subplot(330 + 1 + i)
  pyplot.imshow(testData[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()