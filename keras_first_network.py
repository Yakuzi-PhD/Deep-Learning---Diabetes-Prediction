import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import numpy
import tensorflow
import keras

numpy.random.seed(3424525)

# Load the dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

print (len(X))
'''for x in X:
    print(x)'''
'''for z in x:
        print (z)'''

# Build the neural network model
model = keras.models.Sequential()
Dense = keras.layers.Dense
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit the model
model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=250, batch_size=10)

# Evaluate model
scores = model.evaluate(X,Y)
print ("/n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
