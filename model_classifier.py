import time
from keras.datasets import cifar10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from keras.utils import to_categorical

#Load CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Set input image shape
input_shape = x_train.shape[1:]

#Set labels
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Reshape labels [1] to [0100000000] for categorical crossentropy
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Set number of classes
n_classes = 10

#Normalize through color
x_train = x_train / 255
x_test = x_test / 255

#CNN structure
model = Sequential()
model.add(Conv2D(32,(5,5), activation='relu', input_shape=(32,32,3)) )
model.add(MaxPool2D(pool_size = (2,2)))
model.add( Conv2D(32, (5,5), activation='relu' ))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#Look inside the layer
#model.summary()

#Set loss and optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training of the model
model.fit(x_train, y_train_one_hot, batch_size = 256, epochs=1, validation_split = 0.2)

#Evaluation of the model
loss, acc = model.evaluate(x_test, y_test_one_hot, verbose=0)
print('Accuracy: %.3f' % acc)

#Save the model for TF Serving
ts = int(time.time())
file_path = f"./img_classifier_22_12/{ts}/"
model.save(filepath=file_path, save_format='tf')