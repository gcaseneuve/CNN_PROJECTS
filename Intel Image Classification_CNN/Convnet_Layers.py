from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def CNN():
    #Instantiating a small convet for dogs vs. cats classification
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu',input_shape=(150,150,3)))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), strides=(1, 1), padding='same', activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3), strides=(1, 1), padding='same', activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3), strides=(1, 1), padding='same', activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(6, activation='softmax')) #Second Dense Layer has to be softmax for sign language

    model.summary()


    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model

