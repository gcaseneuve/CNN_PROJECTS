''':cvar
Name : Guy Marvens Caseneuve
School: University of Massachusetts Dartmouth
Date: June 20th 2020
Contact: gmcaseneuve@gmail.com
'''

from tensorflow.keras import layers, models, optimizers
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras.models import load_model
import seaborn as sn; sn.set(font_scale=1.4)
from tensorflow.keras.callbacks import CSVLogger

''':cvar
This is a datasets of a american sign languages images. We have a csv datasets that includes over 20k images, and 7k images of testing
1- First we treat it by creating an array for the label of letters then extract the csv datasets using pandas.
2- We convert the label from being a hot encoded label to a 1D array
3- Rescale the images from being between (0 -255) to (0-1) 
4- Since we are using a 1D we need to convert it into a 2D format pictures
'''

#Create array of label
class_name = ['A', 'B','C', 'D','E', 'F','G', 'H','I','K', 'L','M', 'N','O', 'P','Q', 'R','S', 'T','U', 'V','W', 'X','Y']

#Load the datasets
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")
train_df.head()

#extract the label form the file
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

#we need to define the label from one hot encoded to a 1D array
label_Binarizer = LabelEncoder()
y_train = label_Binarizer.fit_transform(y_train)
y_test = label_Binarizer.fit_transform(y_test)

#Extract the images from the csv file
x_train = train_df.values
x_test = test_df.values

#Normalize the data into rgb color between 0-1
x_train = x_train/255
x_test = x_test/255

#Transform the image from 1D to 3D for the input of the CNN's
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


''':cvar
Now to continue we our explanation
We have 4 different functions in our scripts
1- Img_Visualization: that lets you displat some images in your datasets just to make sure that you are not working with an empty datasets
2- CNN(): It is the architecture of our model 
3- model_history(): Displays a graph of our model
4- VisualizeCMap(): Displays a confusion matrix of our model 
'''
#Make sure that we have the images in x_train Datasets
def Img_Visualization():
    #Set the way images are appearing the matplotlib
    f, ax = plt.subplots(2,5)
    f.set_size_inches(10, 10)
    k = 0

    # iterate through the images to be printed
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
            k += 1
#Let's do some data augumentation

def CNN():
    # Instantiating a small convet for dogs vs. cats classification
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(24, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    return model



#print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

def model_history(epoch, history):
    epochs = [i for i in range(epoch)]
    fig , ax = plt.subplots(1,2)

    #train_acc = history['accuracy']
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,10)
    ax[0].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , 'ro-' , label = 'Validation Loss')
    ax[0].set_title('Model Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Training & Validation Loss")

    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    #test_loss = history['val_loss']

    ax[1].plot(epochs , train_acc , 'g-o' , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , 'r-o' , label = 'Testing Accuracy')
    ax[1].set_title('Model Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training and Validation Accuracy")
    plt.show()
def VisualizeCMap(test_generator_classes, pred_labels, classes_name):
    cm = confusion_matrix(test_generator_classes, pred_labels)
    ax = plt.axes()
    sn.heatmap(cm, annot=True,annot_kws={"size": 10}, xticklabels=classes_name, yticklabels=classes_name, ax = ax)
    ax.set_title('Confusion matrix')
    plt.show()

''':cvar
Once created the function we actually passing our images in the networking 
we will be using three type of information for this training. However, this time we wont be using ImageGenerator for this model since it adapted very well to the training. Below will be the informaiton we will be needed
batch_size : The number of batch we need for the training, in our case it's 32
pred_batch_size: After training the model this will be served for predication purposes. How many images images we'll be needing as batches to determine the steps in our predication. In our cases it's 30 
epoch : How many epochs to get to all our images in our training datasets. In our cases it's 20
'''
batch_size = 32
pred_batch_size = 30
epoch =  20

#Call the CNN Architcture
model = CNN()
''':cvar
While writing the scripts we decided to have the possibility to only train the model once. Therefore, we saved the model and also the history of the model in case we want to use it later on
'''
#variable that will call function tht saves the  model
csv_logger = CSVLogger('Sign_training.log', separator=',', append=False)
history = model.fit(x_train,y_train, batch_size = batch_size ,epochs = epoch , validation_split = 0.2, callbacks=[csv_logger])

#save the model
model.save('Sign_Language_Model.h5')
#load it for later uses
new_model = load_model('Sign_Language_Model.h5')

print("Accuracy of the model is - " , new_model.evaluate(x_test,y_test)[1]*100 , "%")

''':cvar
Code below only predict the model after training it. We also graph the history and the graph the confusion matrix
'''
#use prediction in model
predict = new_model.predict(x_test,batch_size = batch_size, steps = len(x_test) // pred_batch_size)
pred_labels = np.argmax(predict, axis = 1) #Gets the highest probability out of the predictions

#model_history(epoch,history)
VisualizeCMap(y_test, pred_labels, class_name)