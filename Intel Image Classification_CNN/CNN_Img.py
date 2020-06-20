''':cvar
Name : Guy Marvens Caseneuve
School: University of Massachusetts Dartmouth
Date: June 20th 2020
Contact: gmcaseneuve@gmail.com
'''
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Convnet_Layers import CNN
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd

#different label name
classes_name = ['buildings' , 'forest', 'glacier' , 'mountain', 'sea', 'street']
class_names_label = {class_name:i for i, class_name in enumerate(classes_name)}

#Create two directories for tarin and test images
Datasets = ["/Users/cf4ur/Documents/Research_Folder/Intel_Datasets(ML)/seg_train/seg_train", "/Users/cf4ur/Documents/Research_Folder/Intel_Datasets(ML)/seg_test/seg_test"]
Working_Dir = os.path.join('/Users/cf4ur/Documents/Research_Folder/Intel_Image_Classification','Train_Tst_Datasets')


train_list = 0
valid_list = 0
''':cvar
This is a function that will iterate through the original downloaded datasets, and create a folder with two sub-folders
Folder = Train_Tst_Datasets
    |-> 1st Sub-folder = Training
        |-> 6 different training / Validation sub-folders = Different categories of images: 'buildings' , 'forest', 'glacier' , 'mountain', 'sea', 'street'
    |-> 1st Sub-folder = Validation 
        |-> 6 different testing sub-folders = Different categories of images: 'buildings' , 'forest', 'glacier' , 'mountain', 'sea', 'street'
'''
def data_loader():
    i=0
    global train_list
    global valid_list
    Data_Preprocess_arr = ['Training', 'Validation']
    for datasets in Datasets:
        index =0
        labels = []
        Data_Preprocess_dir = os.path.join(Working_Dir, Data_Preprocess_arr[i])
        os.mkdir(Data_Preprocess_dir)
        print("Loading datasets in {}".format(datasets))
        for folder in os.listdir(datasets):
            #Get rid of any hidden files
            if folder == '.DS_Store':
                continue
            label = class_names_label[folder]

            Data_Preprocess_subdir = os.path.join(Data_Preprocess_dir, folder)
            os.mkdir(Data_Preprocess_subdir)
            for image in tqdm(os.listdir(os.path.join(datasets, folder))):
                img_path = os.path.join(os.path.join(datasets, folder), image)
                #Since we have the path of the image now, let's open and do some work with it
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (150,150))
                labels.append(label)
                cv2.imwrite(os.path.join(Data_Preprocess_subdir, str(index)+".jpg"), img)
                index += 1

        i += 1


''':cvar
Apply an if statement to avoid creating or overwriting the 'Train_Tst_Datasets' folder more than once. 
This function check if Train_Tst_Datasets directory exist , and if it exists, it checks how many folders it contains. If failed all these tests, then it will delete whatever 
folder(s) was there and create it properly.
'''
if os.path.exists(Working_Dir):
    count =0
    for file in os.listdir('/Users/cf4ur/Documents/Research_Folder/Intel_Image_Classification/Train_Tst_Datasets'):
        if file == '.DS_Store':
            continue
        else:
            count += 1
    if count < 2:
        file_path = Working_Dir
        shutil.rmtree(file_path)
        os.mkdir(Working_Dir)
        data_loader()
    #This return the directory of the train and validation folders
    train_dir = '/Users/cf4ur/Documents/Research_Folder/Intel_Image_Classification/Train_Tst_Datasets/Training'
    valid_dir = '/Users/cf4ur/Documents/Research_Folder/Intel_Image_Classification/Train_Tst_Datasets/Validation'

else:
    os.mkdir(Working_Dir)
    data_loader()

''':cvar
Now we have all our image , it is time write process the images. 
1- Open the images
2- Rescale it
3- double check the size
4- determine the batch size
5- Do some work with the images (shuffle, rotate, change size, etc...
'''
batch_size = 32
predict_batch_size = 30
img_size = (150,150)
epoch = 50


''':cvar
When processing the images we are gonna split the train datasets as shown below 
Training : 80% of training
Validation : 20 of training 
the model is training on 80 % of the training datasets and validate on 20%  of the same training datasets  
'''
#Train/Validation/Testing Image Generator
train_gen = ImageDataGenerator(rescale= 1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,validation_split=0.2)
test_datagen= ImageDataGenerator(rescale= 1./255)

#Create Train generator, Validation generator and Testing generator
train_generator = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse', shuffle=True, classes= classes_name, subset='training')
validation_generator = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse', shuffle=True, classes= classes_name, subset = 'validation')
test_generator = test_datagen.flow_from_directory(valid_dir, target_size=(150, 150),color_mode="rgb",shuffle = False, class_mode='sparse', batch_size=30)

#Let's train our model
def Train_CNN():
    #Create a model variable and set it equal to an instance of the CNN() class
    model = CNN()
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    #Train the model now
    history = model.fit_generator(train_generator, steps_per_epoch= train_generator.samples // batch_size, epochs = epoch , validation_data = validation_generator, validation_steps=validation_generator.samples // batch_size, callbacks=[csv_logger])
    model.save('Intel_Image_model.h5')#Save the model
    # convert the history.history dict to a numpy DataFrame:

def model_history(epoch, history):
    epochs = [i for i in range(epoch)]
    fig , ax = plt.subplots(1,2)

    #train_acc = history['accuracy']
    train_loss = history['loss']
    test_loss = history['val_loss']

    fig.set_size_inches(20,10)
    ax[0].plot(epochs , train_loss , 'ro-' , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , 'go-' , label = 'Validation Loss')
    ax[0].set_title('Model Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Training & Validation Loss")

    train_acc = history['accuracy']
    test_acc = history['val_accuracy']
    #test_loss = history['val_loss']

    ax[1].plot(epochs , train_acc , 'r-o' , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , 'g-o' , label = 'Testing Accuracy')
    ax[1].set_title('Model Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training & Validation Accuracy")
    plt.show()


def VisualizeCMap(test_generator_classes, pred_labels, classes_name):
    cm = confusion_matrix(test_generator_classes, pred_labels)
    ax = plt.axes()
    sn.heatmap(cm, annot=True,annot_kws={"size": 10}, xticklabels=classes_name, yticklabels=classes_name, ax = ax)
    ax.set_title('Confusion matrix')
    plt.show()

''':cvar
Since now we trained our model we should have three option,
1- Visualize the History graph 
2- Visualize the confusion map to the prediction 
3- Retrain the model
'''
try:
    new_model = load_model('Intel_Image_model.h5')
    history = pd.read_csv('training.log', sep=',', engine='python')
except:
    Train_CNN()
    new_model = load_model('Intel_Image_model.h5')
    history = pd.read_csv('training.log', sep=',', engine='python')


''':cvar
Predict the model after training and evaluate the accuracy of the model on the testing datasets
'''
print("Accuracy of the model is - {:.2f} %".format(new_model.evaluate_generator(test_generator, steps = test_generator.samples // predict_batch_size)[1]*100))
predict = new_model.predict_generator(test_generator,steps = test_generator.samples // predict_batch_size)
pred_labels = np.argmax(predict, axis = 1) #Gets the highest probability out of the predictions


''':cvar
Show the model history and the confusion matrix 
'''
model_history(epoch,history)
VisualizeCMap(test_generator.classes, pred_labels, classes_name)

