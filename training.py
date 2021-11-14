#!/usr/bin/env python
# coding: utf-8

# # Facial Expression Recognition (Emotion Detection)

# ## Dataset : [FER-2013](https://www.kaggle.com/msambare/fer2013)
# ![fer2013](https://miro.medium.com/max/602/1*slyZ64ftG12VU4VTEmSfBQ.png)

# In[2]:


train_path = 'train'
val_path = 'test'


# In[3]:


import matplotlib.pyplot as plt
import os
def plot_images(img_dir, top=10):
    all_img_dirs = os.listdir(img_dir)
    img_files = [os.path.join(img_dir, file) for file in all_img_dirs][:5]
  
    plt.figure(figsize=(10, 10))
  
    for idx, img_path in enumerate(img_files):
        plt.subplot(5, 5, idx+1)
    
        img = plt.imread(img_path)
        plt.tight_layout()         
        plt.imshow(img, cmap='gray') 


# In[4]:


plot_images(train_path+'/angry')


# In[5]:


plot_images(train_path+'/disgust')


# In[6]:


plot_images(train_path+'/fear')


# In[7]:


plot_images(train_path+'/happy')


# In[31]:


plot_images(train_path+'/neutral')


# In[32]:


plot_images(train_path+'/sad')


# In[33]:


plot_images(train_path+'/surprise')


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers, regularizers


# In[35]:


emotion_labels = sorted(os.listdir(train_path))
print(emotion_labels)


# ## Data Generator

# In[116]:


batch_size = 64
target_size = (48,48)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


# ## Build Model

# In[37]:


input_shape = (48,48,1) # img_rows, img_colums, color_channels
num_classes = 7


# In[38]:


# Build Model
model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape)) #, data_format='channels_last', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()


# In[39]:


# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# ## Train Model

# In[118]:


num_epochs = 10
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VAL   = val_generator.n//val_generator.batch_size


# In[41]:


# Train Model
history = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs, verbose=1, validation_data=val_generator, validation_steps=STEP_SIZE_VAL)


# ## Save Model

# In[42]:


# Save Model
models.save_model(model, 'test_fer2013_cnn.h5') 


# ## Evaluate Model

# In[48]:


# Evaluate Model
score = model.evaluate_generator(val_generator, steps=STEP_SIZE_VAL) 
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


# ## Show Training History

# In[44]:


# Show Train History
keys=history.history.keys()
print(keys)

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')

