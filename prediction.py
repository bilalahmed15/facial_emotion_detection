#!/usr/bin/env python
# coding: utf-8

# In[247]:


import keras
import numpy as np
import pandas as pd
import cv2
import PIL
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
from tkinter.ttk import *
n=0


# In[248]:


def myfunction():
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while(result):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("NewPicture1.jpg",frame)
        result = False
        
    videoCaptureObject.release()
    cv2.destroyAllWindows()


# In[249]:



window=Tk()
window.title("Taking image for emotion detection")
window.geometry('500x300')

    
style = Style()
style.configure('W.TButton', font =
               ('calibri', 25, 'bold',),
                foreground = 'Green')

button = Button(window, text='Capture',style='W.TButton', command=myfunction)

button.pack()
button.place(x=155, y=110)
emptylabel=Label(window)
emptylabel.place(x=200, y=300)
window.after(10000, window.destroy)
window.mainloop()


# In[250]:


sad_image = 'NewPicture1.jpg'
im1 = Image.open(r"NewPicture1.jpg") 


# In[251]:



reconstructed_model = keras.models.load_model("test_fer2013_cnn.h5")
    


# In[252]:


import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


img_batch = np.expand_dims(tf.keras.utils.img_to_array(tf.keras.utils.load_img(
    sad_image, color_mode='grayscale', target_size=(48,48,1),
    interpolation='nearest'
).convert('L')), axis=0)
prediction = reconstructed_model.predict(img_batch)


# In[253]:


prediction = prediction.astype(int)


# In[254]:


prediction


# In[255]:


var=''
var2=''
if prediction[0][0] == 1:
    var='You are Angry'
    var2='angry'
elif prediction[0][1] == 1:
    var='You feel Disgusted'
    var2='disgust'
elif prediction[0][2] == 1:
    var='You look Scared'
    var2='fear'
elif prediction[0][3] == 1:
    var='You look Happy'
    var2='happy'
elif prediction[0][4] == 1:
    var='You are feeling nothing'
    var2='neutral'
elif prediction[0][5] == 1:
    var='You are Sad'
    var2='sad'
elif prediction[0][6] == 1:
    var='You look Surprise'
    var2='surprise'


# In[256]:



import tkinter as tk
from tkinter import ttk
import tkinter.font as font
from tkinter import font as tkFont


# In[ ]:





# In[257]:


tkWindow = Tk()
tkWindow.title("Emotion Detection System")
tkWindow.geometry("600x500")
tkWindow.resizable(width=True, height=True)

def myfunction2():
    img = Image.open('NewPicture1.jpg')
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(tkWindow, image=img)
    panel.image = img
    
    panel.pack()
    emptylabel.config(text=var,font=("arial italic", 15)).pack()
    
button = Button(tkWindow, text='Detect Emotions', command=myfunction2)

button.pack()
button.place(x=200, y=200)
#this code is to design the button
style = Style()
 
style.configure('TButton', font =
               ('calibri', 20, 'bold'),
                    borderwidth = '4')
 
# Changes will be reflected
# by the movement of mouse.
style.map('TButton', foreground = [('active', '!disabled', 'green')],
                     background = [('active', 'black')])



#this button is to quit the running application
myFont = font.Font(family='Helvetica')
tk.button = Button(tkWindow, text='Quit The System', command=tkWindow.destroy)

tk.button.place(x=200, y=250)
emptylabel=Label(tkWindow)
emptylabel.place(x=220, y=150)    
tkWindow.mainloop()
window.mainloop()

# In[269]:


import os

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
full_file_paths = get_filepaths("C:/Users/BILAL/Desktop/New folder/train/"+var2)
type(full_file_paths[0])


# In[270]:


maxNum = 0

for i in range(len(full_file_paths)):
    if full_file_paths[i].find('filename_')!=-1:
        # Replace C:/Users/BILAL/Desktop/ with your own file path
        a = full_file_paths[i].replace('C:/Users/BILAL/Desktop/New folder/train/','')
        a = a.replace('\\','')
        a = a.replace(var2,'')
        print(var,a)
        a = a.replace('.jpg','')
        a = a.replace('filename_','')
        a = int(a)
        newFileName = 'filename_'+'0'
        if a >= maxNum:
            maxNum=a
            newFileName = 'filename_'+str(maxNum+1)+'.jpg'
        print(a,newFileName)
        im1.save('train/'+var2+'/'+newFileName)

