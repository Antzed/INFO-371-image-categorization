#!/usr/bin/env python3
##
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

## Define image properties:
imgDir = "./data/"
targetWidth, targetHeight, channels = 256, 256, 1
imageSize = (targetWidth, targetHeight)

print("Load images from", imgDir)

## Prepare dataset for training model:
filenames = os.listdir(os.path.join(imgDir, "train"))
print(len(filenames), "images found")
trainingResults = pd.DataFrame({
    'filename': filenames,
    'category': np.where(pd.Series(filenames).str.contains('EN'), 'EN',
                np.where(pd.Series(filenames).str.contains('ZN'), 'ZN', 'Unknown'))
})
print("data files:")
print(trainingResults.sample(5))
nCategories = trainingResults.category.nunique()
print("categories:\n", trainingResults.category.value_counts())
## Create model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization

# sequential (not recursive) model (one input, one output)
model=Sequential()

model.add(Conv2D(64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 kernel_initializer = initializers.HeNormal(),
                 input_shape=(targetWidth, targetHeight, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(128,
                 kernel_size=3,
                 kernel_initializer = initializers.HeNormal(),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,
                kernel_initializer = initializers.HeNormal(),
                activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,
                kernel_initializer = initializers.HeNormal(),
                activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(nCategories,
                kernel_initializer = initializers.HeNormal(),
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

## Training and validation data generator:
trainingGenerator = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
).\
    flow_from_dataframe(trainingResults,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=True)
label_map = trainingGenerator.class_indices
## Model Training:
history = model.fit(
    trainingGenerator,
    epochs = 10,
)

## Validation data preparation:
validationDir = os.path.join(imgDir, "validation")
fNames = os.listdir(validationDir)
print(len(fNames), "validation images")
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': np.where(pd.Series(fNames).str.contains('EN'), 'EN',
                np.where(pd.Series(fNames).str.contains('ZN'), 'ZN',
                np.where(pd.Series(fNames).str.contains('DA'), 'DA', 'Unknown')))
})
print(validationResults.shape[0], "validation files read from", validationDir)
validationGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(validationResults,
                        os.path.join(imgDir, "validation"),
                        x_col='filename',
                        class_mode = None,
                        target_size = imageSize,
                        shuffle = False,
                        # do _not_ randomize the order!
                        # this would clash with the file name order!
                        color_mode="grayscale"
    )

## Make categorical prediction:
print(" --- Predicting on validation data ---")
phat = model.predict(validationGenerator)
print("Predicted probability array shape:", phat.shape)
print("Example:\n", phat[:5])

## Convert labels to categories:
validationResults['predicted'] = pd.Series(np.argmax(phat, axis=1),
                                           index=validationResults.index)
print(validationResults.head())
labelMap = {v: k for k, v in label_map.items()}
validationResults["predicted"] = validationResults.predicted.replace(labelMap)
print("confusion matrix (validation)")
print(pd.crosstab(validationResults.category, validationResults.predicted))
print("Validation accuracy", np.mean(validationResults.category == validationResults.predicted))

## Print and plot misclassified results
wrongResults = validationResults[validationResults.predicted != validationResults.category]
rows = np.random.choice(wrongResults.index, min(4, wrongResults.shape[0]), replace=False)
print("Example wrong results (validation data)")
print(wrongResults.sample(min(10, wrongResults.shape[0])))

## Plot 4 wrong and 4 correct results
plt.figure(figsize=(12, 12))
index = 1
for row in rows:
    filename = wrongResults.loc[row, 'filename']
    predicted = wrongResults.loc[row, 'predicted']
    img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
    plt.subplot(4, 2, index)
    plt.imshow(img)
    plt.xlabel(filename + " ({})".format(predicted))
    index += 1
# now show correct results
index = 5
correctResults = validationResults[validationResults.predicted == validationResults.category]
rows = np.random.choice(correctResults.index,
                        min(4, correctResults.shape[0]), replace=False)
print("Example correct results (validation data)")
print(correctResults.sample(min(10, correctResults.shape[0])))

for row in rows:
    filename = correctResults.loc[row, 'filename']
    predicted = correctResults.loc[row, 'predicted']
    img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
    plt.subplot(4, 2, index)
    plt.imshow(img)
    plt.xlabel(filename + " ({})".format(predicted))
    index += 1
plt.tight_layout()
# plt.show()

# Save the figure to a file instead of displaying it
plt.savefig('./output/classification_results.png', dpi=300)
plt.close()
