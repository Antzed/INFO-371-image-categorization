# findings

## preparing the environment

In order to get the best result, I decided to run the network training on my laptop. The specs of my laptop is:

|hardware type| hardware name|
|---|---|
|CPU | 11 gen i7-11800H @ 2.3GHz|
| GPU | Nvidia Geforce RTX 3070 laptop GPU|

I decided to use docker. This is way i can skip the process of figuring out which different python packages I need to install.

I also allowed GPU processing, which will speed up the training process.

## Preparing the data

to prepare the data, I first get extracted the data and put them in in here a file structure like this:

```
 PS7/
    └── data/
        ├── train
        └── validate
```

I then run a few commands to get a sense of what the data look like. This is the number from both the training data and the validation data.
```
find . -type f -name "*_EN-*" | wc -l
find . -type f -name "*_DA-*" | wc -l
find . -type f -name "*_RU-*" | wc -l
find . -type f -name "*_ZN-*" | wc -l
find . -type f -name "*_TH-*" | wc -l
```
Number of EN files: 10492
Number of DA files: 5114
Number of RU files: 8204
Number of ZN files: 6702
Number of TH files: 9324


I then took a subset of a 1000 EN files and 1000 ZN files and backed up the rest in `train_big` and `validation_big` folder under the same directory for later use.

```bash
find . -type f -name "*_EN-*" | shuf -n 1000 | xargs -I {} cp {} ../train
find . -type f -name "*_ZH-*" | shuf -n 1000 | xargs -I {} cp {} ../train
```

Finally, I then applied the same logic to the validation dataset and copied 225 `EN` and 225 `ZN` images into the validation folder, along with 50 `DA` images. I choose the number 225, 225, and 50 because they add up to 500. I want to follow the 80:20 split, so when i have 2000 images as my training set, I will need 500 image for the validation set.

```bash
find . -type f -name "*_EN-*" | shuf -n 225 | xargs -I {} cp {} ../validation
find . -type f -name "*_ZN-*" | shuf -n 225 | xargs -I {} cp {} ../validation
find . -type f -name "*_DA-*" | shuf -n 50 | xargs -I {} cp {} ../validation
```

## Getting the code to work

I then copy the `square-circle-cross` code from the lab and updated the code so that the script will map the training data based on the language:

```python
trainingResults = pd.DataFrame({
    'filename': filenames,
    'category': np.where(pd.Series(filenames).str.contains('EN'), 'EN',
                         np.where(pd.Series(filenames).str.contains('ZN'), 'ZN', 'Unknown'))
})
```

I also applied the same logic in the later validation dataframe:

```python
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': np.where(pd.Series(fNames).str.contains('EN'), 'EN',
                np.where(pd.Series(fNames).str.contains('ZN'), 'ZN',
                np.where(pd.Series(fNames).str.contains('DA'), 'DA', 'Unknown')))
})
```

Finally, I changed the code defining image properties:

```python
targetWidth, targetHeight, channels = 256, 256, 1
```

Afterward the code runs. From the 1000 EN and ZN images, I get a validation accuracy of 0.89. This is with a network of (64, 128, 128, 128), kernel size 3, stride 1, and 10 epochs.

the confusion matrix:

predicted category |   EN  | ZN|
|---|---|---|
|DA         | 49    |1|
|EN   |       221  |   4 |
|ZN      |      1  | 224 |


## Test language

### attempt 1
Now let play with this network.

The first thing that needs to be done is increase the sample size. Hence I will use 5000 image from each languages(EN, ZN) for training and 1000 images for each language for testing.

So for the first attempt, we have a neural network of layer (64, 128, 128, 128), which in total uses 63072386 parameters and we trained the network for 10 epochs.

For this attempt we get a validation accuracy of 0.9995. however, it took about 43 minutes to run, which is pretty long training time. 

Here is the confusion matrix for this attempt:

|predicted category    |EN   |ZN|     
|---|---|---|
|EN         |1000    |0|
|ZN|            1|999|

we can see that we identified one chinese image wrong. If we print it out, we can see that it is becuase that particlar image do not have many feature points to begin with. Most of the space in the image is empty. The chinese characters shown on there has been cut in half.

![attempt1](./output/classification_results_attempt1.png)
The first image on the top is the wrongly predicted images.

So overall, this is a pretty accurate model. However, the the down side is that the training time is too long.

### attempt 2

So for my second attempt, I want reduce the training time by reducing the layers from 4 to 2, which means i'll get rid of one Conv2d layer and one dense layer. 

Surprisingly, doing this have increased my number of parameters from the previous 63072386 to 132130562. On a closer look however, this makes a lot of sense. By getting rid of the Conv2D layer, I also got rid of the pooling layer, which was responsible for reduces the number of parameters. So when those parameters hid the flattening layer, the total number of parameters increase.

The second attempt took about 8.6 minutes and resulted in a validation accuracy of 0.999. The result confusion matrix is the following:

|predicted category    |EN   |ZN|     
|---|---|---|       
|EN         |998     |2|
|ZN           |0  |1000|


We can see that the frequency of wrong predict increased by 1, which is not so much given that we have cut down the training time by roughly 3/4. Here some demo on the image prediction:

![attempt2](./output/classification_results_attempt2.png)

We can also see that from the top two images, we still face a similar problem as last attempt, which is that the model effective seems to decrease when the image has a lot of white spaces. This problem seems to stem from the fact that most image do not have that much white space, so  there isn't a lot of data of this scenarios the model can get training from.

### attempt 3

So in order to fix this problem, we will try and use the original training data set with all the EN images and ZN images. But in order to maintain our training time, we will increase the strides to 2.

So this leave us with 8442 EN and 5396 ZN training images and a total parameter of 32515842.

The result produced a validation accuracy is 0.985 with a training time of roughly 10.5 minutes. and the confusion matrix is this:

|predicted category    |EN   |ZN|     
|---|---|---|       
|EN         |2001    |49|
|ZN|            0|  1306|

![attempt3](./output/classification_results_attempt3.png).

This shows that it was not an equal trade between the increase in training sample and increase in strides. The quality of the model decreased with the increase of strides from 1 to 2.

## Adding more language.

