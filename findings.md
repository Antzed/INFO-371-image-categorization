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

I then run a few commands to get a sense of what the data look like:
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

Afterward the code runs. From the 1000 EN and ZN images, I get a validation accuracy of 0.89. This is with a network of (64, 128, 128, 128), kernal size 3, stride 1, and 10 epochs.

the confusion matrix:

predicted category |   EN  | ZN|
|---|---|---|
|DA         | 49    |1|
|EN   |       221  |   4 |
|ZN      |      1  | 224 |


## Test language

Now let play with this network.

The first thing that needs to be done is increase the sample size. Hence I will take 5000 image from each languages(EN, ZN)