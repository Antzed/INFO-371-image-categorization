# findings

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