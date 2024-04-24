# How the idx_to_ILSVRC_ID.csv file was created

The file `idx_to_ILSVRC_ID.csv` was created by cross-referencing the indexes given by `torchvision.datasets.ImageFolder` when loading the ImageNet dataset, and the ILSVRC ID needed for the submission. These IDs can be found in the file `meta_clsloc.mat` provided in the development kit of the ImageNet dataset (`ILSVRC2013_devkit.tar.gz`). 

- The file `meta_clsloc.mat` contains the mapping between the WNIDs (WordNet IDs) and the ILSVRC IDs. The WNIDs are the name of the folders in the ImageNet dataset. 
- The mapping from WNIDs to indexes can be obtained by running the following code:

```python
val_dataset = datasets.ImageFolder("./imagenet/val") #works for train dataset as well
WNID_to_IDX = val_dataset.class_to_idx
```

Using these two dictionaries, we can create the mapping from the indexes to the ILSVRC IDs. 