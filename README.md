# multimodal-evaluation
Word Representation Learning in Multimodal Pre-Trained Transformers: An Intrinsic Evaluation

[repository under construction]

## Steps to reproduce the experiments from scratch

1. download COCO images [train2017](http://images.cocodataset.org/zips/train2017.zip) + [val2017](http://images.cocodataset.org/zips/val2017.zip) and VIST images [train + val & test](https://visionandlanguage.net/VIST/dataset.html) and save them locally

2. download COCO annotations [train/val](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and VIST annotations [DII](https://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz) + [SIS](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz) and save them locally


2. download the following similarity/relatedness benchmarks: RG65, MEN, SIMLEX999, WORDSIM353, SIMVERB3000

3. obtain vocabulary (unique words) of concatenated benchmarks (2453 words)

4. concatenate annotations from COCO + VIST (hence, VICO) and check frequency in it of each V in vocabulary

- 2278 words have frequency >= 1; this set makes up our filtered vocabulary
- filtered vocabulary in [data/unique-words-filtered-2278.txt](data/unique-words-filtered-2278.txt)

5. filter benchmarks based on filtered vocabulary, i.e., only keep pairs whose words are both in VICO, and save them in .json format with sim/rel scores ranging from 0 to 1

6. extract (1) N samples from VICO such as all words in targeted vocabulary are present at least once (we obtained a subset of 113708, that we refer to as dataset) and (2) the indexes of the corresponding images

7. use the indexes to extract the subset of images from VICO included in the dataset (we obtain 80295 unique images)

- to extract VICO's COCO images: scripts/pick_coco2017_imgs_ALL.py
- to extract VICO's VIST images: scripts/pick_vist_imgs_ALL.py

8. extract visual features of images:

- git clone py-bottom-up-attention
- make it work
- use scripts in ../py-bottom-up-attention/demo to extract features .tsv format

- we obtain (1) MMdata/coco-imgs-features.tsv (15 GB) and (2) MMdata/vist-imgs-features.tsv (15 GB)

9. convert visual features from .tsv to .lmdb