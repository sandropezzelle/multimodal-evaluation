# To reproduce the entire experimental pipeline

1. download COCO (train2017 + val2017) and VIST (train + val) images and annotations and save them locally

2. download the following similarity/relatedness benchmarks: RG65, MEN, SIMLEX999, WORDSIM353, SIMVERB3000

3. obtain vocabulary (unique words) of concatenated benchmarks (2453 words)

4. concatenate annotations from COCO + VIST (hence, VICO) and check frequency in it of each V in vocabulary

- 2278 words have frequency >= 1; this set makes up our filtered vocabulary

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


