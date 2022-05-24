# multimodal-evaluation
Word Representation Learning in Multimodal Pre-Trained Transformers: An Intrinsic Evaluation


![diagram](mm_tacl_image.pdf)


***

## Setup

The code used in these experiments build on 4 repos:

- [Contextual2Static](https://github.com/rishibommasani/Contextual2Static) by Bommassani et al. 2020
- [VOLTA](https://github.com/e-bug/volta) by Bugliarello et al. 2021
- [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention) by Tan et al. 2019
- [Vokenization](https://github.com/airsplay/vokenization) by Tan et al. 2020

Before starting, these repos need to be cloned and built correctly following the instructions in the corresponding READMEs

Then, the folders in this repo corresponding to each of the repos, e.g., volta, should be merged with the source ones. 

To do so, you can use rsync -a /this/version/volta/ /source/version/volta/ (mind the trailing slash!). This way, some new files will be added to the source repos in the correct position


***


## Steps to reproduce the experiments from scratch

### BERT

1. download COCO annotations [train/val](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and VIST annotations [DII](https://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz) + [SIS](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz) and save them locally

2. download the similarity/relatedness benchmarks used in our experiments [RG65, MEN, SIMLEX999, WORDSIM353,](https://edatos.consorciomadrono.es/file.xhtml?persistentId=doi:10.21950/AQ1CVX/7DHDQW&version=2.2) [SIMVERB3500](https://github.com/JoonyoungYi/datasets/tree/master/simverb3500) and save them locally

3. concatenate sentences from VIST + COCO (hence, VICO) and check how many words in the concatenated benchmarks are in VICO

	- **VICO** including 1018367 samples: [VICO.txt](data/VICO.txt) 
	- 2300 words (out of 2453) are found in VICO, i.e., have frequency >= 1. Frequency distribution: [freq_vocab_2300_VICO.txt](stats/freq_vocab_2300_VICO.txt)

4. from the list of word pairs in the concatenated benchmarks (7917), retain the ones that have both words found in VICO (i.e., in the 2300-word list)

	- 7194 word pairs satisfy the constraint. These pairs make up our **filtered banchmarks**, which can be found in [benchmarks_filtered](benchmarks_filtered)
	- 2278 unique words make up these pairs. This vocabulary is our **filtered vocabulary**, which can be found in [data/unique-words-filtered-2278.txt](data/unique-words-filtered-2278.txt)

5. extract sentences (and corresponding image indexes) from VICO which contain at least one word in the filtered vocabulary. To keep the size of the dataset computationally tractable, we use a frequency-based method to stop sampling sentences containing highly frequent words

	- Our **dataset** including 113708 samples: [COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle](data/COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle)
	- Our corresponding **indexes**: [COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt](data/COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt)

5b. repeat (5) for Wikipedia to compare visually-grounded with non-visually-grounded data. Wikipedia data can be downloaded [here](https://storage.googleapis.com/lateral-datadumps/wikipedia_utf8_filtered_20pageviews.csv.gz)

	- Our **WIKI dataset** including 127246 samples: [wikipedia_utf8_filtered_20pageviews.csv_stratified_SENTS_100.filter_size=2278.pickle](data/wikipedia_utf8_filtered_20pageviews.csv_stratified_SENTS_100.filter_size=2278.pickle)
	- Our corresponding **WIKI indexes**: [wikipedia_utf8_filtered_20pageviews.csv_INDEXES_100.filter_size=2278.txt](data/wikipedia_utf8_filtered_20pageviews.csv_INDEXES_100.filter_size=2278.txt)

6. obtain contextualized vectors for both our **dataset** and **WIKI dataset** by running bommassani-based scripts 

	- Our dataset, contextualized vectors: **external link to them** (contextualized.pickle) (1.2GB)
	- WIKI dataset, contextualized vectors: **external link to them** (contextualized.pickle) (1.2GB)

7. use contextualized vectors to obtain correlation results for both our **dataset** and **WIKI dataset**


***

### VOLTA models


1. download COCO images [train2017](http://images.cocodataset.org/zips/train2017.zip) + [val2017](http://images.cocodataset.org/zips/val2017.zip) and VIST images [train + val & test](https://visionandlanguage.net/VIST/dataset.html) and save them locally

2. use the [indexes](data/COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt) from (5) to extract the subset of images from VICO included in the dataset. 80295 unique images are found

- to extract COCO images in VICO, use: [pick_coco2017_imgs_ALL.py](scripts/pick_coco2017_imgs_ALL.py)
- to extract VIST images in VICO, use: [pick_vist_imgs_ALL.py](scripts/pick_vist_imgs_ALL.py)

3. extract visual features for these 80295 images using Faster R-CNN with a ResNet-101 backbone:

- git clone py-bottom-up-attention
- build it
- use scripts in [py-bottom-up-attention/demo](../py-bottom-up-attention/demo) to extract features .tsv format
- you obtain (1) MMdata/coco-imgs-features.tsv (15 GB) and (2) MMdata/vist-imgs-features.tsv (15 GB): **TBD: link to download them**
- save the files into [volta/data/coco_vist](volta/data/coco_vist)

4. convert visual features from .tsv to .lmdb (volta-ready)

- the scripts [convert_cocovist_lmdb.sh](volta/data/coco_vist/convert_cocovist_lmdb.sh) and [convert_cocovist_lmdb.py](volta/data/coco_vist/convert_cocovist_lmdb.py) can be used for that
- by running the scripts, you will obtain the files (1) data.mdb and (2) lock.mdb (~30GB)

<!---
(these are in '/project/dmg_data/MMdata/imgfeats/volta/coco-vist_feat.lmdb' (30GB))
-->


5. convert sentences from .pickle to .jsonl (volta-ready)

- the scripts [convert_sentences_to_json.sh](volta/data/coco_vist/convert_sentences_to_json.sh) and [convert_sentences_to_json.py](volta/data/coco_vist/convert_sentences_to_json.py) can be used for that
- you will obtain the file [coco-vist_ann.jsonl](data/coco-vist_ann.jsonl)


***

### Vokenization

TBD



