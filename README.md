# multimodal-evaluation
Word Representation Learning in Multimodal Pre-Trained Transformers: An Intrinsic Evaluation

[repository under construction]

## Steps to reproduce the experiments from scratch

1. download COCO images [train2017](http://images.cocodataset.org/zips/train2017.zip) + [val2017](http://images.cocodataset.org/zips/val2017.zip) and VIST images [train + val & test](https://visionandlanguage.net/VIST/dataset.html) and save them locally

2. download COCO annotations [train/val](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and VIST annotations [DII](https://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz) + [SIS](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz) and save them locally

3. download the similarity/relatedness benchmarks used in our experiments [RG65, MEN, SIMLEX999, WORDSIM353,](https://edatos.consorciomadrono.es/file.xhtml?persistentId=doi:10.21950/AQ1CVX/7DHDQW&version=2.2) [SIMVERB3500](https://github.com/JoonyoungYi/datasets/tree/master/simverb3500) and save them locally

***

4. obtain vocabulary (unique words) of concatenated benchmarks (2453 words)

5. concatenate annotations from COCO + VIST (hence, VICO) and compute frequency of each V in vocabulary in it

- 2278 words have frequency >= 1; this set makes up our filtered vocabulary
- our filtered vocabulary can be found in [data/unique-words-filtered-2278.txt](data/unique-words-filtered-2278.txt)

6. filter the benchmarks based on our filtered vocabulary, i.e., only keep pairs whose words are both in VICO, and save them in .json format with sim/rel scores ranging from 0 to 1

- the five filtered benchmarks in .json format, including 7194 word pairs in total, can be found in [benchmarks_filtered](benchmarks_filtered)

7. extract (1) N samples from VICO such as all words in targeted vocabulary are present at least once (we obtained a subset of 113708, that we refer to as dataset) and (2) the indexes of the corresponding images

- our dataset: [COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle](data/COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle)
- our indexes: [COCO-VIST-final.txt_INDEXES_100.filter_size=2278](data/COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle)

8. use the indexes to extract the subset of images from VICO included in the dataset (we obtain 80295 unique images)

- to extract VICO's COCO images, use: [pick_coco2017_imgs_ALL.py](scripts/pick_coco2017_imgs_ALL.py)
- to extract VICO's VIST images, use: [pick_vist_imgs_ALL.py](scripts/pick_vist_imgs_ALL.py)

9. extract visual features of images:

- git clone py-bottom-up-attention
- make it work
- use scripts in [py-bottom-up-attention/demo](../py-bottom-up-attention/demo) to extract features .tsv format
- we obtain (1) MMdata/coco-imgs-features.tsv (15 GB) and (2) MMdata/vist-imgs-features.tsv (15 GB): **provide link to download these**
- save the files into [volta/data/coco_vist](volta/data/coco_vist)

10. convert visual features from .tsv to .lmdb (volta-ready)

- the scripts [convert_cocovist_lmdb.sh](volta/data/coco_vist/convert_cocovist_lmdb.sh) and [convert_cocovist_lmdb.py](volta/data/coco_vist/convert_cocovist_lmdb.py) can be used for that

- by running the scripts, you will obtain the files (1) data.mdb and (2) lock.mdb (~30GB) (these are in '/project/dmg_data/MMdata/imgfeats/volta/coco-vist_feat.lmdb' (30GB))


11. convert sentences from .pickle to .jsonl (volta-ready)

- the scripts [convert_sentences_to_json.sh](volta/data/coco_vist/convert_sentences_to_json.sh) and [convert_sentences_to_json.py](volta/data/coco_vist/convert_sentences_to_json.py) can be used for that
- you will obtain the file [coco-vist_ann.jsonl](data/coco-vist_ann.jsonl)