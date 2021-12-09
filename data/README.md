# VIST dataset

The VIST dataset was obtained by concatenating COCO (train + val) with  VIST (train + val) and by extracting 113708 samples from it, i.e., <image, sentence> pairs containing at least one of the 2278 words of the 5 similarity/relatedness benchmarks (more details in the paper).

The folder contains:


LANGUAGE

- coco-vist_ann.jsonl: VICO language data in .jsonl format (VOLTA-ready)

- COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle (too large?): VICO sentences in .pickle format

- unique-words-filtered-2278.txt: vocabulary from 5 benchmarks used to filter COCO+VIST and obtain VICO



VISION

- coco_imgs-features.tsv (too large for github): image features of COCO images (39767)

- vist-imgs-features.tsv (too large for github): image features of VIST images (40528)

- [R] /MMdata/imgfeats/volta/coco-vist_feat.lmdb: image features of VICO in .lmdb format (VOLTA-ready) (80295)

- COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt: image identifier for each sample in VICO (VIST has format split-123-123-1; COCO, split-123-123) (113708)

- img_ids_all.txt: identifiers of all unique images in VICO (80295): c_ stands for COCO; v_ stands for VIST





