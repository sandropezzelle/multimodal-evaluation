# import nltk
from tqdm import tqdm
import pickle
import json
import jsonlines
# nltk.download('punkt')
import numpy as np

mysentences = '/project/dmg_data/MMdata/COCO-VIST-final.txt_stratified_SENTS_100.filter_size=2278.pickle'
myvocab = '/project/dmg_data/MMdata/unique-words-filtered-2278.txt'
myimages = '/project/dmg_data/MMdata/img_ids_all.txt'
myindexes = '/project/dmg_data/MMdata/COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt'
myjsonl = '/project/dmg_data/MMdata/coco-vist_ann.jsonl'

def get_vocab(vocabtxt):
    vlist = []
    with open(vocabtxt, 'r') as vocab:
        content = vocab.readlines()
        for v in content:
            v = v.strip()
            vlist.append(v)
        vlists = set(vlist)
    return vlists 

def get_indexes(idxtxt):
    idlist = []
    with open(idxtxt, 'r') as indexes:
        content = indexes.readlines()
        for idx in content:
            idx = idx.strip().split('-')
            if len(idx)==4:
                # print(idx, 'vist')
                mykey = 'v_'
                image = str(idx[2])
            elif len(idx)==3:
                # print(idx, 'coco')
                mykey = 'c_'
                imagef = str(idx[1])
                image = str(imagef.zfill(12))
            fullidx = mykey+image
            idlist.append(fullidx)
    return idlist


def read_sentences(pklfile,sent_idx_list,img_idx_list,save_path):
    data = pickle.load(open(pklfile, 'rb'))
    # freq_list = []
    with jsonlines.open(save_path, mode='w') as writer:
        for elem, s in tqdm(list(enumerate(data))):
            sentences = []
            if sent_idx_list[elem] in img_idx_list:
                # save the datapoint
                # with jsonlines.open(save_path, mode='w') as writer:
                # sentences = []
                sentences.append(s.strip()[1:-1]) # to get rid of double quotes
                img_id = str(sent_idx_list[elem])
                name = str(img_id.split('_')[1])
                d = {'sentences': sentences, 'id': img_id, 'img_path': img_id} # TODO: was img_path: name 
                # print(d)
                writer.write(d)
                # we can also compute stats on frequency of words

def main():
    vocablist = get_vocab(myvocab)
    print('length my vocabulary:',len(vocablist))
    imglist = get_vocab(myimages)
    print('length my image indexes:',len(imglist))
    idxlist = get_indexes(myindexes)
    print('length my sentence indexes:', len(idxlist))
    read_sentences(mysentences,idxlist,imglist,myjsonl)


if __name__ == '__main__':
    main()
