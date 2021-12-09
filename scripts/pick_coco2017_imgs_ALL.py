import os
from shutil import copy2
# https://stackoverflow.com/a/123238

# with open('SIS-with-labels/sis/test.story-in-sequence.json', 'r') as f:
#     test = json.load(f)
#

if not os.path.exists('picked_images'):
    os.mkdir('picked_images')

with open('../vist/COCO-VIST-final.txt_INDEXES_100.filter_size=2278.txt', 'r') as f:
    all_ids = f.readlines()

picked_ids = []
picked_ids_coco = []

for line in all_ids:

    split_line = line.split('-')

    if len(split_line) == 4:
        # VIST! eg: test-72157632590003647-8408830180-4
        picked_ids.append(split_line[2])

    elif len(split_line) == 3:  
        # COCO! eg: train-389935-202941
        if len(split_line[1].strip()) == 12: # was wrongly 2
            padded = str(split_line[1].strip()) # was wrongly 2
            # print('padded original:',padded)
            picked_ids_coco.append(padded)
        else:
            n = split_line[1].strip() # was wrongly 2
            padded = str(n.zfill(12))
            if len(padded) == 12:
                picked_ids_coco.append(padded)
            # else:
                # print('PROBLEM!')

# 63256 40535
# 50452 39767

# X unique imgs in total from VIST and COCO
print(len(picked_ids_coco))
print(len(set(picked_ids_coco)))


count = 0
picked_count = 0

split_dirs = ['images/train2017']

mylist = []

for split_dir in split_dirs:

    for root, subdirs, files in os.walk(split_dir):

        for f in files:
            count += 1

            if count % 100 == 0:
                print(count)

            file_imgid = str(f.split('.')[0].strip())
            mylist.append(file_imgid)

            if file_imgid in picked_ids_coco:
                copy2(os.path.join(root, f), 'picked_images')
                picked_count += 1


# prova = list(set(picked_ids_coco) & set(mylist))
# print(len(prova))

print(len(picked_ids), len(set(picked_ids)))

print(len(picked_ids_coco), len(set(picked_ids_coco)))

print('picked', picked_count)
