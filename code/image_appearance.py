import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm

# Code for simple appearance difference between light and dark images

# calculates largest bounding box size
def bb_size(img_id, coco, catIds):
    aIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=False)
    anns = coco.loadAnns(aIds)
    area = int(coco.loadImgs(img_id)[0]['height'] * coco.loadImgs(img_id)[0]['width'])
    bb, ann_id  = 0, 0
    for a in anns:
        if a['area'] > bb:
            bb = a['area']
            ann_id = a['id']
    return ann_id, bb / area

# calculates number of people objects in image
def ppl_count(img_id, coco, catIds):
    aIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=False)
    anns = coco.loadAnns(aIds)
    if len(anns) > 5:
        return 5
    else:
        return len(anns)

# calculates the distance from the center
def distance_center(img_id, ann_id, coco):
    anns = coco.loadAnns(ann_id)[0]
    bb = anns['bbox']
    bb_cx, bb_cy = int(bb[0] + (bb[2] / 2)), int(bb[1] + (bb[3] / 2))
    img_cx, img_cy = int(coco.loadImgs(img_id)[0]['width'] / 2), int(coco.loadImgs(img_id)[0]['height'] / 2)
    return (np.sqrt((bb_cx-img_cx)**2 + (bb_cy-img_cy)**2))

def main():
    # set up COCO
    instancesFile = '../annotations/instances_val2014.json'
    coco_v = COCO(instancesFile)
    catIds = coco_v.getCatIds(catNms=['person']) # category Id of people subset

    # read in image annotations
    df = pd.read_csv('../annotations/bbAnnotations_2017combined.csv') # image-level annotations
    df = df[(df['bb_skin'] == 'Light') | (df['bb_skin'] == 'Dark')]
    imgIds = list(df['id'])
    labels = [1 if x == 'Dark' else 0 for x in list(df['bb_skin'])]

    # format data
    X, test_X = [], []
    labels, test_labels = [], []

    print('Format data')
    for i in tqdm(imgIds):
        ann_id, ratio = bb_size(i, coco_v, catIds)
        ppl = ppl_count(i, coco_v, catIds)
        dist = distance_center(i, ann_id, coco_v)
        gender = list(df.loc[df['id'] == i]['bb_gender'])[0]
        split = list(df.loc[df['id'] == i]['split'])[0]
        skin = list(df.loc[df['id'] == i]['bb_skin'])[0]

        if skin == 'Light': skin = 0
        else: skin = 1

        f, m, u, n = 0, 0, 0, 0
        if gender == 'Female': f = 1
        elif gender == 'Male': m = 1
        elif gender == 'Unsure': u = 1
        else: n = 1

        if split == 'train':
            X.append([ppl, dist, ratio, f, m, u, n])
            labels.append(skin)
        else:
            test_X.append([ppl, dist, ratio, f, m, u, n])
            test_labels.append(skin)

    # mean 0, std 1
    X, text_X = np.stack(X), np.stack(test_X)
    scaler = StandardScaler()
    scaler.fit(X)
    X = (scaler.transform(X))
    scaler.fit(test_X)
    test_X = (scaler.transform(test_X))

    np.random.seed(1) # set seed

    print('\nAll Features')
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42, class_weight='balanced').fit(X, labels)
    aucs = []
    print('Bootstrap samples')
    for i in tqdm(range(1000)):
        b_X, b_y = resample(test_X, test_labels)
        aucs.append(roc_auc_score(b_y, clf.predict_proba(b_X)[:, 1]))

    print('Mean AUC {} | 95% CI ({}, {})'.format(np.mean(aucs), sorted(aucs)[25], sorted(aucs)[975]))

if __name__=="__main__":
    main()
