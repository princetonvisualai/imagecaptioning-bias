from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import sys, warnings, json
import spacy_universal_sentence_encoder
from tqdm import tqdm

# Code for analyzing the differentiability of sentence embeddings

# convert caption into vector format
def word2vec(nlp, caption):
    cap = nlp(caption)
    return cap.vector

def main(args):
    isGT = False
    # tunable hyperparameters
    max_iter = 90

    # load data
    warnings.filterwarnings("ignore")
    file = '../annotations/captions_val2014.json'
    coco = COCO(file)
    df = pd.read_csv('../annotations/sim_stable.csv')
    df = df[:int(len(df) * 0.4)]
    light_imgs, dark_imgs = list(set(df['light_id'])), list(set(df['dark_id']))
    imgIds = light_imgs + dark_imgs
    nlp = spacy_universal_sentence_encoder.load_model('en_use_md')

    if isGT:
        print('Get caption embeddings')
        X, labels = [], []
        for i in tqdm(imgIds):
            aIds = coco.getAnnIds(imgIds=[i])
            anns = coco.loadAnns(aIds)
            captions = [x['caption'] for x in coco.loadAnns(ids=aIds)]
            for c in range(len(captions)):
                wordvec = list(word2vec(nlp, captions[c]))
                X.append(wordvec)
                if i in light_imgs: labels.append(0)
                else: labels.append(1)
        # train the MLP
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
        clf = MLPClassifier(random_state=0, max_iter = max_iter).fit(X_train, y_train)
        c_aucs = []
        print('Bootstrap results')
        np.random.seed(0)
        for i in tqdm(range(1000)):
            b_X, b_y = resample(X_test, y_test)
            c_aucs.append(roc_auc_score(b_y, clf.predict_proba(b_X)[:, 1]))
        conf = np.mean(c_aucs) - sorted(c_aucs)[25]
        print('Mean AUC: {:.4f} +/- {:.4f}'.format(np.mean(c_aucs) * 100, conf * 100))
    else:
        aucs = []
        model = args[1]
        for i in range(5):
            try:
                data = json.load(open('../results/{}_{}.json'.format(model, i)))
            except:
                raise Exception('File not found')
            print('Get caption embeddings')
            X, labels = [], []
            for i in tqdm(data):
                if i['image_id'] in imgIds:
                    wordvec = word2vec(nlp, i['caption'])
                    X.append(wordvec)
                    if i['image_id'] in light_imgs: labels.append(0)
                    else: labels.append(1)
            # train the MLP
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
            c_aucs = []
            clf = MLPClassifier(random_state=0, max_iter = max_iter).fit(X_train, y_train)
            aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        interval = st.t.interval(0.95, len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))
        print('Mean AUC: {:.4f} +/- {:.4f}'.format(np.mean(aucs) * 100, \
            np.mean(aucs)* 100 - interval[0]* 100))

if __name__=="__main__":
    main(sys.argv)
