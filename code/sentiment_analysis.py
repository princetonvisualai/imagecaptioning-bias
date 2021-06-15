from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import scipy.stats as st
import json

# Code for analyzing the sentiment using VADER

# returns sentiment score using VADER
def sentiment_scores(analyzer, sentence):
    snt = analyzer.polarity_scores(sentence)
    return (list(snt.values()))

def main(args):
    isGT = True # toggle for ground-truth captions or generated captions

    # Get pairs of normalized images
    df = pd.read_csv('../annotations/sim_stable.csv')
    df = df[:int(len(df) * 0.4)]
    light_imgs, dark_imgs = list(df['light_id']), list(df['dark_id'])

    # set up VADER
    analyzer = SentimentIntensityAnalyzer()

    # Analyze ground-truth captions if isGT is True, otherwise analyze generated
    if isGT:
        file = '../annotations/captions_val2014.json'
        coco = COCO(file)

        # lighter images
        aIds = coco.getAnnIds(imgIds=light_imgs)
        anns = coco.loadAnns(ids=aIds)
        captions = [x['caption'] for x in anns]
        scores = [sentiment_scores(analyzer, x) for x in captions]
        l_compound = [x[3] for x in scores]
        interval = st.t.interval(0.95, len(l_compound)-1, loc=np.mean(l_compound), scale=st.sem(l_compound))
        conf = (np.mean(l_compound)  - interval[0])
        print('Light compound score: {:.4f} +/- {:.4f}'.format(np.mean(l_compound), conf))

        # darker images
        aIds = coco.getAnnIds(imgIds=dark_imgs)
        anns = coco.loadAnns(ids=aIds)
        captions = [x['caption'] for x in anns]
        scores = [sentiment_scores(analyzer, x) for x in captions]
        d_compound = [x[3] for x in scores]
        interval = st.t.interval(0.95, len(d_compound)-1, loc=np.mean(d_compound), scale=st.sem(d_compound))
        conf = (np.mean(d_compound)  - interval[0])
        print('Dark compound score: {:.4f} +/- {:.4f}'.format(np.mean(d_compound), conf))
    else:
        diff, light, dark = [], [], []
        if len(args) < 2:
            raise Exception('Please specify the model')

        model = args[1]
        for i in range(5):
            try:
                data = json.load(open('../results/{}_{}.json'.format(model, i)))
            except:
                raise Exception('File not found')
            light_caps, dark_caps = [], []
            for i in data:
                if i['image_id'] in light_imgs: light_caps.append(i['caption'])
                if i['image_id'] in dark_imgs: dark_caps.append(i['caption'])
            light_scores = [sentiment_scores(analyzer, x) for x in light_caps]
            l_compound = [x[3] for x in light_scores]
            dark_scores = [sentiment_scores(analyzer, x) for x in dark_caps]
            d_compound = [x[3] for x in dark_scores]
            diff.append(np.mean(l_compound) - np.mean(d_compound))
            light.append(np.mean(l_compound))
            dark.append(np.mean(d_compound))
        interval = st.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=st.sem(diff))
        conf = (np.mean(diff)  - interval[0]) * 100
        print('Light compound score: {:.4f}'.format(np.mean(light)))
        print('Dark compound score: {:.4f}'.format(np.mean(dark)))
        print('Mean difference (x100) in compound scores: {:.4f} +/- {:.4f}'.format(np.mean(diff) * 100, conf))

if __name__=="__main__":
    main(sys.argv)
