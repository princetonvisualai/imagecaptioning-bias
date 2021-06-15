from eval import COCOEvalCap
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import json, sys
import scipy.stats as st

# Code for evaluating captions using COCO evaluation server metrics
def compute_average(scores):
    b4, cider, spice = [], [], []
    scores_total, scores_sum = [], []
    for s in scores:
        b4.append(s['Bleu_4'])
        cider.append(s['CIDEr'])
        spice.append(s['SPICE']['All']['f'])
    scores = np.stack([b4, cider, spice])
    score = (np.mean(scores, axis=1))
    return score

def print_scores(overall):
    bleu = [x['Bleu_4'] for x in overall if 'Bleu_4' in x]
    cider = [x['CIDEr'] for x in overall if 'CIDEr' in x]
    spice = [x['SPICE'] for x in overall if 'SPICE' in x]
    bleu, cider, spice = np.mean(bleu), np.mean(cider), np.mean(spice)
    print('BLEU-4: {:.4f} | CIDEr: {:.4f} | SPICE: {:.4f}'.format(bleu, cider, spice))

def print_diff(diff):
    stacked_diff = np.stack(diff)
    scorers = ['BLEU', 'CIDEr', 'SPICE']
    for i in range(3):
        interval = st.t.interval(0.95, len(stacked_diff[:,i])-1, loc=np.mean(stacked_diff[:,i]), \
            scale=st.sem(stacked_diff[:,i]))
        conf = np.mean(stacked_diff[:,i]) - interval[0]
        print('{}: {:.4f} +/- {:.4f}'.format(scorers[i], np.mean(stacked_diff[:,i]), conf))

def main(model):
    # create coco object and cocoRes object
    annFile = '../imagecaptioning/annotations/captions_val2014.json'
    instFile = '../imagecaptioning/annotations/instances_val2014.json'
    coco = COCO(annFile)
    cocoInst = COCO(instFile)

    df = pd.read_csv('../imagecaptioning/annotations/images_val2014.csv')
    light_imgs = list(df.loc[df['bb_skin'] == 'Light']['id'])
    dark_imgs =  list(df.loc[df['bb_skin'] == 'Dark']['id'])

    light, dark, diff, overall = [], [], [], []
    for i in range(5):
        try:
            resFile = '../imagecaptioning/results/{}_{}.json'.format(model, i)
        except:
            raise Exception('Cannot find file')
        with open(resFile) as f:
            data = json.load(f)
        ids = [x['image_id'] for x in data]
        cocoRes = coco.loadRes(resFile)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = ids
        cocoEval.evaluate()
        overall.append(cocoEval.eval)
        scores = cocoEval.evalImgs
        li = [eva for eva in scores if int(eva['image_id']) in light_imgs]
        li_score = compute_average(li)
        da = [eva for eva in scores if int(eva['image_id']) in dark_imgs]
        da_score = compute_average(da)
        delta = li_score - da_score

        light.append(li_score)
        dark.append(da_score)
        diff.append(delta)
    # print results
    print('Overall Scores')
    print_scores(overall)
    print('\n')
    print('Difference in Scores')
    print_diff(diff)

if __name__=="__main__":
    if len(sys.argv) < 2:
        raise Exception('Enter name of model to evaluate')
    main(sys.argv[1])
