from collections import deque
from tqdm import tqdm
import numpy as np
import pandas as pd

def pref_to_rank(pref):
    return {
        a: {b: idx for idx, b in enumerate(a_pref)}
        for a, a_pref in pref.items()
    }

def get_difference(feature1, feature2):
    euclidean_loss = np.power(feature1 - feature2, 2)
    euclidean_loss = np.sqrt(np.sum(euclidean_loss))
    return euclidean_loss

def gale_shapley(*, A, B, A_pref, B_pref):
    """Create a stable matching using the
    Gale-Shapley algorithm.

    A -- set[str].
    B -- set[str].
    A_pref -- dict[str, list[str]].
    B_pref -- dict[str, list[str]].

    Output: list of (a, b) pairs.
    """
    B_rank = pref_to_rank(B_pref)
    ask_list = {a: deque(bs) for a, bs in A_pref.items()}
    pair = {}
    #
    remaining_A = set(A)
    while len(remaining_A) > 0:
        a = remaining_A.pop()
        b = ask_list[a].popleft()
        if b not in pair:
            pair[b] = a
        else:
            a0 = pair[b]
            b_prefer_a0 = B_rank[b][a0] < B_rank[b][a]
            if b_prefer_a0:
                remaining_A.add(a)
            else:
                remaining_A.add(a0)
                pair[b] = a
    #
    return [(a, b) for b, a in pair.items()]

def main(args):
    if len(args) < 3:
        raise Exception("Need two input files of the extracted features")
        
    df = pd.read_csv('../annotations/images_val2014.csv')
    light_id = list(df[df['bb_skin'] == 'Light']['id'])
    dark_id = list(df[df['bb_skin'] == 'Dark']['id'])

    l_feat = np.load(args[1])
    d_feat = np.load(args[2])

    l_pref, d_pref = {}, {}

    for index_i, i in enumerate(dark_id):
        dist, ids  = [], []
        for index_j, j in enumerate(light_id):
            dist.append(get_difference(d_feat[index_i], l_feat[index_j]))
            ids.append(j)
        Z = [x for _, x in sorted(zip(dist, ids))]
        d_pref[i] = Z

    for index_i, i in enumerate(light_id):
        dist, ids  = [], []
        for index_j, j in enumerate(dark_id):
            dist.append(get_difference(l_feat[index_i], d_feat[index_j]))
            ids.append(j)
        Z = [x for _, x in sorted(zip(dist, ids))]
        l_pref[i] = Z

    pairs = gale_shapley(
        B=set(light_id),
        A=set(dark_id),
        B_pref=l_pref,
        A_pref=d_pref,
    )

    dark = [i[0] for i in pairs]
    light = [i[1] for i in pairs]

    d = {'dark_id': dark, 'light_id': light}
    df = pd.DataFrame(data=d)
    df.to_csv('sim_stable.csv', index=False)

if __name__=="__main__":
    main(sys.argv)
