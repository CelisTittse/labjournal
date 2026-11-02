import pandas as pd
import random
from tqdm import tqdm
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def create_ms_se_edges (sub_df: pd.DataFrame) -> list[tuple]:
    ms_se_list = []
    for ms, ses_str in zip(sub_df['shelfmark'], sub_df['certain']):
        if isinstance(ses_str, str):
            ses = [se for se in ses_str.split(', ') if se and isinstance(se, str)]

            for se in ses:
                ms_se_list.append([ms, se])

    return ms_se_list

def swap_sermon_between_mss(ms_se_l: list[tuple]) -> list[tuple]:
    idx = range(len(ms_se_l))
    # pick random two ms-se edges
    i1, i2 = random.sample(idx, 2)
    # swap sermon between mss
    ms_se_l[i1][1], ms_se_l[i2][1] = ms_se_l[i2][1], ms_se_l[i1][1]

    return ms_se_l

# create ms-se edgelist, undirected, without differentation of century
def create_se_se(sub_df: pd.DataFrame, randomise_perc: float=0) -> list[tuple]:
    if not (0 <= randomise_perc <= 1):
        raise ValueError('randomise_perc must be between 0 and 1')

    ms_se_list = create_ms_se_edges(sub_df=sub_df)

    if not randomise_perc or randomise_perc > 0:
        # iterate over n edges * percentage swaps to be performed
        for i in range(int(len(ms_se_list) * randomise_perc)):
            ms_se_list = swap_sermon_between_mss(ms_se_l=ms_se_list)

    # make se-se network
    se_se = []
    for ms1, se1 in ms_se_list:
        for ms2, se2 in ms_se_list:
            if se1 > se2 and ms1 == ms2:
                se_se.append((se1, se2))

    return se_se


# os.getcwd()

root_in = './data/data_raw/'
ms_se_path = f"{root_in}sf_mss_okt_25.xlsx"
root_out = "./data/data_deriv"
save_dataframes = False

msSeDf = pd.read_excel(ms_se_path)

print(msSeDf.columns)

PLOT_BOXPLOT = False
NUMBER_ITERATIONS = 1000
RANDOMISE_PERCENTAGE = 0.2
THRESHOLD_ALPHA = 0.05

for century, submsSeDf in msSeDf.groupby('century'):
    if century != 7: continue

    se_se_obs = create_se_se(sub_df=submsSeDf, randomise_perc=0)

    randomEdgeDist = {se_se : [] for se_se in set(se_se_obs)}

    for i in tqdm(range(NUMBER_ITERATIONS)):
        randomSeSeEdges = create_se_se(sub_df=submsSeDf, randomise_perc=RANDOMISE_PERCENTAGE)

        relv_edges = [edge for edge in randomSeSeEdges if edge in se_se_obs]

        edge_count_dict = dict(Counter(randomSeSeEdges))

        for seSe in randomEdgeDist.keys():
            random_edge_count = edge_count_dict.get(seSe, 0)
            randomEdgeDist[seSe].append(random_edge_count)

    se_se_weight_obs = dict(Counter(se_se_obs))

    print(f'N unique edges: {len(randomEdgeDist.keys())}, edges :{len(se_se_obs)}')


    sign_ses = []

    for obs_sese, obs_degree in se_se_weight_obs.items():
        random_degree = np.array(randomEdgeDist[obs_sese])

        random_tail_perc = (random_degree <= obs_degree).sum() / len(random_degree)

        print(obs_sese, random_tail_perc)

        if random_tail_perc < THRESHOLD_ALPHA * 2: # one-tail so time 2
            sign_ses.append(obs_sese)

    print(sign_ses)

    if PLOT_BOXPLOT:
        keys = list(randomEdgeDist.keys())[:10]
        plt.boxplot([randomEdgeDist[k] for k in keys])
        plt.scatter(
            range(1, len(keys) + 1),
            [se_se_weight_obs.get(k) for k in keys],
            color='red'
        )
        plt.show()

    break