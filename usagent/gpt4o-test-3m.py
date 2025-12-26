import os
import json

all_anno = json.load(open('/data2/RL/data/raw/microlens_ranking_data_top10_wres.json', 'r'))





# 示例使用
def get_all_result(all_anno):
    y_true = []
    y_pred = []
    all_num = 0
    correct = 0
    for anno in all_anno:
        all_num += 1
        
        if str(anno["correct_index"]) == str(anno["res"][0]):
            correct += 1
    print(correct / all_num)


get_all_result(all_anno)

