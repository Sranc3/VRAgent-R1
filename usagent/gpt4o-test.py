import os
import json

all_anno = json.load(open('/data2/RL/data/raw/microlens_like_data_wres.json', 'r'))


def calculate_metrics(y_true, y_pred):
    # 初始化真阳性、假阳性、真阴性、假阴性
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # 遍历两个列表
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    # 计算准确率
    if len(y_true) > 0:
        acc = (tp + tn) / len(y_true)
    else:
        acc = 0

    # 计算精确率
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # 计算召回率
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # 计算 F1 分数
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, acc, precision, recall


# 示例使用
def get_all_result(all_anno):
    y_true = []
    y_pred = []
    for anno in all_anno:
        y_true.append(anno["label"])
        y_pred.append(int(anno["res"]))
    return y_true, y_pred

y_true, y_pred = get_all_result(all_anno)

f1, acc, precision, recall = calculate_metrics(y_true, y_pred)
print(f"F1 Score: {f1}")
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
    