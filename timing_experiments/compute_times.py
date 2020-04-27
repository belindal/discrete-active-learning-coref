import pdb
import os
import glob
import json
import numpy as np
import math

docs = glob.glob("timing_results/*.json")
#docs = glob.glob("*.json")
print(docs)


def avg(lst):
    if type(lst[0]) == list:
        total_sum = 0
        total_len = 0
        for sublist in lst:
            total_sum += sum(sublist)
            total_len += len(sublist)
        return total_sum / total_len
    return sum(lst) / len(lst)


def std(lst, lst_avg=None):
    if lst_avg is None:
        lst_avg = avg(lst)
    if type(lst[0]) != list:
        lst = [lst]
    diffs_square_sum = 0
    total_len = 0
    for sublist in lst:
        for item in sublist:
            diffs_square_sum += (item - lst_avg) ** 2
        total_len += len(sublist)
    return math.sqrt(diffs_square_sum / (total_len - 1))


discrete_times = [[] for d in range(len(docs))]
discrete_times_no_zeros = [[] for d in range(len(docs))]
pairwise_times = [[] for d in range(len(docs))]
for d, doc in enumerate(docs):
    doc_json = json.load(open(doc))
    for i, user_action in enumerate(doc_json):
        if user_action[1] == "pair_yes" or user_action[1] == "pair_no":
            pairwise_times[d].append(user_action[3] - user_action[2])
            if user_action[1] == "pair_yes":
                discrete_times[d].append(0)
        else:
            assert user_action[1] == "discrete_submit"
            discrete_times_no_zeros[d].append(user_action[3] - user_action[2])
            discrete_times[d].append(user_action[3] - user_action[2])

print("Average time to answer pairwise question: " + str(avg(pairwise_times)))
print("\n    ".join([str(avg(pairwise_times[i])) for i in range(len(pairwise_times))]))
print(" +/-" + str(std(pairwise_times)))
print("Average time to answer discrete question: " + str(avg(discrete_times_no_zeros)))
print("\n    ".join([str(avg(discrete_times_no_zeros[i])) for i in range(len(discrete_times_no_zeros))]))
print(" +/-" + str(std(discrete_times_no_zeros)))

