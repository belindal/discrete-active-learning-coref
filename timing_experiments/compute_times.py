import pdb
import os
import glob
import json
import numpy as np
import math

docs = glob.glob("results/*.json")
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
    # print(str(sum(lst)) + " / " + str(len(lst)))
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


#total_times = [[], []]
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
        #total_times[d].append(pairwise_times[d][i] + discrete_times[d][i])
        #for line in f:
        #    line = line.split("\t")
        #    total_times[d].append(float(line[2]))
        #    if len(line) > 3:
        #        discrete_times_no_zeros[d].append(float(line[3]))
        #        discrete_times[d].append(float(line[3]))
        #    else:
        #        discrete_times[d].append(0)
        #    pairwise_times[d].append(total_times[d][i] - discrete_times[d][i])
        #    i += 1

print("Average time to answer pairwise question: " + str(avg(pairwise_times)))
print("\n    ".join([str(avg(pairwise_times[i])) for i in range(len(pairwise_times))]))
print(" +/-" + str(std(pairwise_times)))
print("Average time to answer discrete question: " + str(avg(discrete_times_no_zeros)))
print("\n    ".join([str(avg(discrete_times_no_zeros[i])) for i in range(len(discrete_times_no_zeros))]))
print(" +/-" + str(std(discrete_times_no_zeros)))
print("pairwise / discrete: " + str(avg(discrete_times_no_zeros) / avg(pairwise_times) + 1))
print("pairwise + discrete: " + str(avg(discrete_times_no_zeros) + avg(pairwise_times)))
print("    1: " + str(avg(discrete_times_no_zeros[0]) + avg(pairwise_times[0])))
print("    2: " + str(avg(discrete_times_no_zeros[1]) + avg(pairwise_times[1])))

