import json
import glob
import matplotlib.pyplot as plt
import pdb
import os

PAIRWISE_Q_TIME = 15.961803738317756
DISCRETE_Q_TIME = 15.573082474226803
DISCRETE_ONLY_Q_TIME = 28.009875
RESULTS_DIR = "/Users/belindali/allennlp"

discrete_dir = ["discrete_entropy", "pairwise_entropy"]
avg_deltas = []

for selector_fn in discrete_dir:
    pair_or_not = selector_fn[:len(selector_fn)-len("_entropy")]
    selector = 'entropy'
    deltas = {}
    for filename in glob.glob(os.path.join(RESULTS_DIR, "discrete_entropy/*_query_info.json")):
        with open(filename) as info_file:
            discrete_labels_per_doc = int(filename[len(
                os.path.join(RESULTS_DIR, "discrete_entropy") + "/"
            ):len(filename)-len("_query_info.json")])
            labels_queried = 0
            discrete_answers = 0
            query_info = json.load(info_file)
            for doc in query_info:
                labels_queried += query_info[doc]["num_queried"]
                discrete_answers += query_info[doc]["not coref"]
            if discrete_labels_per_doc == 20 and selector == 'entropy':
                print("queried: " + str((labels_queried * PAIRWISE_Q_TIME + discrete_answers * DISCRETE_Q_TIME)))
            with open(os.path.join(
                os.path.join(RESULTS_DIR, selector_fn),
                str(discrete_labels_per_doc) + "_deltas.json"
            )) as f:
                deltas[(labels_queried * PAIRWISE_Q_TIME + discrete_answers * DISCRETE_Q_TIME) / 60 / 2102] = \
                    float(f.read().strip())

    print(pair_or_not + " " + selector)
    print(deltas)
    sum_deltas = 0
    for time in deltas:
        sum_deltas += deltas[time]
    print(sum_deltas / (len(deltas) - 1))  # -1 for 0 data point
    avg_deltas.append(sum_deltas / (len(deltas) - 1))
    plt.plot(*zip(*sorted(deltas.items())), label=pair_or_not + " (" + selector + ")", marker='o')

print("")
print(avg_deltas[0] / avg_deltas[1])

plt.xlabel("Total annotation time (mins / doc)")
plt.ylabel("Average ΔF1")
plt.legend()
plt.show()

