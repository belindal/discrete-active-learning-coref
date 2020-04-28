import json
import glob
import matplotlib.pyplot as plt
import pdb
import os

PAIRWISE_Q_TIME = 15.961803738317756
DISCRETE_Q_TIME = 15.573082474226803
DISCRETE_ONLY_Q_TIME = 28.009875
RESULTS_DIR = "/Users/belindali/discrete_al_results"
GRAPH_TIME_CUTOFF = 75

plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
fig, ax = plt.subplots()

discrete_dir = ["discrete_entropy", "pairwise_entropy"]
avg_deltas = []


def mpd_to_hrs(x):
    return x * 2102 / 60


def hrs_to_mpd(x):
    return x * 60 / 2102


for selector_fn in discrete_dir:
    pair_or_not = selector_fn[:len(selector_fn)-len("_entropy")]
    selector = 'entropy'
    deltas = {}
    for filename in glob.glob(os.path.join(RESULTS_DIR, "{}/*_query_info.json".format(selector_fn))):
        with open(filename) as info_file:
            discrete_labels_per_doc = int(filename[len(
                os.path.join(RESULTS_DIR, selector_fn) + "/"
            ):len(filename)-len("_query_info.json")])
            labels_queried = 0
            discrete_answers = 0
            query_info = json.load(info_file)
            for doc in query_info:
                labels_queried += query_info[doc]["num_queried"]
                if pair_or_not == "discrete":
                    discrete_answers += query_info[doc]["not coref"]
            time = (labels_queried * PAIRWISE_Q_TIME + discrete_answers * DISCRETE_Q_TIME) / 60 / 2102
            if time > GRAPH_TIME_CUTOFF:
                continue
            with open(os.path.join(
                os.path.join(RESULTS_DIR, selector_fn),
                str(discrete_labels_per_doc) + "_deltas.json"
            )) as f:
                deltas[time] = float(f.read().strip()) * 100

    print(pair_or_not + " " + selector)
    print(deltas)
    sum_deltas = 0
    for time in deltas:
        sum_deltas += deltas[time]
    print(sum_deltas / (len(deltas) - 1))  # -1 for 0 data point
    avg_deltas.append(sum_deltas / (len(deltas) - 1))
    ax.plot(*zip(*sorted(deltas.items())), label=pair_or_not + " (" + selector + ")", marker='o')

print("")
print(avg_deltas[0] / avg_deltas[1])

ax.set_xlabel("Annotation time (mins / doc)")
ax.set_ylabel("Average ΔF1")

secax = ax.secondary_xaxis('top', functions=(mpd_to_hrs, hrs_to_mpd))
secax.set_xlabel('Total annotation time (hrs)', fontsize=14)

plt.legend()
plt.show()

