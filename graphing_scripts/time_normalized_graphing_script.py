import json
import glob
import matplotlib.pyplot as plt
import pdb
import os

PAIRWISE_Q_TIME = 15.961803738317756
DISCRETE_Q_TIME = 15.573082474226803
DISCRETE_ONLY_Q_TIME = 28.009875
RESULTS_DIR = "/Users/belindali/allennlp"
GRAPH_TIME_CUTOFF = 75


pairwise_dir = ["pairwise_entropy"] #, "allennlp/pairwise_random/"]
discrete_dir = ["discrete_entropy",
                "discrete_qbc3",
                "discrete_score",
                "discrete_random",
                ]
    #, "allennlp/discrete_entropy_no_link_penalties/",
                # "allennlp/discrete_entropy_no_clusters/", "allennlp/discrete_entropy_no_inter_closure/"]
discrete_dir_2 = ["discrete_random2"]
discrete_dir_2_map = {'0': 0, '20': 35345, '40': 68102, '60': 98773, '80': 128481, '100': 156805, '120': 183362,
                      '140': 206731, '160': 228176, '180': 249807, '200': 267001, '140_actual': 288114,
                      '160_actual': 318120, '180_actual': 349339, '200_actual': 373400}

for selector_fn in discrete_dir:
    selector = selector_fn[len("discrete_"):len(selector_fn)]
    discrete_results = {}
    only_discrete_results = {}
    time_saved = []
    selector_fn = os.path.join(RESULTS_DIR, selector_fn)
    for filename in glob.glob(os.path.join(selector_fn, "*_query_info.json")):
        with open(filename) as info_file:
            discrete_labels_per_doc = int(filename[len(selector_fn+'/'):len(filename)-len("_query_info.json")])
            labels_queried = 0
            discrete_answers = 0
            query_info = json.load(info_file)
            for doc in query_info:
                labels_queried += query_info[doc]["num_queried"]
                discrete_answers += query_info[doc]["not coref"]
            #if labels_queried > 0:
            #    print("percent discrete: " + str(discrete_answers / labels_queried))
            #if discrete_labels_per_doc == 20 and selector == 'entropy':
            #    print("queried: " + str((labels_queried * PAIRWISE_Q_TIME + discrete_answers * DISCRETE_Q_TIME)))
            with open(os.path.join(selector_fn, str(discrete_labels_per_doc) + ".json")) as f:
                metric = "best_validation_coref_f1"
                if selector[:3] == 'qbc':
                    metric = "ensemble_validation_coref_f1"
                    if discrete_labels_per_doc == 140:
                        labels_queried = 235656
                        discrete_answers = 183732
                model_score = json.load(f)[metric]
                discrete_time = (labels_queried * PAIRWISE_Q_TIME + discrete_answers * DISCRETE_Q_TIME) / 60 / 2102
                only_discrete_time = labels_queried * DISCRETE_ONLY_Q_TIME / 60 / 2102
                if discrete_time < GRAPH_TIME_CUTOFF:
                    discrete_results[discrete_time] = model_score
                if only_discrete_time < GRAPH_TIME_CUTOFF:
                    only_discrete_results[only_discrete_time] = model_score
                time_saved.append(only_discrete_time - discrete_time)

    print("discrete " + selector)
    print("time saved: " + str(time_saved))
    print(sum(time_saved) / len(time_saved))
    print(discrete_results)
    if selector[:3] == 'qbc':
        plt.plot(*zip(*sorted(discrete_results.items())),
                 label="discrete (" + selector[:3] + ", " + selector[3:] + " models)", marker='o', color='C2')
    elif selector == 'score':
        plt.plot(*zip(*sorted(discrete_results.items())),
                 label="discrete (LCC/MCU)", marker='o', color='C3')
    elif selector == 'random':
        plt.plot(*zip(*sorted(discrete_results.items())), label="discrete (" + selector + ", partially labelled)",
                 marker='o', color='k', alpha=0.5)
    else:
        #plt.plot(*zip(*sorted(only_discrete_results.items())),
        #         label="discrete ONLY (" + selector + ")", marker='o', color='C2')
        plt.plot(*zip(*sorted(discrete_results.items())), label="discrete (" + selector + ")", marker='o')

for selector_fn in discrete_dir_2:
    selector = selector_fn[len("discrete_"):len(selector_fn)]
    selector_fn = os.path.join(RESULTS_DIR, selector_fn)
    discrete_results = {}
    for filename in glob.glob(os.path.join(selector_fn, "*.json")):
        with open(filename) as info_file:
            discrete_labels_per_doc = filename[len(selector_fn+"/"):len(filename)-len(".json")]
            labels_queried = discrete_dir_2_map[discrete_labels_per_doc]
            with open(os.path.join(selector_fn, str(discrete_labels_per_doc) + ".json")) as f:
                metric = "best_validation_coref_f1"
                time_spent = DISCRETE_ONLY_Q_TIME * labels_queried / 60 / 2102
                value = json.load(f)[metric]
                if time_spent < 70:
                    discrete_results[time_spent] = value 

    print("discrete random 2")
    print(discrete_results)
    plt.plot(*zip(*sorted(discrete_results.items())), '--', label="discrete (" + selector[:len(selector)-1] +
                                                                  ", fully labelled)", marker='o',
             color='k', alpha=0.5)

for selector_fn in pairwise_dir:
    selector = selector_fn[len("pairwise_"):len(selector_fn)]
    selector_fn = os.path.join(RESULTS_DIR, selector_fn)
    pairwise_results = {}
    for filename in glob.glob(os.path.join(selector_fn, "*_query_info.json")):
        with open(filename) as info_file:
            pairwise_labels_per_doc = int(filename[len(selector_fn+"/"):len(filename)-len("_query_info.json")])
            labels_queried = 0
            query_info = json.load(info_file)
            for doc in query_info:
                labels_queried += query_info[doc]["num_queried"]
            with open(os.path.join(selector_fn, str(pairwise_labels_per_doc) + ".json")) as f:
                time_spent = (labels_queried * PAIRWISE_Q_TIME) / 60 / 2102
                if time_spent < GRAPH_TIME_CUTOFF:
                    pairwise_results[time_spent] = json.load(f)["best_validation_coref_f1"]

    print("pair " + selector)
    print(pairwise_results)
    plt.plot(*zip(*sorted(pairwise_results.items())),
             '--',
             label="pairwise (" + selector + ")", marker='o', color='C1',
             alpha=0.5)

#plt.plot(*zip(*sorted(discrete_qbc_results.items())), label="discrete (3-model qbc over clusters)", marker='o')
plt.xlabel("Total annotation time (mins / doc)")
plt.ylabel("F1 score")
plt.legend()
plt.show()

