import json
import glob
import matplotlib.pyplot as plt
import pdb

by_label = False
fully_labelled_percents = [0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
fully_labelled_docs = {0: 0.5179123989590043, 0.05: 0.5293881610085179, 0.1: 0.5371043182015884, 0.2: 0.5577179591542376,
                       0.4: 0.5709037916729929, 0.5: 0.5897278850917959, 0.6: 0.6058165314663988, 0.8: 0.6105955265914443,
                       1: 0.6154417147720617}
fully_labelled_results = [fully_labelled_docs[p] for p in fully_labelled_percents]
fully_labelled_percents = [p * 100 for p in fully_labelled_percents]

results_files = [glob.glob("allennlp/percent_labels_experiment_2/*.json"), glob.glob("allennlp/percent_labels_experiment_random/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_score/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_scordiscrete/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_score_discrete_2/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_entropy/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_qbc/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_qbc_no_cluster/*.json"),
                 glob.glob("allennlp/percent_labels_experiment_entropy_no_cluster/*.json")]

annotations = [[] for i in range(len(results_files))]
percents = [[] for i in range(len(results_files))]
f1s = [[] for i in range(len(results_files))]
percent_to_f1 = [{} for i in range(len(results_files))]
for i, results_file in enumerate(results_files):
    for filename in results_file:
        if i == 0:
            file_percent = int(filename[37:len(filename)-5])
        elif i == 1:
            file_percent = int(filename[42:len(filename)-5])
        elif i == 2:
            file_percent = int(filename[41:len(filename)-5])
        elif i == 3:
            file_percent = int(filename[50:len(filename)-5])
        elif i == 4:
            file_percent = int(filename[52:len(filename)-5])
        elif i == 5:
            file_percent = int(filename[43:len(filename)-5])
        elif i == 6:
            file_percent = int(filename[39:len(filename)-5])
        elif i == 7:
            file_percent = int(filename[50:len(filename)-5])
        elif i == 8:
            file_percent = int(filename[54:len(filename)-5])
        percents[i].append(file_percent)
        with open(filename, "r") as f:
            metric = json.load(f)
            annotations[i].append(metric["best_epoch"])
            if i != 6 or i != 7:
                f1s[i].append(metric["best_validation_coref_f1"])
                percent_to_f1[i][file_percent] = metric["best_validation_coref_f1"]
            else:
                f1s[i].append(metric["ensemble_validation_coref_f1"])
                percent_to_f1[i][file_percent] = metric["ensemble_validation_coref_f1"]
    percents[i].sort()

print(percent_to_f1)
# plt.plot(percents[0], [percent_to_f1[0][percent] for percent in percents[0]], label="by-score selector v.1")
plt.plot(fully_labelled_percents, fully_labelled_results, label="random fully-labelled docs", marker='o')
plt.plot(percents[1], [percent_to_f1[1][percent] for percent in percents[1]], label="random partially-labelled docs (discrete)", marker='o')
#plt.plot(percents[2], [percent_to_f1[2][percent] for percent in percents[2]], label="by-score selector (pairwise)", marker='o')
# plt.plot(percents[3], [percent_to_f1[3][percent] for percent in percents[3]], label="by-score selector (over mentions)", marker='o')
plt.plot(percents[8], [percent_to_f1[8][percent] for percent in percents[8]], label="entropy selector (over mentions)", marker='o')
#plt.plot(percents[4], [percent_to_f1[4][percent] for percent in percents[4]], label="by-score selector (discrete 2)", marker='o')
plt.plot(percents[5], [percent_to_f1[5][percent] for percent in percents[5]], label="entropy selector (over clusters)", marker='o')
plt.plot(percents[6], [percent_to_f1[6][percent] for percent in percents[6]], label="qbc selector (over clusters)", marker='o')
plt.plot(percents[7], [percent_to_f1[7][percent] for percent in percents[7]], label="qbc selector (over mentions)", marker='o')


# for i, txt in enumerate(annotations):
#     plt.annotate(txt, (percents[i], f1s[i]))
plt.xlabel("% labels")
plt.ylabel("F1 score")
plt.legend()
plt.show()
