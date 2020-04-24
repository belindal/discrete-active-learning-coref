import json
import glob
import os
import matplotlib.pyplot as plt
import pdb

plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

RESULTS_DIR = "/Users/belindali/discrete_al_results"

files = ["inc_closure_time_info.txt", "closure_time_info.txt"]
closure_times = {}

for i, fn in enumerate(files):
    inc = fn[:len(fn)-len("_time_info.txt")]
    if inc == 'inc_closure':
        inc = 'incremental closure'
    with open(os.path.join(RESULTS_DIR, fn)) as info_file:
        closure_times = {}
        closure_time_lst = json.loads(info_file.read())
        #closure_time_lst = list(closure_time_lst[1:len(closure_time_lst)-1].strip().split(", "))
        #closure_time_lst = [float(time) for time in closure_time_lst]
        closure_times[inc] = closure_time_lst
    print(inc)
    print(closure_times[inc])
    plt.plot(closure_times[inc], label=inc)

plt.xlabel("Number of existing annotations")
plt.ylabel("Time (sec) to compute next closure")
plt.legend()
plt.show()

