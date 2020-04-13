import json
import glob
import matplotlib.pyplot as plt
import pdb

files = ["inc_closure_time_info.txt", "closure_time_info.txt"]
closure_times = {}

for i, fn in enumerate(files):
    inc = fn[:len(fn)-len("_time_info.txt")]
    if inc == 'inc_closure':
        inc = 'incremental closure'
    with open(fn) as info_file:
        closure_times = {}
        closure_time_lst = info_file.read()
        closure_time_lst = list(closure_time_lst[1:len(closure_time_lst)-1].strip().split(", "))
        closure_time_lst = [float(time) for time in closure_time_lst]
        closure_times[inc] = closure_time_lst
    print(inc)
    print(closure_times[inc])
    plt.plot(closure_times[inc], label=inc)

plt.xlabel("Number of existing annotations")
plt.ylabel("Time (sec) to compute next closure")
plt.legend()
plt.show()

