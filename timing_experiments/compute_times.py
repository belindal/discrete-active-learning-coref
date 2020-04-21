import pdb

docs = ["annotations/belinda_discrete_labels.txt", "annotations/gabi_discrete_labels.txt"]


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


total_times = [[], []]
discrete_times = [[], []]
discrete_times_no_zeros = [[], []]
pairwise_times = [[], []]
for d, doc in enumerate(docs):
    i = 0
    with open(doc) as f:
        for line in f:
            line = line.split("\t")
            total_times[d].append(float(line[2]))
            if len(line) > 3:
                discrete_times_no_zeros[d].append(float(line[3]))
                discrete_times[d].append(float(line[3]))
            else:
                discrete_times[d].append(0)
            pairwise_times[d].append(total_times[d][i] - discrete_times[d][i])
            i += 1

print("Average time to answer pairwise question: " + str(avg(pairwise_times)))
print("    1: " + str(avg(pairwise_times[0])))
print("    2: " + str(avg(pairwise_times[1])))
print("Average time to answer discrete question: " + str(avg(discrete_times_no_zeros)))
print("    1: " + str(avg(discrete_times_no_zeros[0])))
print("    2: " + str(avg(discrete_times_no_zeros[1])))
print("pairwise / discrete: " + str(avg(discrete_times_no_zeros) / avg(pairwise_times) + 1))
print("pairwise + discrete: " + str(avg(discrete_times_no_zeros) + avg(pairwise_times)))
print("    1: " + str(avg(discrete_times_no_zeros[0]) + avg(pairwise_times[0])))
print("    2: " + str(avg(discrete_times_no_zeros[1]) + avg(pairwise_times[1])))

