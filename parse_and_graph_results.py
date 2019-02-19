import matplotlib.pyplot as plt
import math

N = 70
plot_epochs = True

train_f1_values = []
dev_f1_values = []
i = 0
with open('m2_results.txt', 'r') as f:
    for line in f:
        line_parsed = line.split()
        if i % 3 == 0:
            # is a train set result
            assert(line_parsed[4] == 'coref_f1:')
            train_f1_values.append(float(line_parsed[5][:len(line_parsed[5]) - 1]))
        if i % 3 == 1:
            # is a dev set result
            assert(line_parsed[12] == '343/343')
            assert(line_parsed[4] == 'coref_f1:')
            dev_f1_values.append(float(line_parsed[5][:len(line_parsed[5]) - 1]))
        i += 1

print(dev_f1_values)
# '''
if not plot_epochs:
    Xs = [700 + i * N for i in range(len(dev_f1_values))]
    xlabel = '# examples'
else:
    Xs = [i for i in range(len(dev_f1_values))]
    plt.axvline(x=math.ceil((2802 - 700) / N), label='Full Training Set')
    xlabel = 'epoch'
plt.plot(Xs, dev_f1_values, label='Dev set')
plt.plot(Xs, train_f1_values, label='Train set')
plt.xlabel(xlabel)
plt.ylabel('F1 scores')
plt.title('Training Curve (N = ' + str(N) + ')')
plt.legend()
plt.show()
# '''

