import pdb
import time
import argparse
from getch import getch
import textwrap
import os

parser = argparse.ArgumentParser(description='Run setting')
parser.add_argument('policy_type',
                    type=str,
                    default='pairwise',
                    help='how to poll the user (pairwise or discrete)')
parser.add_argument('input_document',
                    type=str,
                    help='input document containing examples to label')
parser.add_argument('output_document',
                    type=str,
                    help='output document in which to write labeled and timed examples')

wrapper = textwrap.TextWrapper(width=100)


args = vars(parser.parse_args())
QUERY_TYPE = args['policy_type']
doc_name = args['input_document']
user_doc_name = args['output_document']
#user_doc_name = "user_" + doc_name[:len(doc_name) - 13] + "_labels.txt"

if QUERY_TYPE != "pairwise" and QUERY_TYPE != "discrete":
    raise ValueError("bad argument, should pass in either 'pairwise' or 'discrete'")


examples = []
user_answers = []
new_ants = {}
time_per_example = []
discrete_time_per_example = {}

with open(doc_name, 'r', encoding='unicode_escape') as f:
    for line in f:
        examples.append(line)

try:
    i = 0
    with open(user_doc_name, 'r') as wf:
        for line in wf:
            line = line.strip().split('\t')
            user_answers.append(bool(line[0] == 'True'))
            assert len(line) >= 3
            if line[0] != 'True':
                new_ants[i] = str(line[1])
            time_per_example.append(float(line[2]))
            if len(line) >= 4:
                discrete_time_per_example[i] = float(line[3])
            i += 1
    num_already_queried = len(user_answers)
except:
    num_already_queried = 0

i = num_already_queried
try:
    while i < len(examples):
        os.system("clear")
        print('\n' + textwrap.indent(text=wrapper.fill(examples[i].strip()), prefix='    ', predicate=lambda line: True) + '\n')
        start_time = time.time()
        discrete_start_time = 0
        print("\x1b[0;31;40mAre these two coreferent? y/[n] ('q' to quit with save, 'p' to go back to previous example):\x1b[0m ")
        val = getch()
        print(val)
        if val.startswith('y') or val.startswith('Y'):
            if i >= len(user_answers):
                user_answers.append(True)
            else:
                user_answers[i] = True
        elif val.startswith('q') or val.startswith('Q'):
            break
        elif val.startswith('p') or val.startswith('P'):
            i -= 2
            if i < 0:
                i = -1
        else:
            if i >= len(user_answers):
                user_answers.append(False)
            else:
                user_answers[i] = False
            if QUERY_TYPE == "discrete":
                discrete_start_time = time.time()
                new_item = input("\x1b[0;31;40mWhat is the *first* appearance of the entity that the white-highlighted text refers to? (copy from document):\x1b[0m \n")
                new_ants[i] = new_item
        end_time = time.time()
        if val.startswith('y') or val.startswith('Y') or val.startswith('n') or val.startswith('N'):
            if QUERY_TYPE == "discrete" and discrete_start_time != 0:
                if i not in discrete_time_per_example:
                    discrete_time_per_example[i] = 0
                discrete_time_per_example[i] += (end_time - discrete_start_time)
            if i >= len(time_per_example):
                time_per_example.append(end_time - start_time)
            else:
                time_per_example[i] += (end_time - start_time)
        print()
        i += 1
except:
    # do nothing
    print()

wf = open(user_doc_name, 'w')
for i, answer in enumerate(user_answers):
    wf.write(str(answer) + "\t")
    if i in new_ants:
        wf.write(new_ants[i])
    wf.write("\t" + str(time_per_example[i]))
    if i in discrete_time_per_example:
        wf.write("\t" + str(discrete_time_per_example[i]))
    wf.write("\n")
wf.close()

