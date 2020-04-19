import json
import random
import os
from tqdm import tqdm
import numpy as np


CLUSTER_COLORS = [
    "red", "blue", "green", "indigo", "gray", "brown", "orange", "purple", "pink",
    "cyan", "maroon"
]

pairwise = False

doc_lengths = []

for save_preds_dir in [
    "timing_experiments_data/discrete_examples/",
]:
    for sample_idx in ['A', 'B']:
        input_file = []
        line_idx = 0
        with open(os.path.join(save_preds_dir, "sample_{}.json".format(sample_idx))) as f:
            for line in tqdm(f):
                input_file.append(json.loads(line))
                line_idx += 1
            if len(input_file) == 1:
                input_file = input_file[0]

    save_formatted_preds_dir = os.path.join(save_preds_dir, "formatted_predictions")
    os.makedirs(save_formatted_preds_dir, exist_ok=True)
    with open(os.path.join(save_formatted_preds_dir, "predictions_{}.js".format(sample_idx[0])), "w") as f:
        all_texts = []
        interactive_texts = []
        for docid, example in enumerate(tqdm(input_file)):
            text = example['tokens'].copy()
            # highlight predicted proform and antecedent
            text[int(example['proform'][0])] = """<font color="black" style="background:yellow;border:solid thick black;">[{}""".format(text[int(example['proform'][0])])
            text[int(example['proform'][1])] = """{}]</font>""".format(text[int(example['proform'][1])])

            text[int(example['antecedent'][0])] = """<font color="black" style="background:lightblue;border:solid thick blue;">[{}""".format(text[int(example['antecedent'][0])])
            text[int(example['antecedent'][1])] = """{}]</font>""".format(text[int(example['antecedent'][1])])

            for batch_idx in range(len(example['existing_span_clusters'])):
                # compute clusters
                cluster_idxs = np.array(example['existing_span_clusters'][batch_idx])
                spans = np.array(example['all_spans'][batch_idx])
                clustered_spans_mask = cluster_idxs >= 0
                cluster_idxs = cluster_idxs[clustered_spans_mask]
                spans = spans[clustered_spans_mask]

                if len(cluster_idxs) == 0:
                    continue

                # iterate over clusters
                for c in range(cluster_idxs.max()):
                    cluster = spans[cluster_idxs == c]
                    if c < len(CLUSTER_COLORS):
                        cluster_color = CLUSTER_COLORS[c]
                    else:
                        cluster_color = CLUSTER_COLORS[random.randint(0, len(CLUSTER_COLORS)-1)]
                    for span in cluster:
                        span_id = "pred{}_{}".format(docid, c)
                        text[span[0]] = """<font color="{}" class="{}" onmouseover="highlight('.{}')" onmouseout="unhighlight('.{}')" ">[[P{}]""".format(
                            cluster_color, span_id, span_id, span_id, c) + text[span[0]]
                        text[span[1]] = text[span[1]] + "]</font>"
                pred_clusters_doc_text = " ".join(text)
                all_texts.append(pred_clusters_doc_text)

                interactive_text = example['tokens'][:example['proform'][1]+1]
                for t, token in enumerate(interactive_text):
                    if t == example['proform'][0]:
                        interactive_text[t] = """<font color="black" style="background:yellow;border:solid thick black;">[{}""".format(interactive_text[t])
                    if t == example['proform'][1]:
                        interactive_text[t] = """{}]</font>""".format(interactive_text[t])
                        break
                    elif t < example['proform'][0]:
                        interactive_text[t] = """<span class="tok{}" onmouseover="highlight('.tok{}')" onmouseout="unhighlight('.tok{}')" onclick="do_click({})" >{}</span>""".format(t, t, t, t, token)
                interactive_texts.append(" ".join(interactive_text))

        json.dump(all_texts, open(os.path.join(save_formatted_preds_dir, "text_{}.json".format(sample_idx)), "w"))
        json.dump(interactive_texts, open(os.path.join(save_formatted_preds_dir, "interactive_text_{}.json".format(sample_idx)), "w"))
        
        f.write("""const dataset_idx = '{}';""".format(sample_idx[0]))
        f.write("""const pairwise = {};""".format(str(pairwise).lower()))
        f.write("""const all_texts = {};""".format(all_texts[:1000]))
        f.write("""const interactive_texts = {};""".format(interactive_texts[:1000]))

