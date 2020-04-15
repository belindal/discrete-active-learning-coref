""" Usage:
    <file-name> [--in=INPUT_FILE] [--out=OUTPUT_FILE] [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt

# Local imports

#----


flatten = lambda l: [item for sublist in l for item in sublist]
f1 = lambda p, r: ((2*p*r) / (p + r)) if (p + r) else 0
pr = lambda intr, pred: (intr / pred) if pred else 1
rc = lambda intr, gold: (intr / gold) if gold else 1

# conersions from (annotations / doc) to (min / doc)
TIME_CONVERSIONS = {0: 0.0, 200: 71.91751439779348, 100: 41.96964765646017, 80: 34.27887468868257, 20: 9.38844897854418}

def get_entities(doc, clusters):
    """
    get a list of entities (strings) indicated by the given spans and document.
    keeps the clustering information.
    """
    ent_clusts = []
    for clust in clusters:
        ent_clust = []
        for (s, e) in clust:
            ent_clust.append(" ".join(doc[s : e + 1]))
        ent_clusts.append(ent_clust)
    return ent_clusts


def micro_f1(pred_cnts, gold_cnts, intr_cnts):
    """
    get micro f1, p, r.
    """
    numerator = sum(intr_cnts)
    micro_p = pr(numerator, sum(pred_cnts))
    micro_r = rc(numerator, sum(gold_cnts))
    micro_f = f1(micro_p, micro_r)

    ret_dict =  {"f1": micro_f,
                 "p": micro_p,
                 "r": micro_r}
    return ret_dict

def macro_f1(pred_cnts, gold_cnts, intr_cnts):
    """
    get micro f1, p, r.
    """
    ps = []
    rs = []
    fs = []
    for pred_cnt, gold_cnt, intr_cnt in zip(pred_cnts, gold_cnts, intr_cnts):
        numerator = intr_cnt
        p = pr(numerator, pred_cnt)
        r = rc(numerator, gold_cnt)
        f = f1(p, r)
        ps.append(p)
        rs.append(r)
        fs.append(f)

    macro_f, macro_p, macro_r = map(np.average, [fs, ps, rs])
    ret_dict =  {"f1": macro_f,
                 "p": macro_p,
                 "r": macro_r}
    return ret_dict


def get_stats(inp_fn):
    """
    get stats from a single file.
    """
    ent_pred_cnts = []
    ent_gold_cnts = []
    ent_intr_cnts = []
    jsons = [json.loads(line) for line in open(inp_fn, encoding = "utf8")]
    for doc_dict in jsons:
        cur_doc = doc_dict["document"]
        gold_clusts = get_entities(cur_doc, doc_dict["gold_clusters"])
        pred_clusts = get_entities(cur_doc, doc_dict["clusters"])
        pred_ents = set(flatten(pred_clusts))
        gold_ents = set(flatten(gold_clusts))
        if (not pred_ents) and (not gold_ents):
            #TODO: what's going on in these instances?
            continue

        ent_pred_cnts.append(len(pred_ents))
        ent_gold_cnts.append(len(gold_ents))
        ent_intr_cnts.append(len(pred_ents & gold_ents))

    ent_micro = micro_f1(ent_pred_cnts, ent_gold_cnts, ent_intr_cnts)
    ent_macro = macro_f1(ent_pred_cnts, ent_gold_cnts, ent_intr_cnts)
    ret_dict = {"ent_micro": ent_micro, "ent_macro": ent_macro}

    logging.debug(f"micro = {ent_micro['f1']:.2f}, macro = {ent_macro['f1']:.2f}")

    return ret_dict


def plot_stats(model_dicts, out_fn, x_title, y_title, *keys):
    """
    Plot a given stat across given models.
    """
    # Collect data to plot
    plot_data = {}
    for model_name, model_stats in model_dicts.items():
        model_name_str = str(model_name)
        sorted_xs = sorted(list(model_stats.keys()))
        plot_data[model_name_str] = [sorted_xs]
        ys = []
        for x in sorted_xs:
            cur_item = model_stats[x]
            # traverse down dictionary to find values
            for key in keys:
                cur_item = cur_item[key]
            ys.append(cur_item)
        plot_data[model_name_str].append(ys)

    # Plot
    fig, ax1 = plt.subplots()
    for model_name, model_data in plot_data.items():
        xs, ys = model_data

        # Convert to min / doc
        xs = [TIME_CONVERSIONS[x] for x in xs]

        ax1.plot(xs, ys, label = model_name)
        ax1.scatter(xs, ys)

    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_title)
    fig.legend()
    fig.savefig(out_fn)

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fns = [Path(inp_fn) for inp_fn in args["--in"].split(",")]
    out_fn = Path(args["--out"]) if args["--out"] else None

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Start computation
    model_dicts = defaultdict(dict)
    for model_dir in tqdm(inp_fns):
        logging.info(f"Analyzing {model_dir}")
        fns = glob(str(model_dir / "predictions_*.json"))
        for fn in fns:
            logging.debug(f"{fn}")
            x = int(fn.split("_")[-1].split(".")[0])
            model_dicts[model_dir][x] = get_stats(fn)

    plot_stats(model_dicts,
               out_fn / "ent_micro.png",
               "Annotation time [mins / doc]", "Entity extraction [micro f1]",
               "ent_micro", "f1")

    # End
    logging.info("DONE")
