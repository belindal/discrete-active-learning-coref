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
import matplotlib.ticker as ticker
from allennlp.models.coreference_resolution.coref import ConllCorefScores
import spacy
from spacy.tokenizer import Tokenizer
from scipy.interpolate import splrep, splev

plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# Local imports

#----

# global spacy instance
nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
nlp.tokenizer = Tokenizer(nlp.vocab) # tokenize using whitespace


flatten = lambda l: [item for sublist in l for item in sublist]
f1 = lambda p, r: ((2*p*r) / (p + r)) if (p + r) else 0
pr = lambda intr, pred: 100 * (intr / pred) if pred else 100
rc = lambda intr, gold: 100 * (intr / gold) if gold else 100
NUM_OF_DOCS = 2102

def get_entities(doc, clusters):
    """
    get a list of entities (strings) indicated by the given spans and document.
    keeps the clustering information.
    """
    ent_clusts = []
    for clust in clusters:
        ent_clust = []
        for (s, e) in clust:
            ent_clust.append(doc[s : e + 1])
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

def contains_pronoun(cluster):
    """
    Returns True iff the given cluster contains a pronoun.
    """
    for mention in cluster:
        if any([w.tag_.startswith("PRP") for w in mention]):
            # Found a mention with a pronoun
            return True
    return False

def get_stats(inp_fn):
    """
    get stats from a single file.
    """
    ent_pred_cnts = []
    ent_gold_cnts = []
    ent_intr_cnts = []
    ccs = ConllCorefScores()
    jsons = [json.loads(line) for line in open(inp_fn, encoding = "utf8")]
    for doc_dict in jsons:
        cur_doc = nlp(" ".join(doc_dict["document"]))
        gold_clusts = doc_dict["gold_clusters"]
        pred_clusts = doc_dict["clusters"]

        gold_ent_clusts = get_entities(cur_doc, gold_clusts)
        pred_ent_clusts = get_entities(cur_doc, pred_clusts)

        gold_clusts_pronouns = [gold_clust
                                for gold_clust, gold_ent_clust
                                in zip(gold_clusts, gold_ent_clusts)
                                if contains_pronoun(gold_ent_clust)]

        pred_clusts_pronouns = [pred_clust
                                for pred_clust, pred_ent_clust
                                in zip(pred_clusts, pred_ent_clusts)
                                if contains_pronoun(pred_ent_clust)]

        update_avg_f1(ccs, gold_clusts_pronouns, pred_clusts_pronouns)

        pred_ents = set(map(str, flatten(pred_ent_clusts)))
        gold_ents = set(map(str, flatten(gold_ent_clusts)))
        if (not pred_ents) and (not gold_ents):
            #TODO: what's going on in these instances?
            continue

        ent_pred_cnts.append(len(pred_ents))
        ent_gold_cnts.append(len(gold_ents))
        ent_intr_cnts.append(len(pred_ents & gold_ents))

    ent_micro = micro_f1(ent_pred_cnts, ent_gold_cnts, ent_intr_cnts)
    ent_macro = macro_f1(ent_pred_cnts, ent_gold_cnts, ent_intr_cnts)
    ret_dict = {"ent_micro": ent_micro,
                "ent_macro": ent_macro,
                "pronoun_avg_f1": get_avg_f1(ccs)}

    logging.debug(f"micro = {ent_micro['f1']:.2f}, macro = {ent_macro['f1']:.2f}")

    return ret_dict


def plot_stats(model_dicts, out_fn, x_title, top_x_title, y_title, smooth, x_min, x_max, *keys):
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
    fig, ax = plt.subplots()
    for model_name, model_data in plot_data.items():
        xs, ys = model_data
        filtered_xs, filtered_ys = zip(*[(x, y) for x, y in zip(xs, ys)
                                         if (x >= x_min) and (x <= x_max)])
        ax.scatter(filtered_xs, filtered_ys)
        if smooth:
            spl = splrep(xs, ys)
            x_new = np.linspace(x_min, x_max, 300)
            y_new = splev(x_new, spl)
            xs = x_new
            ys = y_new

        ax.plot(xs, ys, label = model_name)

    # Fix plot formatting
    logging.info(f"Writing figures to file: {out_fn}")
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    secax = ax.secondary_xaxis('top', functions=(lambda t: (t * NUM_OF_DOCS) / 60,
                                                 lambda t: (t * 60) / NUM_OF_DOCS))
    secax.set_xlabel(top_x_title)
    fig.legend(loc = 'lower right', bbox_to_anchor=(0.9, 0.1))
    fig.savefig(out_fn)


def update_avg_f1(ccs, gold_clusters, pred_clusters):
    """
    update our average f1 for the predicted vs. gold clusters.
    """
    gold_clusters, mention_to_gold = ccs.get_gold_clusters(gold_clusters)
    pred_clusters, mention_to_pred = ccs.get_gold_clusters(pred_clusters)

    # update scores
    scorers = ccs.scorers

    for scorer in scorers:
        scorer.update(pred_clusters, gold_clusters,
                      mention_to_pred, mention_to_gold)

def get_avg_f1(ccs):
    """
    get our average f1 for the predicted vs. gold clusters.
    """
    scorers = ccs.scorers
    f1_scores = []
    for scorer in scorers:
        f1_scores.append(scorer.get_f1())

    avg_f1 = 100 * np.average(f1_scores)

    return avg_f1

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

        # conersions from (annotations / doc) to (min / doc)
        conversions_dict = json.load(open(model_dir / "CONVERSIONS.json"))

        for fn in fns:
            logging.debug(f"{fn}")
            x = fn.split("_")[-1].split(".")[0]
            model_dicts[model_dir][conversions_dict[x]] = get_stats(fn)

    plot_stats(model_dicts,
               out_fn / "mention_macro.png",
               "Annotation time (mins / doc)", "Total annotation time (hrs)", "Mention detection (macro f1)",
               False, 0, 70,
               "ent_macro", "f1")

    plot_stats(model_dicts,
               out_fn / "mention_micro.png",
               "Annotation time (mins / doc)", "Total annotation time (hrs)",
               "Mention detection (micro f1)",
               False, 0, 70,
               "ent_micro", "f1")

    plot_stats(model_dicts,
               out_fn / "pronoun_avg_f1.png",
               "Annotation time (mins / doc)", "Total annotation time (hrs)", "Pronoun clusters (avg f1)",
               False, 0, 70,
               "pronoun_avg_f1")


    # End
    logging.info("DONE")
