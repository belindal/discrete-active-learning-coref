import json
import random
import os
from tqdm import tqdm

CLUSTER_COLORS = [
    "red", "blue", "green", "indigo", "gray", "yellow", "brown", "orange", "purple", "pink",
    "cyan", "maroon"
]

for save_preds_dir in ["discrete_entropy", "pairwise_entropy"]:
    for num_labels in [0, 20, 80, 100, 200]:
        input_file = []
        with open(os.path.join(save_preds_dir, "predictions_{}.json".format(num_labels))) as f:
            for line in f:
                input_file.append(json.loads(line))

        save_formatted_preds_dir = os.path.join(save_preds_dir, "formatted_predictions")
        os.makedirs(save_formatted_preds_dir, exist_ok=True)
        with open(os.path.join(save_formatted_preds_dir, "predictions_{}.html".format(num_labels)), "w") as f:
            f.write("""<!DOCTYPE html>
<html>
  <body>
""")
            for docid, example in enumerate(tqdm(input_file)):
                text = example['document'].copy()
                for i, cluster in enumerate(example['clusters']):
                    if i < len(CLUSTER_COLORS):
                        cluster_color = CLUSTER_COLORS[i]
                    else:
                        cluster_color = CLUSTER_COLORS[random.randint(0, len(CLUSTER_COLORS)-1)]
                    for span in cluster:
                        span_id = "pred{}_{}".format(docid, i)
                        text[span[0]] = """<font color="{}" class="{}" onmouseover="highlight('.{}')" onmouseout="unhighlight('.{}')" onclick="do_click('.{}')">[[P{}]""".format(
                            cluster_color, span_id, span_id, span_id, span_id, i) + text[span[0]]
                        text[span[1]] = text[span[1]] + "]</font>"
                pred_clusters_doc_text = " ".join(text)
                f.write("    <p>Predicted</p>\n")
                f.write("    <p>{}</p>\n".format(pred_clusters_doc_text))

                text = example['document']
                for i, cluster in enumerate(example['gold_clusters']):
                    if i < len(CLUSTER_COLORS):
                        cluster_color = CLUSTER_COLORS[i]
                    else:
                        cluster_color = CLUSTER_COLORS[random.randint(0, len(CLUSTER_COLORS)-1)]
                    for span in cluster:
                        span_id = "gold{}_{}".format(docid, i)
                        text[span[0]] = """<font color="{}" class="{}" onmouseover="highlight('.{}')" onmouseout="unhighlight('.{}')" onclick="do_click('.{}')">[[G{}]""".format(
                            cluster_color, span_id, span_id, span_id, span_id, i) + text[span[0]]
                        text[span[1]] = text[span[1]] + "]</font>"

                gold_clusters_doc_text = " ".join(text)
                f.write("    <p>Gold</p>\n")
                f.write("    <p>{}</p>\n".format(gold_clusters_doc_text))
                f.write("    <hr>\n")

            f.write("""
  </body>
  <script>
    function do_click(x) {
        var el, i;
        el = document.querySelectorAll(x);
        for (i = 0; i < el.length; i++) {
          if (el[i].style.border === "thick solid rgb(0, 0, 0)") {
            el[i].style.border = "none";
          } else {
            el[i].style.border = "thick solid rgb(0, 0, 0)";
          }
        }
    }

    function highlight(x) {
        var el, i;
        el = document.querySelectorAll(x);
        for (i = 0; i < el.length; i++) {
          el[i].style.background = "yellow";
        }
    }

    function unhighlight(x) {
        var el, i;
        el = document.querySelectorAll(x);
        for (i = 0; i < el.length; i++) {
          el[i].style.background = "white";
        }
    }
  </script>
</html>""")

