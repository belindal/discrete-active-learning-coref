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

for save_preds_dir in ["./"]:
    for num_labels in [20]:
        input_file = []
        line_idx = 0
        with open(os.path.join(save_preds_dir, "discrete_examples.json")) as f:
            for line in tqdm(f):
                # if len(json.loads(line)['tokens']) < 300:
                input_file.append(json.loads(line))
                line_idx += 1
                if line_idx > 10:
                    break

        save_formatted_preds_dir = os.path.join(save_preds_dir, "formatted_predictions")
        os.makedirs(save_formatted_preds_dir, exist_ok=True)
        with open(os.path.join(save_formatted_preds_dir, "predictions_{}.js".format(num_labels)), "w") as f:
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

            json.dump(all_texts, open(os.path.join(save_formatted_preds_dir, "text_{}.json".format(num_labels)), "w"))
            json.dump(interactive_texts, open(os.path.join(save_formatted_preds_dir, "interactive_text_{}.json".format(num_labels)), "w"))

            f.write("""
var uid = "1234";
var curr_example = 0;
var last_display_time = -1;
var discrete_q_mode = false;
var selected_idxs = new Set();
var saved_times = [];
var actions = [];

var countdown = 30 * 60 * 1000;
var timerId = setInterval(function(){
  countdown -= 1000;
  var min = Math.floor(countdown / (60 * 1000));
  //var sec = Math.floor(countdown - (min * 60 * 1000));  // wrong
  var sec = Math.floor((countdown - (min * 60 * 1000)) / 1000);  //correct

  if (countdown <= 0) {
     alert("30 min!");
     clearInterval(timerId);
     end();
  } else {
     $("#countTime").html(min + " : " + sec);
  }

}, 1000); //1000ms. = 1sec.
""")
            f.write("""
const pairwise = {};""".format(str(pairwise).lower()))
            f.write("""
const all_texts = {};""".format(all_texts[:1000]))
            f.write("""
const interactive_texts = {};""".format(interactive_texts[:1000]))
            f.write("""

function setup_experiments() {
  body = "    <h3 id='countTime'>" + timerId + "</h3>";
  body += "    <h2>#<span id='idx'></span></h2>";
  body += "    <blockquote id='text'></blockquote><hr>";
  body += "    <h3>Are the boxed examples coreferent?: </h3>";
  body += "    <p id='pairwise_prompt'><input type='button' id='Yes' name='pair' value='Yes' onclick='is_coreferent()'> / ";
  body += "<input type='button' id='No' name='pair' value='No' onclick='not_coreferent()'></p>";
  body += "    <p id='discrete_prompt'></p>";
  body += "    <p id='interactive_text'></p>";
  body += "    <hr>";
  document.getElementsByTagName("body")[0].innerHTML = body;
  display_curr_example();
}

function yes_tutorial() {
  document.getElementById("pairwise_prompt_yes").innerHTML = "YES";
}

function no_tutorial() {
  document.getElementById("pairwise_prompt_no").innerHTML = "NO";
}

function is_coreferent() {
  var time = Date.now();
  logged_info = [curr_example, "pair_yes", last_display_time, time]
  saved_times.push(logged_info);
  actions.push(logged_info[1])
  sendInfo(logged_info, "");
  next_example();
}

function not_coreferent() {
  var time = Date.now();
  logged_info = [curr_example, "pair_no", last_display_time, time]
  saved_times.push(logged_info);
  actions.push(logged_info[1])
  sendInfo(logged_info, "");
  if (pairwise) {
    next_example();
  } else {
    display_followup_q();
  }
}

function display_followup_q() {
  discrete_q_mode = true
  display_curr_example();
}

function record_discrete_response() {
  var time = Date.now();
  logged_info = [curr_example, "discrete_submit", last_display_time, time, Array.from(selected_idxs)]
  saved_times.push(logged_info);
  actions.push(logged_info[1])
  sendInfo(logged_info, "");
  discrete_q_mode = false;
  next_example();
}

function display_curr_example() {
  document.getElementById("idx").innerHTML = curr_example;
  document.getElementById("text").innerHTML = all_texts[curr_example];
  if (discrete_q_mode) {
    document.getElementById("interactive_text").innerHTML = interactive_texts[curr_example];
    document.getElementById("discrete_prompt").innerHTML = "<hr> <h3>Select a continuous sequence of words representing the *first* instance of the yellow-highlighted entity in the text. If there is no such entity (yellow-highlighted entity *is* the first instance, or does not refer to anything else in the text), select nothing. (Click 'Submit' when done): <input type='button' id='Submit' name='pair' value='Submit' onclick='record_discrete_response()'> </h3>"
    document.getElementById("pairwise_prompt").innerHTML = "NO";
  } else {
    document.getElementById("interactive_text").innerHTML = "";
    document.getElementById("discrete_prompt").innerHTML = "";
    document.getElementById("pairwise_prompt").innerHTML = "<input type='button' id='Yes' name='pair' value='Yes' onclick='is_coreferent()'> / <input type='button' id='No' name='pair' value='No' onclick='not_coreferent()'>";
  }
  last_display_time = Date.now();
}

function end() {
  log_output = "<h2>You're Done! But don't close this page just yet! Please copy the following and send it to me (belindazli65 [at] gmail.com):</h2>"
  var i;
  var jsonstr = JSON.stringify([uid, saved_times]);
  log_output += "<p>" + jsonstr + "</p>";
  document.getElementsByTagName("body")[0].innerHTML = log_output;
  sendInfo(saved_times, "final");
}

async function sendInfo(data, final_str) {
  var jsonstr = JSON.stringify([uid, data, final_str]);
  // ajax the JSON to the server
  console.log(jsonstr);
  $.post("http://localhost:8080/receiver", jsonstr, function() {

  });
  // stop link reloading the page
  event.preventDefault();
}

function undo() {
  var time = Date.now();
  saved_times.push(["Undo", time]);
  console.log("Undo: " + time);
  if (actions.length > 0) {
    prev_action = actions.pop()
    if (prev_action === "discrete_submit") {
      discrete_q_mode = true;
      curr_example--;
    } else if (prev_action === "pair_yes") {
      curr_example--;
    } else {  // pair_no
      discrete_q_mode = false;
    }
  }
  display_curr_example();
}

function next_example() { 
  selected_idxs.clear();
  curr_example++;
  if (curr_example >= all_texts.length) {
    end();
  } else {
    display_curr_example();
  }
}

function do_click(x) {
    var el, i;
    deselect = selected_idxs.has(x);
    if (deselect) {
      selected_idxs.delete(x);
    } else {
      selected_idxs.add(x);
    }
    el = document.querySelectorAll('.tok' + x);
    for (i = 0; i < el.length; i++) {
      if (deselect) {
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
}""")

