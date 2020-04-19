var uid = create_UUID();
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
     clearInterval(timerId);
     end();
  } else {
     $("#countTime").html(min + " : " + sec);
  }

}, 1000); //1000ms. = 1sec.


function setup_experiments() {
   body = "    <h3 id='countTime'>" + timerId + "</h3>";
   body += "    <h2>#<span id='idx'></span></h2>";
   body += "    <blockquote id='text'></blockquote><hr>";
   body += "    <h3>Are the boxed examples coreferent?: </h3>";
   body += "    <p id='pairwise_prompt'><input type='button' id='Yes' name='pair' value='Yes' onclick='is_coreferent()'> / ";
   body += "<input type='button' id='No' name='pair' value='No' onclick='not_coreferent()'></p>";
   body += "    <p id='discrete_prompt'></p>";
   body += "    <blockquote id='interactive_text'></blockquote>";
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
   $.post("http://128.208.5.20:8080/receiver", jsonstr, function() {
 
   });
   // stop link reloading the page
   //event.preventDefault();
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
 }

 function create_UUID() {
   var dt = new Date().getTime();
   var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
       var r = (dt + Math.random()*16)%16 | 0;
       dt = Math.floor(dt/16);
       return (c=='x' ? r :(r&0x3|0x8)).toString(16);
   });
   return uuid;
}
