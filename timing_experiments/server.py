import argparse
from flask import Flask, render_template, request, redirect, Response, jsonify

from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)


SAVE_DIR = "timing_results"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/output")
def output():
    return "Hello World!"

@app.route('/receiver', methods = ['POST', 'GET'])
def worker():
    if request.method == "POST":
        result = request.get_data(as_text=True)
        print(result)
        with open(os.path.join(SAVE_DIR, "log.txt"), "a") as f:
            c = f.write(result + "\n")
        try:
            result_json = json.loads(result)
            if result_json[-1] == "final":
                with open(os.path.join(SAVE_DIR, result_json[0] + ".json"), "w") as f:
                    json.dump(result_json[1], f)
        except:
            return 'OK', 200
        return 'OK', 200

    # GET request
    else:
        message = {'greeting':'Server is working!'}
        return jsonify(message)  # serialize and use JSON headers
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int,
                        help='which port to run on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)
    
