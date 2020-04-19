from flask import Flask, render_template, request, redirect, Response, jsonify

from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


@app.route("/output")
def output():
    return "Hello World!"

@app.route('/receiver', methods = ['POST', 'GET'])
def worker():
    if request.method == "POST":
        result = request.get_data(as_text=True)
        print(result)
        with open("log.txt", "a") as f:
            c = f.write(result + "\n")
        try:
            result_json = json.loads(result)
            if result_json[-1] == "final":
                with open(result_json[0] + ".json", "w") as f:
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
    app.run(host='0.0.0.0', port=8080)
    