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

#import socketserver
#import http.server
#import logging
#import cgi
#import tempfile
#
#PORT = 8000
#
#class ServerHandler(http.server.SimpleHTTPRequestHandler):
#
#    def make_file(self):
#        return tempfile.TemporaryFile("wb+")
#    
#    # Comment out this line to reproduce the error
#    cgi.FieldStorage.make_file = make_file
#
#    def do_GET(self):
#        logging.error(self.headers)
#        http.server.SimpleHTTPRequestHandler.do_GET(self)
#
#    def do_POST(self):
#        logging.error(self.headers)
#        form = cgi.FieldStorage(
#            fp=self.rfile,
#            headers=self.headers,
#            environ={'REQUEST_METHOD':'POST',
#                     'CONTENT_TYPE':self.headers['Content-Type'],
#                     })
#        if form.list is not None:
#            for item in form.list:
#                logging.error(item)
#        http.server.SimpleHTTPRequestHandler.do_GET(self)
#
#        with open("data.txt", "w") as file:
#            for key in form.keys(): 
#                file.write(str(form.getvalue(str(key))) + ",")
#
#Handler = ServerHandler
#
#httpd = socketserver.TCPServer(("", PORT), Handler)
#
#print("serving at port", PORT)
#httpd.serve_forever()
