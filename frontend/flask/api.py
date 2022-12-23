from flask import Flask, request
import pandas as pd
import pickle
import json

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def view():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)