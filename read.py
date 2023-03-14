import json
import os

try:
    fileName = "index_lookup.json"
    f = open(fileName)
    data = json.load(f)
    print(data)
    print(fileName+" read successfully")
except Exception as e:
    print("error in reading ::"+ file)
    print (e)