import json
import os
import sys

fileName = "index_lookup.json"
lines = 0
if len(sys.argv) > 2:
	fileName = sys.argv[2]
if(len(sys.argv) > 1):
	lines = int(sys.argv[1])

try:
    f = open(fileName)
    data = json.load(f)
    if(lines > 0):
    	print(data[0:lines])
    else:
    	print(data)
    print(fileName+" read successfully")
except Exception as e:
    print("error in reading ::"+ file)
    print (e)