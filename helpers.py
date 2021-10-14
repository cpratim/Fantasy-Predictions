from datetime import datetime
import json

def dump_json(d, f, indent=4):
	with open(f, 'w') as h:
		json.dump(d, h, indent = indent)

def load_json(f):
	with open(f, 'r') as h:
		return json.load(h)

def timestamp():
	return str(datetime.now())[:19]