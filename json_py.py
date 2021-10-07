import json

with open('chinese_dictionary.json') as json_file:
    data = json.load(json_file)

    print("type: ", type(data))
    print(data['\u6211'])