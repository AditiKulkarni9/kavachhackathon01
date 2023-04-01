import json
import requests

url = 'http://127.0.0.1:8000/profanity_prediction'

input_datamodel = {
        'hate_speech': 14,
        'offensive_language': 45,
        'neither':7,
        'unnamed':2,
        'count':68
}

input_json = json.dumps(input_datamodel)
 
response=requests.post(url, data=input_json) #posting data to url

print(response.text)
