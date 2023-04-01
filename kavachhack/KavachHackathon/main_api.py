#bring in lightweight dependencies

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import pickle

app = FastAPI()



class model_input(BaseModel):
    hate_speech:int  
    offensive_language:int
    neither:int
    unnamed:int
    count:int

profanity_model = pickle.load(open('/home/aditi/Desktop/kavachhack/KavachHackathon/model.pkl','rb')) #loading the saved model



@app.post("/profanity_prediction") #data posted to the input
async def end_point(input_param : model_input): #values given by user
    input_data = input_param.json() #data posted to api as json
    input_dict = json.loads(input_data) #json to dict 

#extracting individual values
    hate_dict = input_dict['hate_speech']
    offensive_dict = input_dict['offensive_language']
    neither_dict = input_dict['neither']
    unnanmed_dict = input_dict['unnamed']
    count_dict = input_dict['count']

    input_list = [hate_dict,offensive_dict,neither_dict,unnanmed_dict,count_dict]

    prediction = profanity_model.predict([input_list])

    if prediction[0] == 0:
        return 'no profanity detected'
    else:
        return 'profanity detected'

