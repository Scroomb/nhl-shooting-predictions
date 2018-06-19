import pickle
from flask import Flask, url_for, request, render_template, make_response
import numpy as np
import pandas as pd
from single_player import SinglePlayer
import pymongo
app = Flask(__name__)

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def load_data(year):
    goals = pd.read_csv('data/'+str(year)+'_goals.csv')
    shots = pd.read_csv('data/'+str(year)+'_shots.csv')
    missed = pd.read_csv('data/'+str(year)+'_missed.csv')
    return goals,shots,missed

@app.route('/', methods=['GET','POST'])
def api_root():
    teams = ['New Jersey Devils',
             'New York Islanders',
             'New York Rangers',
             'Philadelphia Flyers',
             'Pittsburgh Penguins',
             'Boston Bruins',
             'Buffalo Sabres',
             'Montréal Canadiens',
             'Ottawa Senators',
             'Toronto Maple Leafs',
             'Carolina Hurricanes',
             'Florida Panthers',
             'Tampa Bay Lightning',
             'Washington Capitals',
             'Chicago Blackhawks',
             'Detroit Red Wings',
             'Nashville Predators',
             'St. Louis Blues',
             'Calgary Flames',
             'Colorado Avalanche',
             'Edmonton Oilers',
             'Vancouver Canucks',
             'Anaheim Ducks',
             'Dallas Stars',
             'Los Angeles Kings',
             'San Jose Sharks',
             'Columbus Blue Jackets',
             'Minnesota Wild',
             'Winnipeg Jets',
             'Arizona Coyotes',
             'Vegas Golden Knights']
    return render_template('index.html',teams=teams)

@app.route('/get_goalies/<team>',methods=['GET'])
def get_goalie(team):
    goalies = dict()
    for goalie in db.players.find({'currentTeam.id':team,'primaryPosition.code':'G'},{'fullName':1}):
        goalies.setdefault('fullName',[]).append(goalie['fullName'])
    response = make_response(json.dumps(goalies))
    response.content_type = 'application/json'
    return response


@app.route('/gameplan', methods = ['GET','POST'])
def api_gameplan():
    player_name = [request.form['player_name']]
    opposing_team = request.form['opposing_team']
    opposing_team_id = int(db.team.find_one({'name':opposing_team},{'team':1})['team'])
    opposing_goalie = [request.form['opposing_goalie']]
    sp = SinglePlayer(player_name,2017)
    sp.populate_opponent(opposing_team_id,opposing_goalie,opposing_team)
    sp.get_player_data(shots,goals,missed)
    sp.generate_predictions()
    sp.plot_player_densities()
    return render_template('output.html',player_id=sp.player_id[0], \
            opp_goalie=sp.opponent_goalie_id)

# @app.route('/score', methods = ['GET','POST'])
# def api_score():
#     number = int(request.form['new_query'])
#     # for _ in range(number):
#     num_fraud = 0
#     names = []
#     ids = []
#     predictions = []
#     fraud_not = []
#     while num_fraud < number:
#         response, prediction = predict.make_one_prediction(model,coll)
#         prediction = np.round_(prediction,1)
#         if predict.add_to_db(response,prediction,coll):
#             continue
#         # response = coll.find().sort('inserted',-1).limit(1)
#         if prediction[0][1] == 0:
#             continue
#         names.append(response['name'])
#         ids.append(response['object_id'])
#         predictions.append(prediction[0])
#         if predictions[-1][0] < 0.91:
#             fraud_not.append('Fraud')
#         else:
#             fraud_not.append('Not Fraud')
#         # if prediction[0][1] <= 0.1:
#         #     fraud_not.append('Low Risk')
#         # # elif prediction[0][1] < 0.5:
#         #     fraud_not.append('Medium Risk')
#         # elif prediction[0][1] >= 0.5:
#         #     fraud_not.append('High Risk')
#         num_fraud+=1
#     return render_template('tables.html',data=zip(ids,names,predictions,fraud_not))
#
# @app.route('/submit',methods = ['GET','POST'])
# def api_submit():
#     return render_template('index.html')
#
# @app.route('/db_query',methods = ['GET','POST'])
# def api_db_query():
#     object_id = int(request.form['old_data_query'])
#     response = coll.find_one({'object_id':object_id},{'object_id':1,'name':1,'prediction':1})
#     list_val = [response['object_id'],response['name'],response['prediction'][0]]
#     if list_val[-1][0] < 0.91:
#         list_val.append('Fraud')
#     else:
#         list_val.append('Not Fraud')
#     return render_template('tables.html',data=[list_val])
#
# def api_db_query():
#     object_id = int(request.form['new_query'])
#     response = coll.find_one({'object_id':object_id},{'object_id':1,'name':1,'prediction':1})
#     list_val = [response['object_id'],response['name'],response['prediction'][0]]
#     if list_val[-1][0] < 0.91:
#         list_val.append('Fraud')
#     else:
#         list_val.append('Not Fraud')
#     return render_template('tables.html',data=[list_val])

if __name__ == '__main__':
    goals,shots,missed = load_data(2017)
    # predict.make_one_prediction(model,coll)
    db = _init_mongo()

    app.run(host='0.0.0.0', port=8080, debug=True)
