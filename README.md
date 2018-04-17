# Predicting Optimal Shooting Locations

### Background  
Starting in 2010 the NHL has tracked in game events on an XY coordinate system.  Using this data for a given season, can optimal shooting locations be predicted for a team?

### Data
The data is accessed through a public API.  Endpoints exist for individual games in a season, player statistics for a given season, player information, and many others.  Each query produced a JSON file that I have stored in a MongoDB.  Since the XY coordinates only measure a single one foot by one foot square on the rink, I have generated KDE esitmations for the distribution of shots and goals for both the shooters and goalies.

##### Nathan MacKinnon Goals
![Nathan MacKinnon](/figs/nm_goals.png)

##### Nathan MacKinnon Shots
![Nathan MacKinnon](/figs/nm_shots.png)

##### Pekka Rinne Goals
![Pekka Rinne](/figs/pr_goals.png)

##### Pekka Rinne Shots
![Pekka Rinne](/figs/pr_shots.png)

### Model  
Right now I am using a 2 hidden layer MLP with 100 nodes each.

![MacKinnon vs Rinne](/figs/nm_vs_pr.png)
