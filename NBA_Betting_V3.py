
# Quarterly Betting
# The goal of this model will be to bet on games in the 3rd quarter.
# Timing is going to be the biggest issue for this modelling.

#Import Libraries
#train function

import requests
import random
import json
import math
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, date
from pandas import DataFrame
import time
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier  # For classification
from sklearn.model_selection import GridSearchCV
from threading import Timer

import basketball_reference_web_scraper
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType

directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'
from basketball_reference_web_scraper.data import OutputType

standing=pd.DataFrame()

Schedule2023=pd.DataFrame(client.season_schedule(season_end_year=2023))
Schedule2023['Season Year'] = '2023'

Schedule2022=pd.DataFrame(client.season_schedule(season_end_year=2022))
Schedule2022['Season Year'] = '2022'

Schedule2021=pd.DataFrame(client.season_schedule(season_end_year=2021))
Schedule2021['Season Year'] = '2021'

#2020 Schedule
Schedule2020=pd.DataFrame(client.season_schedule(season_end_year=2020))
Schedule2020['Season Year'] = '2020'

#2019 Schedule
Schedule2019=pd.DataFrame(client.season_schedule(season_end_year=2019))
Schedule2019['Season Year'] = '2019'

#Repeat
Schedule2018=pd.DataFrame(client.season_schedule(season_end_year=2018))
Schedule2018['Season Year'] = '2018'
Schedule2017=pd.DataFrame(client.season_schedule(season_end_year=2017))
Schedule2017['Season Year'] = '2017'
Schedule2016=pd.DataFrame(client.season_schedule(season_end_year=2016))
Schedule2016['Season Year'] = '2016'
Schedule2015=pd.DataFrame(client.season_schedule(season_end_year=2015))
Schedule2015['Season Year'] = '2015'
schedule = pd.DataFrame()
Schedule2014=pd.DataFrame(client.season_schedule(season_end_year=2014))
Schedule2014['Season Year'] = '2014'
schedulelist = [Schedule2023,Schedule2022,Schedule2021,Schedule2020,Schedule2019,Schedule2018,Schedule2017,Schedule2016,Schedule2015,Schedule2014]

schedule=pd.DataFrame()

#Append each schedule and do appropriate conversions
schedule=schedule.append(Schedule2023).append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)
schedule['start_time']=schedule['start_time']-timedelta(hours=12)
schedule['start_time']=schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team']=schedule['away_team'].astype('S')
awayteams=schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team']=schedule['home_team'].astype('S')
hometeams=schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
#gamehist = schedule.loc[schedule['Season Year']!=currentyear]['start_time']
teammapping = pd.read_csv(directory1+'dataframe.csv')
schedule = pd.merge(schedule,teammapping,left_on = schedule['home_team'],right_on=teammapping['Team_Names'],how='left').drop('Team_Names',axis=1).rename(columns={'Mapping':'HOME_TEAM_ABBREVIATION'})
schedule=schedule.drop('key_0',axis=1)
schedule = pd.merge(schedule,teammapping,left_on = schedule['away_team'],right_on=teammapping['Team_Names'],how='left').drop('Team_Names',axis=1).rename(columns={'Mapping':'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())

seasonyears = ['2023','2022','2021','2020','2019','2018','2017','2016','2015','2014']

currentyear='2023'

#We want to add the new data to our original file, therefore we filter on the new season and compare the dates in our file to the games that have been played.

quarterdatalist=[]
quarterdataplayoffslist=[]
notcaptured=[]
dataytd=[]
dates=pd.DataFrame()
datestemp=pd.DataFrame()

for s in seasonyears:
    scheduletemp=schedule[(schedule['Season Year'] == s)]
    dataytdtemp=scheduletemp.filter(['start_time','home_team_score']).dropna()['start_time'].drop_duplicates()
    dataytd.insert(seasonyears.index(s),dataytdtemp)
    dates=dates.append(dataytd[seasonyears.index(s)].tolist())
    #dates = dataytd[0].tolist()+dataytd[1].tolist()+dataytd[2].tolist()+dataytd[3].tolist()+dataytd[4].tolist()+dataytd[5].tolist()+dataytd[6].tolist()

datelist=dates
datelist=datelist.reset_index()
quarterdata=pd.read_csv(directory1+'nbadataytd_quarterly.csv',low_memory=False)
quarterdata=quarterdata[['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN',
       'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'GP_RANK', 'W_RANK', 'L_RANK',
       'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
       'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
       'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
       'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
       'PTS_RANK', 'PLUS_MINUS_RANK', 'CFID', 'CFPARAMS', 'Gamedate',
       'Quarter']]

captureddates=pd.DataFrame(quarterdata['Gamedate'].unique())


datelist=[x for x in dates[0].tolist() if x not in captureddates[0].tolist()]
datelist = [x for x in datelist if x > max(captureddates[0])]
#leftover games are generally the playoff games.

#The following for loop reads the new data from nba.com into a list

quarterworking=pd.DataFrame()
quarterworkingplayoffs=pd.DataFrame()

for period in range(0,4):

        for e in range(0,len(dates)):
            try:
                year1 = datelist[e][:4]
                month1 = datelist[e][5:7]
                day1 = datelist[e][8:10]
                a='https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
                if datelist[e] in dataytd[0].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[1].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[2].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[3].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[4].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[5].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[6].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[7].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[8].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[9].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                c='&DateTo='
                d='%2F'
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4692.45 Safari/537.36','x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/','Host': 'stats.nba.com','Origin': 'https://www.nba.com'}
                url = a+month1+d+day1+d+year1+c+month1+d+day1+d+year1+b
                r=requests.get(url,headers=headers)
                numrecords = len(r.json()['resultSets'][0]['rowSet'])
                fields=r.json()['resultSets'][0]['headers']
                data = pd.DataFrame(index = np.arange(numrecords) , columns = fields)
                for i in range(0,numrecords):
                    records = r.json()['resultSets'][0]['rowSet'][i]
                    for j in range(0,len(records)):
                        data.iloc[[i],[j]]=records[j]
                quarterdatalist.insert(e,[data,datelist[e],period+1])
            except:
                pass

quartertemp=pd.DataFrame()

#Read the data from the website into a dataframe
for i in range(0,len(quarterdatalist)):
    quartertemp=pd.DataFrame(quarterdatalist[i][0])
    quartertemp['Gamedate'] = quarterdatalist[i][1]
    quartertemp['Quarter'] = quarterdatalist[i][2]
    quarterworking=quarterworking.append(quartertemp)

quarterplayoffs=pd.DataFrame()

for period in range(0,4):
        for e in range(0,len(dates)):
            try:
                year1 = datelist[e][:4]
                month1 = datelist[e][5:7]
                day1 = datelist[e][8:10]
                a='https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
                if datelist[e] in dataytd[0].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[1].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[2].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[3].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[4].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[5].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[6].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[7].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[8].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[9].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                c='&DateTo='
                d='%2F'
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4692.45 Safari/537.36','x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/','Host': 'stats.nba.com','Origin': 'https://www.nba.com'}
                url = a+month1+d+day1+d+year1+c+month1+d+day1+d+year1+b
                r=requests.get(url,headers=headers)
                numrecords = len(r.json()['resultSets'][0]['rowSet'])
                fields=r.json()['resultSets'][0]['headers']
                data = pd.DataFrame(index = np.arange(numrecords) , columns = fields)
                for i in range(0,numrecords):
                    records = r.json()['resultSets'][0]['rowSet'][i]
                    for j in range(0,len(records)):
                        data.iloc[[i],[j]]=records[j]
                quarterdataplayoffslist.insert(e,[data,datelist[e],period+1])
            except:
                pass

quartertempplayoffs=pd.DataFrame()

#Read the data from the website into a dataframe
for i in range(0,len(quarterdataplayoffslist)):
    quartertempplayoffs=pd.DataFrame(quarterdataplayoffslist[i][0])
    quartertempplayoffs['Gamedate'] = quarterdataplayoffslist[i][1]
    quartertempplayoffs['Quarter'] = quarterdataplayoffslist[i][2]
    quarterworkingplayoffs=quarterworkingplayoffs.append(quartertempplayoffs)

quarterdata=quarterdata.append(quarterworking)
quarterdata=quarterdata.append(quarterworkingplayoffs)

directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'

quarterdata=quarterdata[['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT',
       'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'GP_RANK', 'W_RANK', 'L_RANK',
       'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
       'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
       'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
       'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
       'PTS_RANK', 'PLUS_MINUS_RANK', 'CFID', 'CFPARAMS', 'Gamedate',
       'Quarter', 'Playoffs']]

quarterdata=quarterdata.drop_duplicates()

quarterdata.to_csv(directory1+'nbadataytd_quarterly_v2.csv')

quarterdata = pd.read_csv(directory1+'nbadataytd_quarterly_v2.csv')

quarterdata=quarterdata[['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT',
       'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'GP_RANK', 'W_RANK', 'L_RANK',
       'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
       'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
       'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
       'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
       'PTS_RANK', 'PLUS_MINUS_RANK', 'CFID', 'CFPARAMS', 'Gamedate',
       'Quarter']]

quarterdata['TEAM_NAME'] = quarterdata['TEAM_NAME'].str.replace(" ", "_").str.upper()

quarterdata=quarterdata.drop_duplicates()

teammapping = pd.read_csv(directory1+'dataframe.csv')

quarterdata = pd.merge(quarterdata, teammapping, left_on=quarterdata['TEAM_NAME'], right_on=teammapping['Team_Names'],
                       how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'TEAM_ABBREVIATION'})

# schedule['home_teamID'] = schedule['HOME_TEAM_ABBREVIATION'] + schedule['start_time']
# schedule['away_teamID'] = schedule['AWAY_TEAM_ABBREVIATION'] + schedule['start_time']

standingslist = []
standing = pd.DataFrame()
#2023 Schedule
Schedule2023=pd.DataFrame(client.season_schedule(season_end_year=2023))
Schedule2023['Season Year'] = '2023'
#2022 Schedule
Schedule2022=pd.DataFrame(client.season_schedule(season_end_year=2022))
Schedule2022['Season Year'] = '2022'
#2021 Schedule
Schedule2021=pd.DataFrame(client.season_schedule(season_end_year=2021))
Schedule2021['Season Year'] = '2021'
#2020 Schedule
Schedule2020=pd.DataFrame(client.season_schedule(season_end_year=2020))
Schedule2020['Season Year'] = '2020'
#2019 Schedule
Schedule2019=pd.DataFrame(client.season_schedule(season_end_year=2019))
Schedule2019['Season Year'] = '2019'
#Repeat
Schedule2018=pd.DataFrame(client.season_schedule(season_end_year=2018))
Schedule2018['Season Year'] = '2018'
Schedule2017=pd.DataFrame(client.season_schedule(season_end_year=2017))
Schedule2017['Season Year'] = '2017'
Schedule2016=pd.DataFrame(client.season_schedule(season_end_year=2016))
Schedule2016['Season Year'] = '2016'
Schedule2015=pd.DataFrame(client.season_schedule(season_end_year=2015))
Schedule2015['Season Year'] = '2015'
schedule = pd.DataFrame()
Schedule2014=pd.DataFrame(client.season_schedule(season_end_year=2014))
Schedule2014['Season Year'] = '2014'
schedulelist = [Schedule2023,Schedule2022,Schedule2021,Schedule2020,Schedule2019,Schedule2018,Schedule2017,Schedule2016,Schedule2015,Schedule2014]

schedule = pd.DataFrame()

schedulelist = [Schedule2023,Schedule2022,Schedule2021, Schedule2020, Schedule2019, Schedule2018, Schedule2017, Schedule2016, Schedule2015,Schedule2014]

schedule = schedule.append(Schedule2023).append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)

schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]

schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())

seasonyears = ['2023','2022','2021', '2020', '2019', '2018', '2017', '2016', '2015']

seasonmerge = schedule.filter(['Season Year', 'start_time'])
seasonmerge = seasonmerge.drop_duplicates()
quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, seasonmerge, left_on=quarterdata['Gamedate'], right_on=seasonmerge['start_time'],how='left').drop('start_time', axis=1)

# The following for loop develops the standings for each team and assigns H or A for home or away games.
from pandas.core.common import SettingWithCopyWarning  # I need this to not get a billion warnings while completing the loop.
nbaschedule=pd.DataFrame()
for s in seasonyears:
        scheduletemp = schedule[(schedule['Season Year'] == s) & (schedule['start_time'] < (max(quarterdata[(quarterdata['Season Year'] == s)]['Gamedate'])))]
        scheduletemp.sort_values('start_time', inplace=True, ascending=False)
        scheduletemp['HomeWinner'] = np.where(scheduletemp['home_team_score'] > scheduletemp['away_team_score'], 1, 0)
        scheduletemp['AwayWinner'] = np.where(scheduletemp['home_team_score'] < scheduletemp['away_team_score'], 1, 0)
        scoresdata = scheduletemp.loc[scheduletemp['HomeWinner'] == 1]
        scoresdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['HomeWinner'] == 0]
        scoresdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['AwayWinner'] == 1]
        scoresdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['AwayWinner'] == 0]
        scoresdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        HomeWinnersdata = pd.DataFrame()
        HomeWinnersdata['HOME_TEAM_ABBREVIATION'] = scheduletemp['HOME_TEAM_ABBREVIATION']
        HomeWinnersdata['home_team_score'] = scheduletemp['home_team_score']
        HomeWinnersdata['HomeWinner'] = scheduletemp['HomeWinner']
        a = HomeWinnersdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        AwayWinnersdata = pd.DataFrame()
        AwayWinnersdata['AWAY_TEAM_ABBREVIATION'] = scheduletemp['AWAY_TEAM_ABBREVIATION']
        AwayWinnersdata['away_team_score'] = scheduletemp['away_team_score']
        AwayWinnersdata['AwayWinner'] = scheduletemp['AwayWinner']
        b = AwayWinnersdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        standingtemp = pd.concat([a, b], axis=1)
        standingtemp['Total_Wins'] = standingtemp['HomeWinner'] + standingtemp['AwayWinner']
        standingtemp['Team Rank'] = standingtemp['Total_Wins'].rank(ascending=False, method='first')
        standingtemp.sort_values('Total_Wins', inplace=True, ascending=False)
        standingtemp['Win@Home%'] = standingtemp['HomeWinner'] / 41
        standingtemp['Win@Away%'] = standingtemp['AwayWinner'] / 41
        standingtemp['Season Year'] = s
        standing = standing.append(standingtemp)
        standingslist.insert(seasonyears.index(s), standingtemp)
        # Secondary component of for loop is to set Home and away teams
        for team in nbateams[0]:
            homeschedule = scheduletemp.loc[scheduletemp['HOME_TEAM_ABBREVIATION'] == team]
            homeschedule = [homeschedule['start_time'], homeschedule['HOME_TEAM_ABBREVIATION']]
            homeschedule = pd.DataFrame(homeschedule)
            homeschedule = np.transpose(homeschedule)
            homeschedule = homeschedule.rename(columns={homeschedule.columns[0]: 'home_schedule'})
            awayschedule = scheduletemp.loc[scheduletemp['AWAY_TEAM_ABBREVIATION'] == team]
            awayschedule = [awayschedule['start_time'], awayschedule['AWAY_TEAM_ABBREVIATION']]
            awayschedule = pd.DataFrame(awayschedule)
            awayschedule = np.transpose(awayschedule)
            awayschedule = awayschedule.rename(columns={awayschedule.columns[0]: 'away_schedule'})
            teamschedule = pd.concat([homeschedule, awayschedule], axis=1)
            teamschedule = pd.DataFrame(teamschedule)
            teamschedule.transpose()
            teamschedule['H/A'] = np.where(teamschedule['HOME_TEAM_ABBREVIATION'] == team, 'H', 'A')
            teamschedule['home_schedule'] = np.where(teamschedule['home_schedule'].isnull(), teamschedule['away_schedule'],
                                                     teamschedule['home_schedule'])
            teamschedule = teamschedule.rename(columns={teamschedule.columns[0]: 'schedule'})
            teamschedule = teamschedule.filter(['schedule', 'H/A'])
            teamschedule.reset_index(inplace=True)
            del [teamschedule['index']]
            teamschedule = teamschedule.drop_duplicates(subset=['schedule'], keep='last')
            scheduledates = list(teamschedule['schedule'])
            # For loop for days in between games for players to rest
            daysrest = list()
            for i in scheduledates:
                x = scheduledates.index(i)
                year = int(scheduledates[x][:4])
                month = int(scheduledates[x][5:7])
                day = int(scheduledates[x][8:10])
                if (x + 1 == len(scheduledates)):
                    daysrest.insert(0, 5)
                else:
                    year1 = int(scheduledates[x + 1][:4])
                    month1 = int(scheduledates[x + 1][5:7])
                    day1 = int(scheduledates[x + 1][8:10])
                    d = (datetime.datetime(year1, month1, day1) - datetime.datetime(year, month, day))
                    d = int(d.total_seconds() / 60 / 24 / 60)
                    daysrest.insert(x, d)
            teamschedule['DaysRest'] = daysrest
            teamschedule['Team'] = team
            teamschedule['ScheduleID'] = teamschedule['schedule'] + teamschedule['H/A'] + teamschedule['Team']
            nbaschedule = nbaschedule.append(teamschedule)

# finalize schedule dataframe
schedule['HomeScheduleID'] = schedule['start_time'] + 'H' + schedule['HOME_TEAM_ABBREVIATION']
schedule['AwayScheduleID'] = schedule['start_time'] + 'A' + schedule['AWAY_TEAM_ABBREVIATION']
schedule['TotalScore'] = schedule['away_team_score'] + schedule['home_team_score']
schedule = schedule.drop('key_0', axis=1)
nbaschedule = nbaschedule.reset_index()
schedule = pd.merge(schedule, nbaschedule, left_on=schedule['HomeScheduleID'], right_on=nbaschedule['ScheduleID'],how='left')
schedule = schedule.drop('Team', 1).drop('schedule', 1).drop('H/A', 1).drop('ScheduleID', 1)
schedule = schedule.rename(columns={schedule.columns[len(schedule.columns) - 1]: 'HomeDaysRest'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, nbaschedule, left_on=schedule['AwayScheduleID'], right_on=nbaschedule['ScheduleID'],how='left')
schedule = schedule.drop('Team', 1).drop('schedule', 1).drop('H/A', 1).drop('ScheduleID', 1)
schedule = schedule.rename(columns={schedule.columns[len(schedule.columns) - 1]: 'AwayDaysRest'})
schedule = schedule.dropna()

quarterdata['ScheduleID'] = quarterdata['Gamedate'] + quarterdata['TEAM_ABBREVIATION']
nbaschedule['ScheduleID2'] = nbaschedule['schedule'] + nbaschedule['Team']
schedulemerge = nbaschedule.filter(['ScheduleID2', 'H/A'])
schedulemerge = schedulemerge.drop_duplicates()
quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, schedulemerge, left_on=quarterdata['ScheduleID'], right_on=schedulemerge['ScheduleID2'],
                    how='left').drop('ScheduleID2', axis=1).drop('ScheduleID', axis=1)

nbaschedule = nbaschedule[nbaschedule['schedule'].isin(quarterdata['Gamedate'].unique())]
#
quarterdata['ScheduleID'] = quarterdata['Gamedate'] + quarterdata['H/A'] + quarterdata['TEAM_ABBREVIATION']
a = schedule.filter(['HomeScheduleID', 'TotalScore', 'HomeDaysRest', 'home_team_score'])
b = schedule.filter(['AwayScheduleID', 'TotalScore', 'AwayDaysRest', 'away_team_score'])
b = b.rename(columns={'AwayScheduleID': 'HomeScheduleID', 'AwayDaysRest': 'HomeDaysRest'})
schedulemerge = a.append(b)
schedulemerge = schedulemerge.drop_duplicates()
quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, schedulemerge, left_on=quarterdata['ScheduleID'], right_on=schedulemerge['HomeScheduleID'],
                    how='left')
quarterdata['ScheduleID'] = quarterdata['Gamedate'] + quarterdata['H/A'] + quarterdata['TEAM_ABBREVIATION']
a = schedule.filter(['HomeScheduleID', 'AwayScheduleID'])
b = schedule.filter(['AwayScheduleID', 'HomeScheduleID'])
b = b.rename(columns={'AwayScheduleID': 'HomeScheduleID', 'HomeScheduleID': 'AwayScheduleID'})
b = b.filter(['HomeScheduleID', 'AwayScheduleID'])
schedulemerge = a.append(b)
schedulemerge = schedulemerge.rename(columns={'HomeScheduleID': 'Team', 'AwayScheduleID': 'Opponent'})
schedulemerge = schedulemerge.drop_duplicates()
quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, schedulemerge, left_on=quarterdata['ScheduleID'], right_on=schedulemerge['Team'],
                    how='left').drop('ScheduleID', axis=1).drop('Team', axis=1)
schedulemerge = schedulemerge.rename(columns={'AwayScheduleID': 'Opponent'})

quarterdata['Opponent'] = quarterdata['Opponent'].str[-3:]
quarterdata['OpponentID'] = quarterdata['Opponent'] + quarterdata['Season Year']
quarterdata['SeasonID'] = quarterdata['TEAM_ABBREVIATION'] + quarterdata['Season Year']

opponentrank = quarterdata.filter(['Season Year', 'Opponent'])
opponentrank = opponentrank.drop_duplicates().dropna()
opponentrank['SeasonID'] = opponentrank['Opponent'] + opponentrank['Season Year']
rankingtemp = quarterdata.filter(['SeasonID', 'Team Rank']).drop_duplicates().dropna()
opponentrank = pd.merge(rankingtemp, opponentrank, left_on=rankingtemp['SeasonID'], right_on=opponentrank['SeasonID'],
                        how='left').drop('SeasonID_x', axis=1)
opponentrank = opponentrank.filter(['SeasonID_y', 'Team Rank'])
opponentrank = opponentrank.rename(columns=({'SeasonID_y': 'SeasonID', 'Team Rank': 'Opponent Rank'}))

quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, opponentrank, left_on=quarterdata['OpponentID'], right_on=opponentrank['SeasonID'],
                   how='left').drop('SeasonID_x', axis=1).drop('SeasonID_y', axis=1)

# Assign Conference
conference = pd.DataFrame(data={'Team': ['BOS', 'GSW', 'HOU', 'CLE', 'ORL', 'UTA', 'BKN', 'DET', 'MIN',
                                         'NOP', 'PHI', 'WAS', 'MEM', 'SAC', 'CHA', 'POR', 'SAS', 'IND',
                                         'ATL', 'DAL', 'PHX', 'MIA', 'MIL', 'DEN', 'OKC', 'LAL', 'TOR',
                                         'NYK', 'LAC', 'CHI'],
                                'Conference': ['E', 'W', 'W', 'E', 'E', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W',
                                               'W', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'W',
                                               'E', 'E', 'W', 'E']})

# merge conference into your data
quarterdata = pd.merge(quarterdata, conference, left_on='TEAM_ABBREVIATION', right_on='Team', how='left').drop('Team', axis=1)
# assign opponent conference merge.
opponentconference = quarterdata.filter(['Conference', 'TEAM_ABBREVIATION'])
opponentconference = opponentconference.drop_duplicates()
opponentconference = opponentconference.rename(
    columns=({'Conference': 'Opponent Conference', 'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION1'}))
quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, opponentconference, left_on=quarterdata['Opponent'],
                   right_on=opponentconference['TEAM_ABBREVIATION1'], how='left').drop('TEAM_ABBREVIATION1', axis=1)
quarterdata['ConferenceBinary'] = np.where(quarterdata['Conference'] == 'W', 0, 1)
quarterdata['OpponentConferenceBinary'] = np.where(quarterdata['Opponent Conference'] == 'W', 0, 1)

#In between seasons this will give extra large numbers and in-between February break
quarterdata['HomeDaysRest'] = quarterdata['HomeDaysRest'].clip(upper=5)
quarterdata['away_team_score'][np.isnan(quarterdata['away_team_score'])] = 0
quarterdata['home_team_score'][np.isnan(quarterdata['home_team_score'])] = 0
quarterdata = quarterdata.drop('key_0', axis=1)

# Merge to nbadata, rankings, home and away info, etc.
#quarterdata['TEAM_ABBREVIATION'] = quarterdata['ID'].str[:3]

for s in seasonyears:
        scheduletemp = schedule[(schedule['Season Year'] == s)&(schedule['start_time']<(max(quarterdata[(quarterdata['Season Year']==s)]['Gamedate'])))] # &quarterdata['Playoffs']==0
        scheduletemp['HomeWinner'] = np.where(scheduletemp['home_team_score'] > scheduletemp['away_team_score'], 1, 0)
        scheduletemp['AwayWinner'] = np.where(scheduletemp['home_team_score'] < scheduletemp['away_team_score'], 1, 0)
        scoresdata = scheduletemp.loc[scheduletemp['HomeWinner'] == 1]
        scoresdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['HomeWinner'] == 0]
        scoresdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['AwayWinner'] == 1]
        scoresdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        scoresdata = scheduletemp.loc[scheduletemp['AwayWinner'] == 0]
        scoresdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        HomeWinnersdata = pd.DataFrame()
        HomeWinnersdata['HOME_TEAM_ABBREVIATION'] = scheduletemp['HOME_TEAM_ABBREVIATION']
        HomeWinnersdata['home_team_score'] = scheduletemp['home_team_score']
        HomeWinnersdata['HomeWinner'] = scheduletemp['HomeWinner']
        a = HomeWinnersdata.groupby('HOME_TEAM_ABBREVIATION').agg({'HomeWinner': 'sum', 'home_team_score': 'mean'})
        AwayWinnersdata = pd.DataFrame()
        AwayWinnersdata['AWAY_TEAM_ABBREVIATION'] = scheduletemp['AWAY_TEAM_ABBREVIATION']
        AwayWinnersdata['away_team_score'] = scheduletemp['away_team_score']
        AwayWinnersdata['AwayWinner'] = scheduletemp['AwayWinner']
        b = AwayWinnersdata.groupby('AWAY_TEAM_ABBREVIATION').agg({'AwayWinner': 'sum', 'away_team_score': 'mean'})
        standingtemp = pd.concat([a, b], axis=1)
        standingtemp['Total_Wins'] = standingtemp['HomeWinner'] + standingtemp['AwayWinner']
        standingtemp['Team Rank'] = standingtemp['Total_Wins'].rank(ascending=False, method='first')
        standingtemp.sort_values('Total_Wins', inplace=True, ascending=False)
        standingtemp['Win@Home%'] = standingtemp['HomeWinner'] / 41
        standingtemp['Win@Away%'] = standingtemp['AwayWinner'] / 41
        standingtemp['Season Year'] = s
        standing = standing.append(standingtemp)
        standingslist.insert(seasonyears.index(s), standingtemp)

schedule=schedule.drop('key_0', axis=1)

schedule = pd.merge(schedule, conference, left_on=schedule['HOME_TEAM_ABBREVIATION'], right_on=conference['Team'],
                how='left').drop('Team', axis=1).rename(columns={'Conference': 'Conference_Home'})

schedule=schedule.drop('key_0', axis=1)

schedule = pd.merge(schedule, conference, left_on=schedule['AWAY_TEAM_ABBREVIATION'], right_on=conference['Team'],
                        how='left').drop('key_0', axis=1).drop('Team', axis=1).rename(columns={'Conference': 'Conference_Away'})

schedule['rankIDhome']=schedule['HOME_TEAM_ABBREVIATION']+schedule['Season Year']

schedule['rankIDaway']=schedule['AWAY_TEAM_ABBREVIATION']+schedule['Season Year']

standing['Index']=standing.index
standing['StandingID'] = standing['Index'] + standing['Season Year']
standingmerge = standing.filter(['StandingID', 'Team Rank'])

standingmerge=standingmerge.drop_duplicates()

schedule = pd.merge(schedule, standingmerge, left_on=schedule['rankIDhome'], right_on=standingmerge['StandingID'],
                    how='left').rename(columns={'Team Rank': 'Team Rank Home'})

schedule=schedule.drop('key_0', axis=1)

schedule = pd.merge(schedule, standingmerge, left_on=schedule['rankIDaway'], right_on=standingmerge['StandingID'],
                        how='left').rename(columns={'Team Rank': 'Team Rank Away'})

quarterdata['StandingID'] = quarterdata['TEAM_ABBREVIATION'] + quarterdata['Season Year']

#quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata = pd.merge(quarterdata, standingmerge, left_on=quarterdata['StandingID'], right_on=standingmerge['StandingID'],
                   how='left').drop('StandingID_y', axis=1).drop('StandingID_x', axis=1)

quarterdata['ScheduleID'] = quarterdata['Gamedate'] + quarterdata['TEAM_ABBREVIATION']
nbaschedule['ScheduleID2'] = nbaschedule['schedule'] + nbaschedule['Team']

# schedulemerge = nbaschedule.filter(['ScheduleID2', 'H/A'])
# schedulemerge = schedulemerge.drop_duplicates()
# quarterdata = quarterdata.drop('key_0', axis=1)
# quarterdata = pd.merge(quarterdata, schedulemerge, left_on=quarterdata['ScheduleID'], right_on=schedulemerge['ScheduleID2'],
#                    how='left').drop('ScheduleID2', axis=1).drop('ScheduleID', axis=1)

nbaschedule = nbaschedule[nbaschedule['schedule'].isin(quarterdata['Gamedate'].unique())]

# playerqdata=pd.read_csv(directory1+'playerytd_quarterly.csv')
#
# playerdatalist = []
# playoffslist = []
#
# for period in range(0, 4):
#     for e in range(0, len(datelist)):
#         try:
#             year1 = datelist[e][:4]
#             month1 = datelist[e][5:7]
#             day1 = datelist[e][8:10]
#             a = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom='
#             if datelist[e] in dataytd[0].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[1].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[2].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[3].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[4].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[5].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[6].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[6].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             c = '&DateTo='
#             d = '%2F'
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.42 Safari/537.36',
#                 'x-nba-stats-origin': 'stats'}
#             url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
#             r = requests.get(url, headers=headers)
#             numrecords = len(r.json()['resultSets'][0]['rowSet'])
#             fields = r.json()['resultSets'][0]['headers']
#             data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
#             for i in range(0, numrecords):
#                 records = r.json()['resultSets'][0]['rowSet'][i]
#                 for j in range(0, len(records)):
#                     data.iloc[[i], [j]] = records[j]
#             playerdatalist.insert(e, [data, datelist[e],period+1])
#         except:
#             pass
#
# for period in range(0, 4):
#     for e in range(0, len(datelist)):
#         try:
#             year1 = datelist[e][:4]
#             month1 = datelist[e][5:7]
#             day1 = datelist[e][8:10]
#             a = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom='
#             if datelist[e] in dataytd[0].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[1].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[2].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[3].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[4].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[5].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             elif datelist[e] in dataytd[6].tolist():
#                 b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' + str(
#                     period + 1) + '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
#             c = '&DateTo='
#             d = '%2F'
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
#                 'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/'}
#             url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
#             r = requests.get(url, headers=headers)
#             numrecords = len(r.json()['resultSets'][0]['rowSet'])
#             fields = r.json()['resultSets'][0]['headers']
#             data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
#             for i in range(0, numrecords):
#                 records = r.json()['resultSets'][0]['rowSet'][i]
#                 for j in range(0, len(records)):
#                     data.iloc[[i], [j]] = records[j]
#             playoffslist.insert(e, [data, datelist[e],period+1])
#         except:
#             pass
#
# playoffsworking = pd.DataFrame()
# playoffstemp = pd.DataFrame()
#
# # Read the data from the website into a dataframe
# for i in range(0, len(playoffslist)):
#     playoffstemp = pd.DataFrame(playoffslist[i][0])
#     playoffstemp['Gamedate'] = playoffslist[i][1]
#     playoffsworking = playoffsworking.append(playoffstemp)
#
# playoffsworking['Playoffs'] = 1
#
# if not (playoffsworking.empty):
#     playoffsdata = playoffsworking[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
#                                     'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
#                                     'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
#                                     'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
#                                     'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
#                                     'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
#                                     'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
#                                     'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
#                                     'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
#                                     'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
#                                     'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs']]
# if (playoffsworking.empty):
#     playoffsdata = pd.DataFrame()
#
# playoffsdata = playoffsdata.append(playoffsworking)
# playoffsdata = playoffsdata.reset_index()
#
# playerdataworking = pd.DataFrame()
# playerdatatemp = pd.DataFrame()
#
# # Read the data from the website into a dataframe
# for i in range(0, len(playerdatalist)):
#     playerdatatemp = pd.DataFrame(playerdatalist[i][0])
#     playerdatatemp['Gamedate'] = playerdatalist[i][1]
#     playerdatatemp['Quarter'] = playerdatalist[i][2]
#     playerdataworking = playerdataworking.append(playerdatatemp)
#
# playerdataworking['Playoffs'] = 0
#
# if not (playerdataworking.empty):
#     playerdataworking = playerdataworking[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
#                                      'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
#                                      'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
#                                      'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
#                                      'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
#                                      'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
#                                      'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
#                                      'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
#                                      'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
#                                      'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
#                                      'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs','Quarter']]
#
# if (playerdataworking.empty):
#     playerdatatworking = pd.DataFrame()
#
# playerdata = pd.DataFrame()
# playerdataworking = playerdataworking.append(playoffsdata)
# playerdata = playerdata.append(playerdataworking)
# playerdata.to_csv(directory1+'playerytd_quarterly.csv')

playerqdata=pd.read_csv(directory1+'playerytd_quarterly.csv')

playerqdata = playerqdata[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                   'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs','Quarter']]

playerqdata_update=pd.DataFrame()

for r in range(1,5):
    playerqtemp=playerqdata[playerqdata['Quarter']<= r]
    playerqtemp = playerqtemp[playerqtemp['Quarter'] <= r]
    playerqtemp['PlayerID'] = playerqtemp['PLAYER_ID'].astype(str) + playerqtemp['Gamedate']
    playerqtemp = playerqtemp.groupby(['PlayerID','TEAM_ABBREVIATION','Gamedate']).agg({'PTS': 'sum'}).reset_index()
    playerqtemp['>40'] = playerqtemp['PTS'] > 40
    playerqtemp['>30'] = playerqtemp['PTS'] > 30
    playerqtemp['>20'] = playerqtemp['PTS'] > 20
    playerqtemp['>15'] = playerqtemp['PTS'] > 15
    playerqtemp['>10'] = playerqtemp['PTS'] > 10
    playerqtemp['>5'] = playerqtemp['PTS'] > 5

    playerqtemp['GameID'] = playerqtemp['TEAM_ABBREVIATION'] + playerqtemp['Gamedate'] + str(r)
    playerqtemp['PointsID'] = playerqtemp['TEAM_ABBREVIATION'] + playerqtemp['Gamedate'] + str(r)

    morethan40 = playerqtemp[playerqtemp['>40'] == True][['GameID', '>40']].groupby('GameID').agg(
        {'>40': 'sum'}).reset_index()
    morethan30 = playerqtemp[playerqtemp['>30'] == True][['GameID', '>30']].groupby('GameID').agg(
        {'>30': 'sum'}).reset_index()
    morethan20 = playerqtemp[playerqtemp['>20'] == True][['GameID', '>20']].groupby('GameID').agg(
        {'>20': 'sum'}).reset_index()
    morethan15 = playerqtemp[playerqtemp['>15'] == True][['GameID', '>15']].groupby('GameID').agg(
        {'>15': 'sum'}).reset_index()
    morethan10 = playerqtemp[playerqtemp['>10'] == True][['GameID', '>10']].groupby('GameID').agg(
        {'>10': 'sum'}).reset_index()
    morethan5 = playerqtemp[playerqtemp['>5'] == True][['GameID', '>5']].groupby('GameID').agg(
        {'>5': 'sum'}).reset_index()

    playerqtemp = playerqtemp.drop(['>40', '>30', '>20', '>15', '>10', '>5', 'GameID'], axis=1)

    playerqtemp = pd.merge(playerqtemp, morethan40, left_on=playerqtemp['PointsID'], right_on=morethan40['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>40'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = pd.merge(playerqtemp, morethan30, left_on=playerqtemp['PointsID'], right_on=morethan30['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>30'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = pd.merge(playerqtemp, morethan20, left_on=playerqtemp['PointsID'], right_on=morethan20['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>20'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = pd.merge(playerqtemp, morethan15, left_on=playerqtemp['PointsID'], right_on=morethan15['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>15'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = pd.merge(playerqtemp, morethan10, left_on=playerqtemp['PointsID'], right_on=morethan10['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>10'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = pd.merge(playerqtemp, morethan5, left_on=playerqtemp['PointsID'], right_on=morethan5['GameID'],
                           how='left').drop(
        'GameID', axis=1)
    playerqtemp['>5'].fillna(0, inplace=True)
    playerqtemp = playerqtemp.drop('key_0', axis=1)
    playerqtemp = playerqtemp[['PointsID', '>40', '>30', '>20', '>15', '>10', '>5']]
    playerqtemp = playerqtemp.drop_duplicates()
    playerqdata_update=playerqdata_update.append(playerqtemp)

playerqdata=playerqdata_update

quarterdata['ID'] = quarterdata['TEAM_ABBREVIATION'] + quarterdata['Gamedate'] + quarterdata['Quarter'].astype(str)

quarterdata = quarterdata.drop('key_0', axis=1)
quarterdata=pd.merge(quarterdata,playerqdata,left_on=quarterdata['ID'],right_on=playerqdata['PointsID'],how='left')
quarterdata=quarterdata.dropna()

qw = quarterdata

qw1_old=pd.read_csv(directory1+'qw1.csv')
qw2_old=pd.read_csv(directory1+'qw2.csv')
qw3_old=pd.read_csv(directory1+'qw3.csv')
qw4_old=pd.read_csv(directory1+'qw4.csv')
qw1p_old=pd.read_csv(directory1+'qw1p.csv')
qw2p_old=pd.read_csv(directory1+'qw2p.csv')

captureddates=pd.DataFrame(qw['Gamedate'].unique())
qw = qw[qw['Gamedate']>max(captureddates[0].dropna().astype(str))]
qw = qw.dropna()
qw['ID'] = qw['TEAM_ABBREVIATION'] + qw['Gamedate']
IDs = list(qw['ID'].unique())

quartersummary = []
halfsummary = []
threequartersummary = []
fourquartersummary = []
postfirstsummary = []
posthalfsummary = []

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 1)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    quartersummary.insert(IDs.index(ID), [ID, a, b])

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 2)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    halfsummary.insert(IDs.index(ID), [ID, a, b])

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 3)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    threequartersummary.insert(IDs.index(ID), [ID, a, b])

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] == 4)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    fourquartersummary.insert(IDs.index(ID), [ID, a, b])

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] > 1)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    postfirstsummary.insert(IDs.index(ID), [ID, a, b])

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] > 2)]
    a=qwtemp.groupby('ID').agg({'PTS':'sum','FGM':'sum', 'FGA':'sum', 'FG3M':'sum', 'FG3A':'sum', 'FG3_PCT':'mean', 'FTM':'sum', 'FTA':'sum', 'OREB':'sum', 'DREB':'sum', 'REB':'sum', 'AST':'sum', 'TOV':'sum', 'STL':'sum', 'BLK':'sum', 'BLKA':'sum','PF':'sum', 'PFD':'sum','FT_PCT':'mean','FG_PCT':'mean'})
    b = (qwtemp['Gamedate']).unique()
    posthalfsummary.insert(IDs.index(ID), [ID, a, b])

qwdata = [x[1] for x in threequartersummary]
if len(qwdata)==0:
    qw3 = pd.DataFrame()
else:
    qw3 = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in threequartersummary]
    qw3['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in fourquartersummary]
if len(qwdata)==0:
    qw4 = pd.DataFrame()
else:
    qw4 = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in fourquartersummary]
    qw4['Gamedate'] = pd.DataFrame(qwgamedates)
#qw4 = qw4.add_prefix('P')
#qw4['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in quartersummary]
if len(qwdata)==0:
    qw1 = pd.DataFrame()
else:
    qw1 = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in quartersummary]
    qw1['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in postfirstsummary]
if len(qwdata)==0:
    qw1p = pd.DataFrame()
else:
    qw1p = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in postfirstsummary]
    qw1p['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in halfsummary]
if len(qwdata)==0:
    qw2 = pd.DataFrame()
else:
    qw2 = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in halfsummary]
    qw2['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in posthalfsummary]
if len(qwdata)==0:
    qw2p = pd.DataFrame()
else:
    qw2p = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in halfsummary]
    qw2p['Gamedate'] = pd.DataFrame(qwgamedates)

qwdata = [x[1] for x in quartersummary]
if len(qwdata)==0:
    qw1 = pd.DataFrame()
else:
    qw1 = pd.concat(qwdata).reset_index()
    qwgamedates = [x[2] for x in quartersummary]
    qw1['Gamedate'] = pd.DataFrame(qwgamedates)


qw1_old=qw1_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw2_old=qw2_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw3_old=qw3_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw4_old=qw4_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw1p_old=qw1p_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw2p_old=qw2p_old[['ID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FG3_PCT',
       'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'FT_PCT', 'FG_PCT']]

qw1 = qw1_old.append(qw1)
qw1 = qw1.drop_duplicates()
qw1.to_csv(directory1+'qw1.csv')

qw2 = qw2_old.append(qw2)
qw2 = qw2.drop_duplicates()
qw2.to_csv(directory1+'qw2.csv')

qw3 = qw3_old.append(qw3)
qw3 = qw3.drop_duplicates()
qw3.to_csv(directory1+'qw3.csv')

qw4 = qw4_old.append(qw4)
qw4 = qw4.drop_duplicates()
qw4.to_csv(directory1+'qw4.csv')
qw4 = qw4.add_prefix('P')

qw1p = qw1p_old.append(qw1p)
qw1p.to_csv(directory1+'qw1p.csv')
qw1p = qw1p.add_prefix('P')

qw2p = qw2p_old.append(qw2p)
qw2p.to_csv(directory1+'qw2p.csv')
qw2p = qw2p.add_prefix('P')

qaddons=quarterdata[['ID', 'Gamedate', 'TEAM_ABBREVIATION', 'Season Year', 'Team Rank', 'H/A', 'HomeDaysRest', 'HomeScheduleID',
'TotalScore', 'away_team_score','home_team_score', 'Opponent', 'OpponentID',
'Conference', 'Opponent Conference', 'ConferenceBinary','OpponentConferenceBinary','>40', '>30', '>20', '>15', '>10','>5','TotalScore']]

qaddons=qaddons.drop_duplicates()

qw1['ID']=qw1['ID']+'1'
qw2['ID']=qw2['ID']+'2'
qw3['ID']=qw3['ID']+'3'
qw4['PID']=qw4['PID']+'3'
qw2p['PID']=qw2p['PID']+'2'
qw1p['PID']=qw1p['PID']+'1'

qw1=pd.merge(qw1,qaddons,left_on=qw1['ID'],right_on=qaddons['ID'],how='left').rename(columns={'ID_x': 'ID'})

qw2=pd.merge(qw2,qaddons,left_on=qw2['ID'],right_on=qaddons['ID'],how='left').rename(columns={'ID_x': 'ID'})

qw3=pd.merge(qw3,qaddons,left_on=qw3['ID'],right_on=qaddons['ID'],how='left').rename(columns={'ID_x': 'ID'})

qw4=pd.merge(qw4,qaddons,left_on=qw4['PID'],right_on=qaddons['ID'],how='left')

qw2p=pd.merge(qw2p,qaddons,left_on=qw2p['PID'],right_on=qaddons['ID'],how='left')

qw1p=pd.merge(qw1p,qaddons,left_on=qw1p['PID'],right_on=qaddons['ID'],how='left')

q4data=qw4
q4data['HOME_TEAM_ABBREVIATION']=q4data['PID'].str[:3]
q4 = []
tempscoring=pd.DataFrame()
for s in list(q4data['Season Year'].unique()):
    for i in list(q4data['HOME_TEAM_ABBREVIATION'].unique()):
        tempscoring = q4data[(q4data['HOME_TEAM_ABBREVIATION'] == i) & ((q4data['Season Year'] == s))].dropna().reset_index()
        tempscoring.sort_values('Gamedate')
        for j in range(0, len(tempscoring)):
            if j>0:
                tempmeanframe = tempscoring.loc[0:j-1]
                tempmeanframe = tempmeanframe.groupby('HOME_TEAM_ABBREVIATION').agg({'PPTS': 'mean', 'PFGM': 'mean', 'PFGA': 'mean', 'PFG_PCT': 'mean', 'PFG3M': 'mean', 'PFG3A': 'mean',
                     'PFG3_PCT': 'mean','PFTM': 'mean', 'PFTA': 'mean', 'PFT_PCT': 'mean', 'POREB': 'mean', 'PDREB': 'mean', 'PREB': 'mean',
                     'PAST': 'mean','PTOV': 'mean', 'PSTL': 'mean', 'PBLK': 'mean', 'PBLKA': 'mean', 'PPF': 'mean', 'PPFD': 'mean'})
                tempmeanframe['PID']=tempscoring['PID'][j]
                q4.insert(0, [tempmeanframe])

q4ytdmean = [x[0] for x in q4]
q4ytdmean = pd.concat(q4ytdmean).reset_index()
q4ytdmean=q4ytdmean.drop_duplicates()

qw3=qw3.drop('key_0',axis=1)
qw3final = pd.merge(qw3,q4ytdmean,left_on=qw3['ID'],right_on=q4ytdmean['PID'],how='left')

q2data=qw2p
q2data['HOME_TEAM_ABBREVIATION']=q2data['PID'].str[:3]
q2 = []
tempscoring=pd.DataFrame()
for s in list(q2data['Season Year'].unique()):
    for i in list(q2data['HOME_TEAM_ABBREVIATION'].unique()):
        tempscoring = q2data[(q2data['HOME_TEAM_ABBREVIATION'] == i) & ((q2data['Season Year'] == s))].dropna().reset_index()
        tempscoring.sort_values('Gamedate')
        for j in range(0, len(tempscoring)):
            if j>0:
                tempmeanframe = tempscoring.loc[0:j-1]
                tempmeanframe = tempmeanframe.groupby('HOME_TEAM_ABBREVIATION').agg({'PPTS': 'mean', 'PFGM': 'mean', 'PFGA': 'mean', 'PFG_PCT': 'mean', 'PFG3M': 'mean', 'PFG3A': 'mean',
                     'PFG3_PCT': 'mean','PFTM': 'mean', 'PFTA': 'mean', 'PFT_PCT': 'mean', 'POREB': 'mean', 'PDREB': 'mean', 'PREB': 'mean',
                     'PAST': 'mean','PTOV': 'mean', 'PSTL': 'mean', 'PBLK': 'mean', 'PBLKA': 'mean', 'PPF': 'mean', 'PPFD': 'mean'})
                tempmeanframe['PID']=tempscoring['PID'][j]
                q2.insert(0, [tempmeanframe])

q2ytdmean = [x[0] for x in q2]
q2ytdmean = pd.concat(q2ytdmean).reset_index()
q2ytdmean=q2ytdmean.drop_duplicates()

qw2=qw2.drop('key_0',axis=1)
qw2final = pd.merge(qw2,q2ytdmean,left_on=qw2['ID'],right_on=q2ytdmean['PID'],how='left')

q1data=qw1p
q1data['HOME_TEAM_ABBREVIATION']=q1data['PID'].str[:3]
q1 = []
tempscoring=pd.DataFrame()
for s in list(q1data['Season Year'].unique()):
    for i in list(q1data['HOME_TEAM_ABBREVIATION'].unique()):
        tempscoring = q1data[(q1data['HOME_TEAM_ABBREVIATION'] == i) & ((q1data['Season Year'] == s))].dropna().reset_index()
        tempscoring.sort_values('Gamedate')
        for j in range(0, len(tempscoring)):
            if j>0:
                tempmeanframe = tempscoring.loc[0:j-1]
                tempmeanframe = tempmeanframe.groupby('HOME_TEAM_ABBREVIATION').agg({'PPTS': 'mean', 'PFGM': 'mean', 'PFGA': 'mean', 'PFG_PCT': 'mean', 'PFG3M': 'mean', 'PFG3A': 'mean',
                     'PFG3_PCT': 'mean','PFTM': 'mean', 'PFTA': 'mean', 'PFT_PCT': 'mean', 'POREB': 'mean', 'PDREB': 'mean', 'PREB': 'mean',
                     'PAST': 'mean','PTOV': 'mean', 'PSTL': 'mean', 'PBLK': 'mean', 'PBLKA': 'mean', 'PPF': 'mean', 'PPFD': 'mean'})
                tempmeanframe['PID']=tempscoring['PID'][j]
                q1.insert(0, [tempmeanframe])

q1ytdmean = [x[0] for x in q1]
q1ytdmean = pd.concat(q1ytdmean).reset_index()
q1ytdmean=q1ytdmean.drop_duplicates()

qw1=qw1.drop('key_0',axis=1)
qw1final = pd.merge(qw1,q1ytdmean,left_on=qw1['ID'],right_on=q1ytdmean['PID'],how='left')

#We sort the values for each team by date for away and home games
teamscoring = schedule.sort_values(['AWAY_TEAM_ABBREVIATION', 'start_time']).filter(
    ['away_team_score', 'AWAY_TEAM_ABBREVIATION', 'TotalScore', 'start_time', 'Season Year']).rename(
    columns={'away_team_score': 'home_team_score', 'AWAY_TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}).append(
    schedule.sort_values(['HOME_TEAM_ABBREVIATION', 'start_time']).filter(
        ['home_team_score', 'HOME_TEAM_ABBREVIATION', 'TotalScore', 'start_time', 'Season Year'])).sort_values(
    ['HOME_TEAM_ABBREVIATION', 'start_time']).rename(
    columns={'home_team_score': 'Team_Score', 'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'}).drop_duplicates()

teamscoring2 = pd.DataFrame()

for s in list(teamscoring['Season Year'].unique()):
    for i in list(teamscoring['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == i) & (teamscoring['Season Year'] == s)].drop(
                'level_0', axis=1).reset_index()
        except:
            tempscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == i) & (teamscoring['Season Year'] == s)].reset_index()
        try:
            tempscoring['Last_Game_Score'] = tempscoring['Team_Score'].shift(1)
            tempscoring['Last_Game_Score'].iloc[0] = np.mean(tempscoring[tempscoring['Season Year'] == s]['Team_Score'])
            teamscoring2 = teamscoring2.append(tempscoring)
        except:
            pass

teamscoring = teamscoring2.drop_duplicates()
teamscoring['avgID'] = teamscoring['TEAM_ABBREVIATION'] + teamscoring['start_time']
teamscoring.dropna()

avgrollteamscore = []
for s in list(teamscoring['Season Year'].unique()):
    for i in list(teamscoring['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamscoring[
                (teamscoring['TEAM_ABBREVIATION'] == i) & ((teamscoring['Season Year'] == s))].reset_index()
        except:
            tempscoring = teamscoring[
                (teamscoring['TEAM_ABBREVIATION'] == i) & ((teamscoring['Season Year'] == s))].drop('level_0',axis=1).reset_index()
        for j in range(0, len(tempscoring)):
            tempmean = tempscoring
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['Team_Score'])
            avgrollteamscore.insert(0, [i, s, gamedate, tempmean])

avgrollteamscore = pd.DataFrame(avgrollteamscore).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

avgrollteamscore['Month'] = avgrollteamscore['Date'].str[5:7].apply(int)

avgrollteamscore['avgID'] = avgrollteamscore['TEAM_ABBREVIATION'] + avgrollteamscore['Date']

avgrollteamscore = avgrollteamscore.filter(['avgID', 'Avg Score']).drop_duplicates()

totalscoring = schedule.sort_values(['HOME_TEAM_ABBREVIATION', 'start_time']).filter(
    ['HOME_TEAM_ABBREVIATION', 'TotalScore', 'start_time', 'Season Year']).rename(
    columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'}).drop_duplicates().append((schedule.filter(
    ['AWAY_TEAM_ABBREVIATION', 'TotalScore', 'start_time', 'Season Year']).sort_values(
    ['AWAY_TEAM_ABBREVIATION', 'start_time']).rename(
    columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'}))).drop_duplicates()

avgrolltotalscore = []
for s in list(totalscoring['Season Year'].unique()):
    for i in list(totalscoring['TEAM_ABBREVIATION'].unique()):
        tempscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == i) & (
                totalscoring['Season Year'] == s)].reset_index().drop_duplicates().drop('index', axis=1)
        for j in range(0, len(tempscoring)):
            tempmean = tempscoring
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['TotalScore'])
            avgrolltotalscore.insert(0, [i, s, gamedate, tempmean])

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

avgrolltotalscore['Month'] = avgrolltotalscore['Date'].str[5:7].apply(int)  # use for gamedate

avgrolltotalscore['avgID'] = avgrolltotalscore['TEAM_ABBREVIATION'] + avgrolltotalscore['Date']

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score', 'Month']).drop_duplicates()

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

schedule['home_teamID'] = schedule['HOME_TEAM_ABBREVIATION'] + schedule['start_time']
schedule['away_teamID'] = schedule['AWAY_TEAM_ABBREVIATION'] + schedule['start_time']
schedule['TotalScore'] = schedule['home_team_score'] + schedule['away_team_score']
opponent = schedule.filter(['home_teamID', 'Season Year', 'TotalScore', 'away_teamID'])

# 2023 Schedule
Schedule2022 = pd.DataFrame(client.season_schedule(season_end_year=2022))
Schedule2022['Season Year'] = '2022'
Schedule2022['start_time'] = (Schedule2022['start_time'] - timedelta(hours=12)).apply(lambda x: x.strftime('%Y-%m-%d'))
Schedule2022['away_team'] = Schedule2022['away_team'].astype('S')
awayteams=Schedule2022['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
Schedule2022['away_team'] = [x[1] for x in awayteams]
Schedule2022['home_team'] = Schedule2022['home_team'].astype('S')
hometeams = Schedule2022['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
Schedule2022['home_team'] = [x[1] for x in hometeams]
Schedule2022 = pd.merge(Schedule2022, teammapping, left_on=Schedule2022['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
Schedule2022 = Schedule2022.drop('key_0', axis=1)
Schedule2022 = pd.merge(Schedule2022, teammapping, left_on=Schedule2022['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
gamestoday=Schedule2022[Schedule2022['start_time']==str(datetime.datetime.today().date())]
gamestoday=gamestoday.reset_index().drop('index',axis=1)

completetesting1 = pd.DataFrame()
predicted=[]

completetesting2 = pd.DataFrame()
predicted=[]

completetesting3 = pd.DataFrame()
predicted=[]


today = datetime.datetime.today().strftime('%Y-%m-%d')

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com'}
game_url = requests.get("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")

# consider number of games
numgames = len(game_url.json()['scoreboard']['games'])

# ID to use in the score_url
gameidlist = []
for n in range(0, numgames):
    gameid = game_url.json()['scoreboard']['games'][n]['gameId']
    gameidlist.insert(len(gameidlist), gameid)

# Get Matchups
# TEAM_ABBREVIATION
matchups = []
for n in range(0, numgames):
    awayteam = game_url.json()['scoreboard']['games'][n]['awayTeam']['teamTricode']
    hometeam = game_url.json()['scoreboard']['games'][n]['homeTeam']['teamTricode']
    gameid = game_url.json()['scoreboard']['games'][n]['gameId']
    gamestatus = game_url.json()['scoreboard']['games'][n]['gameStatusText']
    matchups.insert(len(matchups), [awayteam, hometeam, gameid, gamestatus])
    matchupsdf = pd.DataFrame(matchups)
    matchupsdf = matchupsdf.rename(columns={0: 'Away', 1: 'Home', 2: 'GameID', 3: 'Status'})

# Get Scores
livematchups = matchupsdf[~matchupsdf['Status'].str.contains('pm ET')]['GameID']
gamestats = pd.DataFrame()


scoreslist = []
for gameid in livematchups:
    score_url = requests.get("https://cdn.nba.com/static/json/liveData/boxscore/boxscore_" + gameid + ".json")
    numplayersaway = len(score_url.json()['game']['awayTeam']['players'])
    numplayershome = len(score_url.json()['game']['homeTeam']['players'])
    playerstats = pd.DataFrame()
    playerstatstemp2 = pd.DataFrame()

    for n in range(0, min(numplayershome, numplayersaway)):
        playerhometemp = pd.DataFrame(score_url.json()['game']['homeTeam']['players'][n]['statistics'], index=[0])
        playerhometemp['H/A'] = 'Home'
        playerawaytemp = pd.DataFrame(score_url.json()['game']['awayTeam']['players'][n]['statistics'], index=[0])
        playerawaytemp['H/A'] = 'Away'
        awayteam = game_url.json()['scoreboard']['games'][list(livematchups).index(gameid)]['awayTeam'][
            'teamTricode']
        hometeam = game_url.json()['scoreboard']['games'][list(livematchups).index(gameid)]['homeTeam'][
            'teamTricode']
        playerhometemp['Team'] = awayteam
        playerawaytemp['Team'] = hometeam
        playerhometemp['Matchup']=str(hometeam)+"vs"+str(awayteam)
        playerawaytemp['Matchup'] = str(hometeam) + "vs" + str(awayteam)
        playerstatstemp = playerhometemp.append(playerawaytemp)
        gamestatus = game_url.json()['scoreboard']['games'][list(livematchups).index(gameid)]['gameStatusText']
        playerstatstemp['Time'] = gamestatus
        playerstatstemp['Date'] = today
        playerstatstemp = playerstatstemp.rename(
            columns={'assists': 'AST', 'blocks': 'BLK', 'blocksReceived': 'BLKA', 'fieldGoalsAttempted': 'FGA',
                     'fieldGoalsMade': 'FGM', 'fieldGoalsPercentage': 'FG_PCT', 'freeThrowsAttempted': 'FTA',
                     'freeThrowsMade': 'FTM', 'freeThrowsPercentage': 'FT_PCT', 'reboundsTotal': 'REB',
                     'reboundsDefensive': 'DREB', 'reboundsOffensive': 'OREB', 'steals': 'STL',
                     'threePointersAttempted': 'FG3A', 'threePointersMade': 'FG3M',
                     'threePointersPercentage': 'FG3_PCT', 'foulsPersonal': 'PF', 'points': 'PTS',
                     'turnovers': 'TOV', 'plusMinusPoints': '+/-'})
        playerstatstemp['>40'] = playerstatstemp['PTS'] > 40
        playerstatstemp['>40'] = (np.where(playerstatstemp['>40'] == True, 1, 0))
        playerstatstemp['>30'] = playerstatstemp['PTS'] > 30
        playerstatstemp['>30'] = (np.where(playerstatstemp['>30'] == True, 1, 0))
        playerstatstemp['>20'] = playerstatstemp['PTS'] > 20
        playerstatstemp['>20'] = (np.where(playerstatstemp['>20'] == True, 1, 0))
        playerstatstemp['>15'] = playerstatstemp['PTS'] > 15
        playerstatstemp['>15'] = (np.where(playerstatstemp['>15'] == True, 1, 0))
        playerstatstemp['>10'] = playerstatstemp['PTS'] > 10
        playerstatstemp['>10'] = (np.where(playerstatstemp['>10'] == True, 1, 0))
        playerstatstemp['>5'] = playerstatstemp['PTS'] > 5
        playerstatstemp['>5'] = (np.where(playerstatstemp['>5'] == True, 1, 0))
        playerstatstemp2 = playerstatstemp2.append(playerstatstemp)
    playerstats = playerstatstemp2.groupby(['H/A', 'Team', 'Time', 'Date','Matchup']).agg(
        {'AST': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'FGA': 'sum', 'FGM': 'sum', 'FG_PCT': 'mean', 'FTA': 'sum',
         'FTM': 'sum', 'FT_PCT': 'mean', 'REB': 'sum', 'DREB': 'sum', 'OREB': 'sum', 'STL': 'sum', 'FG3A': 'sum',
         'FG3M': 'sum', 'FG3_PCT': 'mean', 'PF': 'sum', 'PTS': 'sum', 'TOV': 'sum', '+/-': 'sum', '>40': 'sum',
         '>30': 'sum', '>20': 'sum', '>15': 'sum', '>10': 'sum', '>5': 'sum'})
    gamestats = gamestats.append(playerstats)
gamestats.reset_index()

gamestats1 = gamestats[(gamestats['Time']=='End Q1')]
gamestats2 = gamestats[(gamestats['Time']=='Half')]
gamestats3 = gamestats[(gamestats['Time']=='End Q3')]

matchupslive1=list(gamestats1['Matchup'].unique())
matchupslive2=list(gamestats2['Matchup'].unique())
matchupslive3=list(gamestats3['Matchup'].unique())

#live data for q2
for match in matchupslive2:
    testing = pd.DataFrame()
    str(datetime.datetime.today().date())
    home = gamestats2[(gamestats2['H/A']=='Home')&(gamestats2['Matchup']==match)][j]
    opponent = gamestats2[(gamestats2['H/A']=='Away')&(gamestats2['Matchup']==match)][j]
    date_format = "%Y-%m-%d"
    teamstanding1 = standing[(standing['Index'] == home) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    teamstanding2 = standing[(standing['Index'] == opponent) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    testing = pd.concat([testing, teamstanding2], axis=1).drop('index', axis=1)
    testing = testing.rename(columns={'Team Rank': 'Opponent Rank'})
    testing = pd.concat([testing, teamstanding1], axis=1).drop('index', axis=1)
    a = datetime.datetime.today()
    b = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == home]['Gamedate']), date_format)
    c = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == opponent]['Gamedate']), date_format)
    delta = a - b
    delta2 = a - c
    testing['HHomeDaysRest'] = min(delta.days, 5)
    testing['AHomeDaysRest'] = min(delta2.days, 5)

    if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':

        testing['HConferenceBinary'] = 1
    else:
        testing['HConferenceBinary'] = 0

    if conference[conference['Team'] == opponent]['Conference'].reset_index()['Conference'][0] == 'E':
        testing['HOpponentConferenceBinary'] = 1
    else:
        testing['HOpponentConferenceBinary'] = 0

    testing['S2015'] = 0
    testing['S2016'] = 0
    testing['S2017'] = 0
    testing['S2018'] = 0
    testing['S2019'] = 0
    testing['S2020'] = 0
    testing['S2021'] = 0
    testing['S2022'] = 0
    testing['S2023'] = 1

    testteamscoring = teamscoring[
        (teamscoring['TEAM_ABBREVIATION'] == home) & ((teamscoring['Season Year'] == currentyear))].reset_index()
    testing['HAverage_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == home) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['HAverage_Team_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])
    testteamscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == opponent) & (
    (teamscoring['Season Year'] == currentyear))].reset_index()
    testing['AAverage_Opp_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == opponent) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['AAverage_Opp_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])

    testing['HTeam_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
        schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]
    testing['AOpponent_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
        columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]

    testinghome=gamestats2[gamestats2['Team']==home].rename(columns={'FTA':'HFTA', 'FG3_PCT':'HFG3_PCT','PTS':'HPTS','STL':'HSTL','REB':'HREB','FGM':'HFGM','FT_PCT':'HFT_PCT','FGA':'HFGA','FG3A':'HFG3A', 'FG3M':'HFG3M', 'PF':'HPF', 'FG_PCT':'HFG_PCT', 'AST':'HAST', 'OREB':'HOREB','BLK':'HBLK', 'DREB':'HDREB','TOV':'HTOV', 'FTM':'HFTM','BLKA':'HBLKA', 'PFD':'HPFD'})
    testingaway = gamestats2[gamestats2['Team'] == opponent].rename(columns={'FTA':'AFTA', 'FG3_PCT':'AFG3_PCT','PTS':'APTS','STL':'ASTL','REB':'AREB','FGM':'AFGM','FT_PCT':'AFT_PCT','FGA':'AFGA','FG3A':'AFG3A', 'FG3M':'AFG3M', 'PF':'APF', 'FG_PCT':'AFG_PCT', 'AST':'AAST', 'OREB':'AOREB','BLK':'ABLK', 'DREB':'ADREB','TOV':'ATOV', 'FTM':'AFTM','BLKA':'ABLKA', 'PFD':'APFD'})

    q2ytdmeanhome=q2ytdmean[q2ytdmean['Team']==home]['HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST','HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA','HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M']
    q2ytdmeanaway=q2ytdmean[q2ytdmean['Team']==opponent]['APFD','APSTL','APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST','APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA','APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M']

    testing = pd.concat([testing, testingaway], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, testinghome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q2ytdmeanhome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q2ytdmeanaway], axis=1).drop('index', axis=1)
    testing['Playoffs']=0
    testing['HomeTeam']=home
    completetesting = completetesting.append(testing)

#live data forq1
for match in matchupslive1:
    testing = pd.DataFrame()
    str(datetime.datetime.today().date())
    home = gamestats1[(gamestats1['H/A']=='Home')&(gamestats1['Matchup']==match)][j]
    opponent = gamestats1[(gamestats1['H/A']=='Away')&(gamestats1['Matchup']==match)][j]
    date_format = "%Y-%m-%d"
    teamstanding1 = standing[(standing['Index'] == home) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    teamstanding2 = standing[(standing['Index'] == opponent) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    testing = pd.concat([testing, teamstanding2], axis=1).drop('index', axis=1)
    testing = testing.rename(columns={'Team Rank': 'Opponent Rank'})
    testing = pd.concat([testing, teamstanding1], axis=1).drop('index', axis=1)
    a = datetime.datetime.today()
    b = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == home]['Gamedate']), date_format)
    c = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == opponent]['Gamedate']), date_format)
    delta = a - b
    delta2 = a - c
    testing['HHomeDaysRest'] = min(delta.days, 5)
    testing['AHomeDaysRest'] = min(delta2.days, 5)

    if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':

        testing['HConferenceBinary'] = 1
    else:
        testing['HConferenceBinary'] = 0

    if conference[conference['Team'] == opponent]['Conference'].reset_index()['Conference'][0] == 'E':
        testing['HOpponentConferenceBinary'] = 1
    else:
        testing['HOpponentConferenceBinary'] = 0

    testing['S2015'] = 0
    testing['S2016'] = 0
    testing['S2017'] = 0
    testing['S2018'] = 0
    testing['S2019'] = 0
    testing['S2020'] = 0
    testing['S2021'] = 0
    testing['S2022'] = 0
    testing['S2023'] = 1

    testteamscoring = teamscoring[
        (teamscoring['TEAM_ABBREVIATION'] == home) & ((teamscoring['Season Year'] == currentyear))].reset_index()
    testing['HAverage_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == home) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['HAverage_Team_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])
    testteamscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == opponent) & (
    (teamscoring['Season Year'] == currentyear))].reset_index()
    testing['AAverage_Opp_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == opponent) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['AAverage_Opp_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])

    testing['HTeam_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
        schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]
    testing['AOpponent_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
        columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]

    testinghome=gamestats1[gamestats1['Team']==home].rename(columns={'FTA':'HFTA', 'FG3_PCT':'HFG3_PCT','PTS':'HPTS','STL':'HSTL','REB':'HREB','FGM':'HFGM','FT_PCT':'HFT_PCT','FGA':'HFGA','FG3A':'HFG3A', 'FG3M':'HFG3M', 'PF':'HPF', 'FG_PCT':'HFG_PCT', 'AST':'HAST', 'OREB':'HOREB','BLK':'HBLK', 'DREB':'HDREB','TOV':'HTOV', 'FTM':'HFTM','BLKA':'HBLKA', 'PFD':'HPFD'})
    testingaway = gamestats1[gamestats1['Team'] == opponent].rename(columns={'FTA':'AFTA', 'FG3_PCT':'AFG3_PCT','PTS':'APTS','STL':'ASTL','REB':'AREB','FGM':'AFGM','FT_PCT':'AFT_PCT','FGA':'AFGA','FG3A':'AFG3A', 'FG3M':'AFG3M', 'PF':'APF', 'FG_PCT':'AFG_PCT', 'AST':'AAST', 'OREB':'AOREB','BLK':'ABLK', 'DREB':'ADREB','TOV':'ATOV', 'FTM':'AFTM','BLKA':'ABLKA', 'PFD':'APFD'})

    q1ytdmeanhome=q1ytdmean[q1ytdmean['Team']==home]['HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST','HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA','HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M']
    q1ytdmeanaway=q1ytdmean[q1ytdmean['Team']==opponent]['APFD','APSTL','APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST','APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA','APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M']

    testing = pd.concat([testing, testingaway], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, testinghome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q1ytdmeanhome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q1ytdmeanaway], axis=1).drop('index', axis=1)
    testing['Playoffs']=0
    testing['HomeTeam']=home
    completetesting1 = completetesting1.append(testing)

#live data for q3
for match in matchupslive3:
    testing = pd.DataFrame()
    str(datetime.datetime.today().date())
    home = gamestats3[(gamestats3['H/A']=='Home')&(gamestats3['Matchup']==match)][j]
    opponent = gamestats3[(gamestats3['H/A']=='Away')&(gamestats3['Matchup']==match)][j]
    date_format = "%Y-%m-%d"
    teamstanding1 = standing[(standing['Index'] == home) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    teamstanding2 = standing[(standing['Index'] == opponent) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    testing = pd.concat([testing, teamstanding2], axis=1).drop('index', axis=1)
    testing = testing.rename(columns={'Team Rank': 'Opponent Rank'})
    testing = pd.concat([testing, teamstanding1], axis=1).drop('index', axis=1)
    a = datetime.datetime.today()
    b = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == home]['Gamedate']), date_format)
    c = datetime.datetime.strptime(max(quarterdata[quarterdata['TEAM_ABBREVIATION'] == opponent]['Gamedate']), date_format)
    delta = a - b
    delta2 = a - c
    testing['HHomeDaysRest'] = min(delta.days, 5)
    testing['AHomeDaysRest'] = min(delta2.days, 5)

    if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':

        testing['HConferenceBinary'] = 1
    else:
        testing['HConferenceBinary'] = 0

    if conference[conference['Team'] == opponent]['Conference'].reset_index()['Conference'][0] == 'E':
        testing['HOpponentConferenceBinary'] = 1
    else:
        testing['HOpponentConferenceBinary'] = 0

    testing['S2015'] = 0
    testing['S2016'] = 0
    testing['S2017'] = 0
    testing['S2018'] = 0
    testing['S2019'] = 0
    testing['S2020'] = 0
    testing['S2021'] = 0
    testing['S2022'] = 0
    testing['S2023'] = 1

    testteamscoring = teamscoring[
        (teamscoring['TEAM_ABBREVIATION'] == home) & ((teamscoring['Season Year'] == currentyear))].reset_index()
    testing['HAverage_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == home) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['HAverage_Team_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])
    testteamscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == opponent) & (
    (teamscoring['Season Year'] == currentyear))].reset_index()
    testing['AAverage_Opp_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[0:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == opponent) & (
                totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)['TotalScore']
    testing['AAverage_Opp_Total_Score'] = np.mean(testscoring.loc[0:len(testscoring)])

    testing['HTeam_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
        schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]
    testing['AOpponent_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
        columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].dropna().iloc[0]

    testinghome=gamestats3[gamestats3['Team']==home].rename(columns={'FTA':'HFTA', 'FG3_PCT':'HFG3_PCT','PTS':'HPTS','STL':'HSTL','REB':'HREB','FGM':'HFGM','FT_PCT':'HFT_PCT','FGA':'HFGA','FG3A':'HFG3A', 'FG3M':'HFG3M', 'PF':'HPF', 'FG_PCT':'HFG_PCT', 'AST':'HAST', 'OREB':'HOREB','BLK':'HBLK', 'DREB':'HDREB','TOV':'HTOV', 'FTM':'HFTM','BLKA':'HBLKA', 'PFD':'HPFD'})
    testingaway = gamestats3[gamestats3['Team'] == opponent].rename(columns={'FTA':'AFTA', 'FG3_PCT':'AFG3_PCT','PTS':'APTS','STL':'ASTL','REB':'AREB','FGM':'AFGM','FT_PCT':'AFT_PCT','FGA':'AFGA','FG3A':'AFG3A', 'FG3M':'AFG3M', 'PF':'APF', 'FG_PCT':'AFG_PCT', 'AST':'AAST', 'OREB':'AOREB','BLK':'ABLK', 'DREB':'ADREB','TOV':'ATOV', 'FTM':'AFTM','BLKA':'ABLKA', 'PFD':'APFD'})

    q4ytdmeanhome=q4ytdmean[q4ytdmean['Team']==home]['HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST','HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA','HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M']
    q4ytdmeanaway=q4ytdmean[q4ytdmean['Team']==opponent]['APFD','APSTL','APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST','APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA','APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M']

    testing = pd.concat([testing, testingaway], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, testinghome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q4ytdmeanhome], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, q4ytdmeanaway], axis=1).drop('index', axis=1)
    testing['Playoffs']=0
    testing['HomeTeam']=home
    completetesting3 = completetesting3.append(testing)


#testingdata for q1
completetesting1=completetesting1[['HFTA', 'HFG3_PCT', 'HPTS', 'HSTL', 'HREB', 'HFGM',
       'HFT_PCT', 'HFGA', 'HFG3A', 'HFG3M', 'HPF', 'HFG_PCT', 'HAST', 'HOREB',
       'HBLK', 'HDREB', 'HTOV', 'HFTM', 'HBLKA', 'HPFD','S2015','S2016','S2017','S2018','S2019','S2020','S2021','S2022','S2023',
       'HTeam Rank', 'HHomeDaysRest',
       'HConferenceBinary', 'HOpponentConferenceBinary',
       'HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST',
       'HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA',
       'HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M', 'HTeam_Last_Game_Score',
       'HAverage_Team_Score', 'HAverage_Team_Total_Score', 'AFTA', 'AFG3_PCT', 'APTS', 'ASTL', 'AREB',
       'AFGM', 'AFT_PCT', 'AFGA', 'AFG3A', 'AFG3M', 'APF', 'AFG_PCT',
       'AAST', 'AOREB', 'ABLK', 'ADREB', 'ATOV', 'AFTM', 'ABLKA',
       'APFD','AHomeDaysRest','ATeam Rank', 'APSTL',
       'APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST',
       'APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA',
       'APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M',
        'AOpponent_Last_Game_Score','AAverage_Opp_Team_Score', 'AAverage_Opp_Total_Score','H>40', 'H>30', 'H>20', 'H>15', 'H>10', 'H>5','A>40', 'A>30', 'A>20', 'A>15', 'A>10', 'A>5']]

#testingdata for q2
completetesting2=completetesting2[['HFTA', 'HFG3_PCT', 'HPTS', 'HSTL', 'HREB', 'HFGM',
       'HFT_PCT', 'HFGA', 'HFG3A', 'HFG3M', 'HPF', 'HFG_PCT', 'HAST', 'HOREB',
       'HBLK', 'HDREB', 'HTOV', 'HFTM', 'HBLKA', 'HPFD','S2015','S2016','S2017','S2018','S2019','S2020','S2021','S2022','S2023',
       'HTeam Rank', 'HHomeDaysRest',
       'HConferenceBinary', 'HOpponentConferenceBinary',
       'HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST',
       'HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA',
       'HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M', 'HTeam_Last_Game_Score',
       'HAverage_Team_Score', 'HAverage_Team_Total_Score', 'AFTA', 'AFG3_PCT', 'APTS', 'ASTL', 'AREB',
       'AFGM', 'AFT_PCT', 'AFGA', 'AFG3A', 'AFG3M', 'APF', 'AFG_PCT',
       'AAST', 'AOREB', 'ABLK', 'ADREB', 'ATOV', 'AFTM', 'ABLKA',
       'APFD','AHomeDaysRest','ATeam Rank', 'APSTL',
       'APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST',
       'APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA',
       'APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M',
        'AOpponent_Last_Game_Score','AAverage_Opp_Team_Score', 'AAverage_Opp_Total_Score','H>40', 'H>30', 'H>20', 'H>15', 'H>10', 'H>5','A>40', 'A>30', 'A>20', 'A>15', 'A>10', 'A>5']]

#testingdata for q3
completetesting3=completetesting3[['HFTA', 'HFG3_PCT', 'HPTS', 'HSTL', 'HREB', 'HFGM',
       'HFT_PCT', 'HFGA', 'HFG3A', 'HFG3M', 'HPF', 'HFG_PCT', 'HAST', 'HOREB',
       'HBLK', 'HDREB', 'HTOV', 'HFTM', 'HBLKA', 'HPFD','S2015','S2016','S2017','S2018','S2019','S2020','S2021','S2022','S2023',
       'HTeam Rank', 'HHomeDaysRest',
       'HConferenceBinary', 'HOpponentConferenceBinary',
       'HPSTL', 'HPBLKA', 'HPFGA', 'HPTOV', 'HPFG_PCT', 'HPDREB', 'HPAST',
       'HPFT_PCT', 'HPBLK', 'HPFGM', 'HPFG3A', 'HPPF', 'HPPFD', 'HPFTA',
       'HPPTS', 'HPFG3_PCT', 'HPFTM', 'HPREB', 'HPOREB', 'HPFG3M', 'HTeam_Last_Game_Score',
       'HAverage_Team_Score', 'HAverage_Team_Total_Score', 'AFTA', 'AFG3_PCT', 'APTS', 'ASTL', 'AREB',
       'AFGM', 'AFT_PCT', 'AFGA', 'AFG3A', 'AFG3M', 'APF', 'AFG_PCT',
       'AAST', 'AOREB', 'ABLK', 'ADREB', 'ATOV', 'AFTM', 'ABLKA',
       'APFD','AHomeDaysRest','ATeam Rank', 'APSTL',
       'APBLKA', 'APFGA', 'APTOV', 'APFG_PCT', 'APDREB', 'APAST',
       'APFT_PCT', 'APBLK', 'APFGM', 'APFG3A', 'APPF', 'APPFD', 'APFTA',
       'APPTS', 'APFG3_PCT', 'APFTM', 'APREB', 'APOREB', 'APFG3M',
        'AOpponent_Last_Game_Score','AAverage_Opp_Team_Score', 'AAverage_Opp_Total_Score','H>40', 'H>30', 'H>20', 'H>15', 'H>10', 'H>5','A>40', 'A>30', 'A>20', 'A>15', 'A>10', 'A>5']]



completetesting=completetesting.reset_index().drop('index',axis=1)
completetesting=completetesting.reset_index().drop('index',axis=1)


