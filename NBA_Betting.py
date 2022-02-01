# Import Libraries
import requests
import random
import json
import math
import warnings
import pickle

from cffi.setuptools_ext import execfile
import undetected_chromedriver.v2 as uc

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime
from datetime import timedelta, date
from pandas import DataFrame
import selenium
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier  # For classification
from sklearn.decomposition import PCA
import statsmodels as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from threading import Timer

import basketball_reference_web_scraper
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import schedule

scaler = StandardScaler()

# Starts importing model data

# directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'

nbadata = pd.read_csv('nbadataytd.csv', low_memory=False)

nbadata = nbadata[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                   'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs']]

nbadata['Gamedate'] = nbadata['Gamedate'].astype(str)
nbadata['TEAM_ABBREVIATION'] = nbadata['TEAM_ABBREVIATION'].astype(str)

# We want to set ranks for the teams bsaed on wins and get the end result for each game.
# This is uses the sports reference api to acheive this.

standing = pd.DataFrame()
currentyear = '2022'

# 2022 Schedule
Schedule2022 = pd.DataFrame(client.season_schedule(season_end_year=2022))
Schedule2022['Season Year'] = '2022'

# 2021 Schedule
Schedule2021 = pd.DataFrame(client.season_schedule(season_end_year=2021))
Schedule2021['Season Year'] = '2021'

# 2020 Schedule
Schedule2020 = pd.DataFrame(client.season_schedule(season_end_year=2020))
Schedule2020['Season Year'] = '2020'

# 2019 Schedule
Schedule2019 = pd.DataFrame(client.season_schedule(season_end_year=2019))
Schedule2019['Season Year'] = '2019'

# Repeat
Schedule2018 = pd.DataFrame(client.season_schedule(season_end_year=2018))
Schedule2018['Season Year'] = '2018'
Schedule2017 = pd.DataFrame(client.season_schedule(season_end_year=2017))
Schedule2017['Season Year'] = '2017'
Schedule2016 = pd.DataFrame(client.season_schedule(season_end_year=2016))
Schedule2016['Season Year'] = '2016'
Schedule2015 = pd.DataFrame(client.season_schedule(season_end_year=2015))
Schedule2015['Season Year'] = '2015'
Schedule2014 = pd.DataFrame(client.season_schedule(season_end_year=2014))
Schedule2014['Season Year'] = '2014'
Schedule2013 = pd.DataFrame(client.season_schedule(season_end_year=2013))
Schedule2013['Season Year'] = '2013'
Schedule2012 = pd.DataFrame(client.season_schedule(season_end_year=2012))
Schedule2012['Season Year'] = '2012'
Schedule2011 = pd.DataFrame(client.season_schedule(season_end_year=2011))
Schedule2011['Season Year'] = '2011'
Schedule2010 = pd.DataFrame(client.season_schedule(season_end_year=2010))
Schedule2010['Season Year'] = '2010'
schedulelist = [Schedule2022, Schedule2021, Schedule2020, Schedule2019, Schedule2018, Schedule2017, Schedule2016,
                Schedule2015,
                Schedule2014, Schedule2013, Schedule2012, Schedule2011, Schedule2010]
Schedule2020.filter(['start_time', 'home_team_score'])

schedule = pd.DataFrame()
# Append each schedule and do appropriate conversions
schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(
    Schedule2018).append(
    Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014).append(
    Schedule2013).append(Schedule2012).append(Schedule2011).append(Schedule2010)
schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
# gamehist = schedule.loc[schedule['Season Year']!=currentyear]['start_time']
teammapping = pd.read_csv('dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())
seasonyears = ['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
# We want to add the new data to our original file, therefore we filter on the new season and compare the dates in our file to the games that have been played.
playoffslist = []
nbadatalist = []
notcaptured = []
dataytd = []
dates = pd.DataFrame()
datestemp = pd.DataFrame()

for s in seasonyears:
    scheduletemp = schedule[(schedule['Season Year'] == s)]
    dataytdtemp = scheduletemp.filter(['start_time', 'home_team_score']).dropna()['start_time'].drop_duplicates()
    dataytd.insert(seasonyears.index(s), dataytdtemp)
    dates = dates.append(dataytd[seasonyears.index(s)].tolist())
    # dates = dataytd[0].tolist()+dataytd[1].tolist()+dataytd[2].tolist()+dataytd[3].tolist()+dataytd[4].tolist()+dataytd[5].tolist()+dataytd[6].tolist()

captureddates = pd.DataFrame(nbadata['Gamedate'].unique())
datelist = [x for x in dates[0].tolist() if x not in captureddates[0].tolist()]
# leftover games are generally the playoff games.
datelist = [x for x in datelist if x > max(captureddates[0])]
# The following for loop reads the new data from nba.com into a list
for e in range(0, len(datelist)):
    try:
        year1 = datelist[e][:4]
        month1 = datelist[e][5:7]
        day1 = datelist[e][8:10]
        a = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom='
        if datelist[e] in dataytd[0].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[1].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[2].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[3].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[4].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[5].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[6].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[7].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        c = '&DateTo='
        d = '%2F'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.42 Safari/537.36',
            'x-nba-stats-origin': 'stats'}
        url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
        r = requests.get(url, headers=headers)
        numrecords = len(r.json()['resultSets'][0]['rowSet'])
        fields = r.json()['resultSets'][0]['headers']
        data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
        for i in range(0, numrecords):
            records = r.json()['resultSets'][0]['rowSet'][i]
            for j in range(0, len(records)):
                data.iloc[[i], [j]] = records[j]
        nbadatalist.insert(e, [data, datelist[e]])
    except:
        pass

for e in range(0, len(datelist)):
    try:
        year1 = datelist[e][:4]
        month1 = datelist[e][5:7]
        day1 = datelist[e][8:10]
        a = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom='
        if datelist[e] in dataytd[0].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[1].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[2].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[3].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[4].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[5].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[6].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        elif datelist[e] in dataytd[7].tolist():
            b = '&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='
        c = '&DateTo='
        d = '%2F'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
            'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/'}
        url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
        r = requests.get(url, headers=headers)
        numrecords = len(r.json()['resultSets'][0]['rowSet'])
        fields = r.json()['resultSets'][0]['headers']
        data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
        for i in range(0, numrecords):
            records = r.json()['resultSets'][0]['rowSet'][i]
            for j in range(0, len(records)):
                data.iloc[[i], [j]] = records[j]
        playoffslist.insert(e, [data, datelist[e]])
    except:
        pass

playoffsworking = pd.DataFrame()
playoffstemp = pd.DataFrame()

# Read the data from the website into a dataframe
for i in range(0, len(playoffslist)):
    playoffstemp = pd.DataFrame(playoffslist[i][0])
    playoffstemp['Gamedate'] = playoffslist[i][1]
    playoffsworking = playoffsworking.append(playoffstemp)

playoffsworking['Playoffs'] = 1

if not (playoffsworking.empty):
    playoffsdata = playoffsworking[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                                    'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                                    'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                                    'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                                    'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                                    'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                                    'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                                    'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                                    'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                                    'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                                    'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs']]
if (playoffsworking.empty):
    playoffsdata = pd.DataFrame()

playoffsdata = playoffsdata.append(playoffsworking)
playoffsdata = playoffsdata.reset_index()

nbadataworking = pd.DataFrame()
nbadatatemp = pd.DataFrame()

# Read the data from the website into a dataframe
for i in range(0, len(nbadatalist)):
    nbadatatemp = pd.DataFrame(nbadatalist[i][0])
    nbadatatemp['Gamedate'] = nbadatalist[i][1]
    nbadataworking = nbadataworking.append(nbadatatemp)

nbadataworking['Playoffs'] = 0

if not (nbadataworking.empty):
    nbadataworking = nbadataworking[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                                     'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                                     'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                                     'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                                     'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                                     'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                                     'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                                     'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                                     'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                                     'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                                     'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs']]

if (nbadataworking.empty):
    nbadatatworking = pd.DataFrame()

nbadata = nbadata.append(nbadataworking)
nbadataworking = nbadataworking.append(playoffsdata)
nbadata.to_csv('nbadataytd.csv')
nbadata = pd.read_csv('nbadataytd.csv',
                      low_memory=False)
nbadata = nbadata[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                   'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs']]

execfile('/Users/kabariquaye/PycharmProjects/pythonProject1/AdditionalData.py')
nbadataaddon = pd.read_csv('nbadataytdaddon.csv',
                           low_memory=False)
nbadata['TEAM_ID'] = nbadata['TEAM_ID'].astype(str)
nbadataaddon['ID'] = nbadataaddon['Gamedate'].astype(str) + nbadataaddon['TEAM_ID'].astype(str).str[:-2]
nbadataaddon = nbadataaddon[
    ['ID', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
     'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']]
nbadata['AddonID'] = nbadata['Gamedate'] + nbadata['TEAM_ID'].astype(str).str[:-2]
nbadata = pd.merge(nbadata, nbadataaddon, left_on=nbadata['AddonID'], right_on=nbadataaddon['ID'], how='left')
nbadata = nbadata.drop_duplicates()
nbadata = nbadata[nbadata['PLAYER_ID'] != 'PLAYER_NAME']
nbadata = nbadata[nbadata['PLAYER_ID'] != 'False']
nbadata = nbadata.dropna()
nbadata.reset_index().drop(['index', 'key_0'], axis=1)
nbadata = nbadata.drop(['AddonID', 'ID'], axis=1)

nbadata = nbadata[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'NBA_FANTASY_PTS', 'DD2', 'TD3', 'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK',
                   'TD3_RANK', 'CFID', 'CFPARAMS', 'Gamedate', 'Playoffs',
                   'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
                   'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']]

# Access to Website and then information pull of daily games
nbadata = nbadata.infer_objects()
nbadata['Gamedate'] = nbadata['Gamedate'].astype(str)
nbadata['TEAM_ABBREVIATION'] = nbadata['TEAM_ABBREVIATION'].astype(str)
nbaschedule = pd.DataFrame()
standingslist = []
standing = pd.DataFrame()
Schedule2022 = pd.DataFrame(client.season_schedule(season_end_year=2022))
Schedule2022['Season Year'] = '2022'
Schedule2021 = pd.DataFrame(client.season_schedule(season_end_year=2021))
Schedule2021['Season Year'] = '2021'
Schedule2020 = pd.DataFrame(client.season_schedule(season_end_year=2020))
Schedule2020['Season Year'] = '2020'
Schedule2019 = pd.DataFrame(client.season_schedule(season_end_year=2019))
Schedule2019['Season Year'] = '2019'
Schedule2018 = pd.DataFrame(client.season_schedule(season_end_year=2018))
Schedule2018['Season Year'] = '2018'
Schedule2017 = pd.DataFrame(client.season_schedule(season_end_year=2017))
Schedule2017['Season Year'] = '2017'
Schedule2016 = pd.DataFrame(client.season_schedule(season_end_year=2016))
Schedule2016['Season Year'] = '2016'
Schedule2015 = pd.DataFrame(client.season_schedule(season_end_year=2015))
Schedule2015['Season Year'] = '2015'
Schedule2014 = pd.DataFrame(client.season_schedule(season_end_year=2014))
Schedule2014['Season Year'] = '2014'

schedule = pd.DataFrame()

schedulelist = [Schedule2022, Schedule2021, Schedule2020, Schedule2019, Schedule2018, Schedule2017, Schedule2016,
                Schedule2015, Schedule2014]
# Schedule2021.filter(['start_time','home_team_score'])

schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(
    Schedule2018).append(
    Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)
schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
teammapping = pd.read_csv('dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())

seasonyears = ['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015']

seasonmerge = schedule.filter(['Season Year', 'start_time'])
seasonmerge = seasonmerge.drop_duplicates()
nbadata = pd.merge(nbadata, seasonmerge, left_on=nbadata['Gamedate'], right_on=seasonmerge['start_time'],
                   how='left').drop('start_time', axis=1)

# The following for loop develops the standings for each team and assigns H or A for home or away games.
from pandas.core.common import \
    SettingWithCopyWarning  # I need this to not get a billion warnings while completing the loop.

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

for s in seasonyears:
    scheduletemp = schedule[(schedule['Season Year'] == s) & (schedule['start_time'] <= (
        max(nbadata[(nbadata['Season Year'] == s) & nbadata['Playoffs'] == 0]['Gamedate'])))]
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
        teamschedule['home_schedule'] = np.where(teamschedule['home_schedule'].isnull(),
                                                 teamschedule['away_schedule'],
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
schedule = pd.merge(schedule, nbaschedule, left_on=schedule['HomeScheduleID'], right_on=nbaschedule['ScheduleID'],
                    how='left')
schedule = schedule.drop('Team', 1).drop('schedule', 1).drop('H/A', 1).drop('ScheduleID', 1)
schedule = schedule.rename(columns={schedule.columns[len(schedule.columns) - 1]: 'HomeDaysRest'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, nbaschedule, left_on=schedule['AwayScheduleID'], right_on=nbaschedule['ScheduleID'],
                    how='left')
schedule = schedule.drop('Team', 1).drop('schedule', 1).drop('H/A', 1).drop('ScheduleID', 1)
schedule = schedule.rename(columns={schedule.columns[len(schedule.columns) - 1]: 'AwayDaysRest'})
schedule = schedule.dropna()

# Merge to nbadata, rankings, home and away info, etc.
nbadata['StandingID'] = nbadata['TEAM_ABBREVIATION'] + nbadata['Season Year']
standing['Index'] = standing.index
standing['StandingID'] = standing['Index'] + standing['Season Year']
standingmerge = standing.filter(['StandingID', 'Team Rank'])
standingmerge = standingmerge.drop_duplicates()
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, standingmerge, left_on=nbadata['StandingID'], right_on=standingmerge['StandingID'],
                   how='left').drop('StandingID_y', axis=1).drop('StandingID_x', axis=1)
nbadata['ScheduleID'] = nbadata['Gamedate'] + nbadata['TEAM_ABBREVIATION']
nbaschedule['ScheduleID2'] = nbaschedule['schedule'] + nbaschedule['Team']
schedulemerge = nbaschedule.filter(['ScheduleID2', 'H/A'])
schedulemerge = schedulemerge.drop_duplicates()
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, schedulemerge, left_on=nbadata['ScheduleID'], right_on=schedulemerge['ScheduleID2'],
                   how='left').drop('ScheduleID2', axis=1).drop('ScheduleID', axis=1)
pp = pd.DataFrame()
teamdatalist = []
nbaschedule = nbaschedule[nbaschedule['schedule'].isin(nbadata['Gamedate'].unique())]

# A player ranking system that I developped to rank players for modelling.
# The original  model was going to rank players by their contribution by game and use that as a variable (top 5 player output)
# for i in list(nbadata['TEAM_ABBREVIATION'].unique()):
#     nbadatatemp = nbadata[nbadata['TEAM_ABBREVIATION'] == i]
#     for game in list(nbadatatemp['Gamedate'].unique()):
#         b = nbadatatemp[nbadatatemp['Gamedate'] == game]
#         a = b.groupby('PLAYER_NAME').agg(
#             {'AST': 'sum', 'PTS': 'sum', 'FGM': 'sum', 'FG3M': 'sum', 'FTM': 'sum', 'OREB': 'sum', 'FTA': 'sum',
#              'FGA': 'sum', 'FG3A': 'sum', 'FTA': 'sum', 'TOV': 'sum', 'BLK': 'sum', 'DREB': 'sum', 'STL': 'sum',
#              'PF': 'sum'})
#         a['FTPenalty'] = -(a['FTA'] - a['FTM']) / 2
#         a.drop('FTA', 1)
#         a['FGPenalty'] = -(a['FGA'] - a['FGM']) / 2
#         a.drop('FTA', 1)
#         a['FG3Penalty'] = -(a['FG3A'] - a['FG3M']) / 3
#         a.drop('FG3A', 1)
#         a['TOV'] = a['TOV'] * (-1)
#         a['PlayerRating'] = a.sum(axis=1)
#         a['PF'] = a['PF'] * (-1)
#         a['PlayerRating'] = a.sum(axis=1)
#         a['Player Rank'] = a['PlayerRating'].rank(ascending=False, method='first')
#         a = a.sort_values(by=['PlayerRating'], ascending=False)
#         a['TEAM'] = i
#         a['Gamedate'] = game
#         a['PLAYER_NAME'] = a.index
#         a['PlayerID'] = a['PLAYER_NAME'] + a['Gamedate']
#         pp = pp.append(a)
#         # insert(list(nbadatatemp['TEAM_ABBREVIATION'].unique()).index(i),a)
#
# # Create PlayerIDs and merge with the schedule dataframe
# nbadata['PlayerID'] = nbadata['PLAYER_NAME'] + nbadata['Gamedate']
# ppmerge = pp.filter(['PlayerID', 'Player Rank'])
# nbadata = nbadata.drop('key_0', axis=1)
# nbadata = pd.merge(nbadata, ppmerge, left_on=nbadata['PlayerID'], right_on=ppmerge['PlayerID'], how='left')

nbadata['ScheduleID'] = nbadata['Gamedate'] + nbadata['H/A'] + nbadata['TEAM_ABBREVIATION']
a = schedule.filter(['HomeScheduleID', 'TotalScore', 'HomeDaysRest', 'home_team_score'])
b = schedule.filter(['AwayScheduleID', 'TotalScore', 'AwayDaysRest', 'away_team_score'])
b = b.rename(columns={'AwayScheduleID': 'HomeScheduleID', 'AwayDaysRest': 'HomeDaysRest'})
schedulemerge = a.append(b)
schedulemerge = schedulemerge.drop_duplicates()
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, schedulemerge, left_on=nbadata['ScheduleID'], right_on=schedulemerge['HomeScheduleID'],
                   how='left')
nbadata['ScheduleID'] = nbadata['Gamedate'] + nbadata['H/A'] + nbadata['TEAM_ABBREVIATION']
a = schedule.filter(['HomeScheduleID', 'AwayScheduleID'])
b = schedule.filter(['AwayScheduleID', 'HomeScheduleID'])
b = b.rename(columns={'AwayScheduleID': 'HomeScheduleID', 'HomeScheduleID': 'AwayScheduleID'})
b = b.filter(['HomeScheduleID', 'AwayScheduleID'])
schedulemerge = a.append(b)
schedulemerge = schedulemerge.rename(columns={'HomeScheduleID': 'Team', 'AwayScheduleID': 'Opponent'})
schedulemerge = schedulemerge.drop_duplicates()
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, schedulemerge, left_on=nbadata['ScheduleID'], right_on=schedulemerge['Team'],
                   how='left').drop('ScheduleID', axis=1).drop('Team', axis=1)
schedulemerge = schedulemerge.rename(columns={'AwayScheduleID': 'Opponent'})
nbadata['Opponent'] = nbadata['Opponent'].str[-3:]
nbadata['OpponentID'] = nbadata['Opponent'] + nbadata['Season Year']
nbadata['SeasonID'] = nbadata['TEAM_ABBREVIATION'] + nbadata['Season Year']

opponentrank = nbadata.filter(['Season Year', 'Opponent'])
opponentrank = opponentrank.drop_duplicates().dropna()
opponentrank['SeasonID'] = opponentrank['Opponent'] + opponentrank['Season Year']
rankingtemp = nbadata.filter(['SeasonID', 'Team Rank']).drop_duplicates().dropna()
opponentrank = pd.merge(rankingtemp, opponentrank, left_on=rankingtemp['SeasonID'],
                        right_on=opponentrank['SeasonID'],
                        how='left').drop('SeasonID_x', axis=1)
opponentrank = opponentrank.filter(['SeasonID_y', 'Team Rank'])
opponentrank = opponentrank.rename(columns=({'SeasonID_y': 'SeasonID', 'Team Rank': 'Opponent Rank'}))
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, opponentrank, left_on=nbadata['OpponentID'], right_on=opponentrank['SeasonID'],
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
nbadata = pd.merge(nbadata, conference, left_on='TEAM_ABBREVIATION', right_on='Team', how='left').drop('Team',
                                                                                                       axis=1)
# assign opponent conference merge.
opponentconference = nbadata.filter(['Conference', 'TEAM_ABBREVIATION'])
opponentconference = opponentconference.drop_duplicates()
opponentconference = opponentconference.rename(
    columns=({'Conference': 'Opponent Conference', 'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION1'}))
nbadata = nbadata.drop('key_0', axis=1)
nbadata = pd.merge(nbadata, opponentconference, left_on=nbadata['Opponent'],
                   right_on=opponentconference['TEAM_ABBREVIATION1'], how='left').drop('TEAM_ABBREVIATION1', axis=1)
nbadata['ConferenceBinary'] = np.where(nbadata['Conference'] == 'W', 0, 1)
nbadata['OpponentConferenceBinary'] = np.where(nbadata['Opponent Conference'] == 'W', 0, 1)
nbadata['HomeDaysRest'] = nbadata['HomeDaysRest'].clip(upper=5)
nbadata['away_team_score'][np.isnan(nbadata['away_team_score'])] = 0
nbadata['home_team_score'][np.isnan(nbadata['home_team_score'])] = 0

nbadata = nbadata.drop('key_0', axis=1)

averaging = 3

teamavgH = []
teamavgHdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'H') & (nbadata['MIN'] > 0)]
    teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
    for team in teamtemp:
        a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
        gamesplayed = len(a['Gamedate'].unique())
        a = a.groupby(['TEAM_ABBREVIATION']).agg(
            {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
             'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum', 'AST_PCT': 'sum', 'AST_TO': 'sum',
             'AST_RATIO': 'sum', 'OREB_PCT': 'sum', 'DREB_PCT': 'sum', 'REB_PCT': 'sum',
             'TM_TOV_PCT': 'sum', 'EFG_PCT': 'sum', 'TS_PCT': 'sum', 'E_PACE': 'sum', 'PACE': 'sum',
             'PACE_PER40': 'sum', 'POSS': 'sum', 'PIE': 'sum'})  # 3PTs made may be important, FG as well
        a = a.div(gamesplayed)
        a = a.add_prefix('AvgH')
        a['Season'] = s
        a['Team'] = team
        teamavgHdf = teamavgHdf.append(a)

teamavgHdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'H') & (nbadata['MIN'] > 0)]
    teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
    for team in teamtemp:
        a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
        a = a.groupby(['TEAM_ABBREVIATION', 'Gamedate']).agg(
            {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
             'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum', 'AST_PCT': 'sum', 'AST_TO': 'sum',
             'AST_RATIO': 'sum', 'OREB_PCT': 'sum', 'DREB_PCT': 'sum', 'REB_PCT': 'sum',
             'TM_TOV_PCT': 'sum', 'EFG_PCT': 'sum', 'TS_PCT': 'sum', 'E_PACE': 'sum', 'PACE': 'sum',
             'PACE_PER40': 'sum', 'POSS': 'sum', 'PIE': 'sum'})  # 3PTs made may be important, FG as well
        a = a.add_prefix('AvgH').reset_index()
        a['Season'] = s
        a['Team'] = team
        teamavgHdf = teamavgHdf.append(a)

teamavgHdf = teamavgHdf.reset_index()

avgHomeStats = pd.DataFrame()

for s in list(teamavgHdf['Season'].unique()):
    for i in list(teamavgHdf['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamavgHdf[
                (teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgHdf['Season'] == s))].reset_index()
        except:
            tempscoring = teamavgHdf[(teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgHdf['Season'] == s))].drop(
                'level_0', axis=1).reset_index()
        for j in range(averaging + 1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging + 1):j - 1]
            gamedate = tempscoring['Gamedate'][j]
            tempmean = tempmean.groupby("Team", as_index=True)[
                'AvgHPF', 'AvgHFGM', 'AvgHBLK', 'AvgHFG3M', 'AvgHOREB', 'AvgHAST', 'AvgHREB', 'AvgHSTL', 'AvgHAST_PCT', 'AvgHAST_TO', 'AvgHAST_RATIO', 'AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
                'AvgHTM_TOV_PCT', 'AvgHEFG_PCT', 'AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE'].mean()
            tempmean = tempmean.reset_index()
            tempmean['Gamedate'] = gamedate
            tempmean['Season'] = s
            avgHomeStats = avgHomeStats.append(tempmean)

avgHomeStats = avgHomeStats.reset_index()

teamavgAdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'A') & (nbadata['MIN'] > 0)]
    teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
    for team in teamtemp:
        a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
        a = a.groupby(['TEAM_ABBREVIATION', 'Gamedate']).agg(
            {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
             'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum', 'AST_PCT': 'sum', 'AST_TO': 'sum',
             'AST_RATIO': 'sum', 'OREB_PCT': 'sum', 'DREB_PCT': 'sum', 'REB_PCT': 'sum',
             'TM_TOV_PCT': 'sum', 'EFG_PCT': 'sum', 'TS_PCT': 'sum', 'E_PACE': 'sum', 'PACE': 'sum',
             'PACE_PER40': 'sum', 'POSS': 'sum', 'PIE': 'sum'})  # 3PTs made may be important, FG as well
        a = a.add_prefix('AvgA')
        a['Season'] = s
        a['Team'] = team
        teamavgAdf = teamavgAdf.append(a)

teamavgAdf = teamavgAdf.reset_index()

avgAwayStats = pd.DataFrame()

for s in list(teamavgAdf['Season'].unique()):
    for i in list(teamavgAdf['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamavgAdf[
                (teamavgAdf['TEAM_ABBREVIATION'] == i) & ((teamavgAdf['Season'] == s))].reset_index()
        except:
            tempscoring = teamavgAdf[(teamavgAdf['TEAM_ABBREVIATION'] == i) & ((teamavgAdf['Season'] == s))].drop(
                'level_0', axis=1).reset_index()
        for j in range(averaging + 1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging + 1):j - 1]
            gamedate = tempscoring['Gamedate'][j]
            tempmean = tempmean.groupby("Team", as_index=True)[
                'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgAAST_PCT', 'AvgAAST_TO', 'AvgAAST_RATIO', 'AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
                'AvgATM_TOV_PCT', 'AvgAEFG_PCT', 'AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE'].mean()
            tempmean = tempmean.reset_index()
            tempmean['Gamedate'] = gamedate
            tempmean['Season'] = s
            avgAwayStats = avgAwayStats.append(tempmean)

avgAwayStats = avgAwayStats.reset_index()

avgAwayStats['avgID'] = avgAwayStats['Team'] + avgAwayStats['Gamedate']
avgHomeStats['avgID'] = avgHomeStats['Team'] + avgHomeStats['Gamedate']

avgAwayStats = avgAwayStats.drop(['Season', 'Gamedate'], axis=1)
avgHomeStats = avgHomeStats.drop(['Season', 'Gamedate'], axis=1)

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
            tempscoring = teamscoring[
                (teamscoring['TEAM_ABBREVIATION'] == i) & (teamscoring['Season Year'] == s)].drop(
                'level_0', axis=1).reset_index()
        except:
            tempscoring = teamscoring[
                (teamscoring['TEAM_ABBREVIATION'] == i) & (teamscoring['Season Year'] == s)].reset_index()
        try:
            tempscoring['Last_Game_Score'] = tempscoring['Team_Score'].shift(1)
            tempscoring['Last_Game_Score'].iloc[0] = np.mean(
                tempscoring[tempscoring['Season Year'] == s]['Team_Score'])
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
                (teamscoring['TEAM_ABBREVIATION'] == i) & ((teamscoring['Season Year'] == s))].drop('level_0',
                                                                                                    axis=1).reset_index()
        for j in range(6, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging + 1):j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['Team_Score'])
            avgrollteamscore.insert(0, [i, s, gamedate, tempmean])

avgrollteamscore = pd.DataFrame(avgrollteamscore).rename(
    columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

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
        for j in range(averaging + 1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging + 1):j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['TotalScore'])
            avgrolltotalscore.insert(0, [i, s, gamedate, tempmean])

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(
    columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

avgrolltotalscore['Month'] = avgrolltotalscore['Date'].str[5:7].apply(int)  # use for gamedate

avgrolltotalscore['avgID'] = avgrolltotalscore['TEAM_ABBREVIATION'] + avgrolltotalscore['Date']

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score', 'Month']).drop_duplicates()

driver = uc.Chrome('chromedriver')  # need to have the right version of chromedriver installed on computer.
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
# The gamedate to check if the games are today
driver.maximize_window()
gamedatesi = driver.find_element_by_xpath('//*[@id="page"]/div[2]/ul/li[1]/h4').text
gamedatesi = gamedatesi + ' ' + str(datetime.datetime.today().year)
gamedatesi = datetime.datetime.strptime(gamedatesi, '%a %b %d %Y')
today = datetime.datetime.today().date()
today = datetime.datetime(today.year, today.month, today.day)
betgamesodds = []
betgameclick = []

# Actual Site Log-in
time.sleep(random.uniform(10, 20))
driver.find_element_by_xpath('//*[@id="app"]/div/span/div/div/div/span').click()
prairieaccounts = [['kabariq@gmail.com', '########'], ['lcao0ca', 'nba-algo-betting-2021'],
                   ['hunter', 'nba-algo-betting-2021'], ['wchienenyanga', 'nba-algo-betting-2021']]
montrealaccounts = [[]]
torontoaccounts = [[]]

driver.find_element_by_xpath('//*[@id="LoginForm__account-name"]').send_keys(prairieaccounts[0][0])
time.sleep(random.uniform(5, 10))
driver.find_element_by_xpath('//*[@id="LoginForm__password"]').send_keys(prairieaccounts[0][1])

driver.find_element_by_xpath(
    '//*[@id="portal-target-overlays"]/div[1]/div[2]/div/div/div/div/div/form/button').click()
time.sleep(random.uniform(10, 20))
try:
    driver.find_element_by_xpath('//*[@id="portals-container"]/div/div[1]/div[1]/div[2]/span').click()
    time.sleep(random.uniform(5, 10))
except:
    pass
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
time.sleep(random.uniform(2, 5))
try:
    driver.find_element_by_xpath('//*[@id="portals-container"]/div/div[1]/div[1]/div[2]/span').click()
    time.sleep(random.uniform(5, 10))
except:
    pass
balance = driver.find_element_by_xpath('//*[@id="app"]/div/span/div/div/div/div[2]/div[1]/div/span').text
newbalance = balance
account = []
winslosses = []
account.insert(len(account), [balance, datetime.datetime.today().strftime('%Y-%m-%d')])
bot = pd.DataFrame(account).rename(columns={0: 'Account_Balance', 1: 'Date'})  # balance over time
botfile = pd.read_csv('BalanceOverTime.csv')

botfile = botfile[['Account_Balance', 'Date']]

# This should track daily activity
if max(botfile['Date']) < datetime.datetime.today().strftime('%Y-%m-%d'):
    botfile = botfile.append(bot)
else:
    pass

botfile = botfile[['Account_Balance', 'Date']]

botfile.to_csv('BalanceOverTime.csv')

if gamedatesi == today:

    for i in range(0, 20):
        try:
            driver.maximize_window()
            driver.implicitly_wait(random.uniform(6, 9))
            # Over Under Away Team, generally this is set at 1.91
            overunderodds = '1.91'
            underid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/ul/li[1]/div[' + str(
                i + 1) + "]/div[2]/span/div/div[3]/div/div[1]/div').click()"
            overid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/ul/li[1]/div[' + str(
                i + 1) + "]/div[2]/span/div/div[3]/div/div[2]/div').click()"
            # TotalScore
            score = driver.find_element_by_xpath('//*[@id="page"]/div[2]/ul/li[1]/div[' + str(
                i + 1) + ']/div[2]/span/div/div[3]/div/div[1]/div/span[1]/span[2]').text
            # HomeTeam
            hometeam = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/ul/li[1]/div[' + str(i + 1) + ']/div[2]/span/a/ul/li[2]').text
            # AwayTeam
            awayteam = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/ul/li[1]/div[' + str(i + 1) + ']/div[2]/span/a/ul/li[1]').text
            # Away Team Money Line
            awayline = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/ul/li[1]/div[' + str(
                    i + 1) + ']/div[2]/span/div/div[2]/div/div[1]/div/span').text
            awaylineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/ul/li[1]/div[' + str(
                i + 1) + "]/div[2]//span/div/div[2]/div/div[1]/div').click()"
            # Home Team Money Line
            homeline = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/ul/li[1]/div[' + str(
                    i + 1) + ']/div[2]/span/div/div[2]/div/div[2]/div/span').text
            homelineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/ul/li[1]/div[' + str(
                i + 1) + "]/div[2]//span/div/div[2]/div/div[2]/div').click()"
            betgamesodds.insert(i, ([overunderodds, score, hometeam, awayteam, homeline, awayline]))
            betgameclick.insert(i, ([overid, underid, homelineid, awaylineid]))
        except:
            pass

# DataFrame of Betgameodds
bets = pd.DataFrame(betgamesodds)

# Rename Columns
bets = bets.rename(
    columns={0: 'Over-Under Odds', 1: 'Over-Under Scores', 2: 'Home Team', 3: 'Away Team', 4: 'Home Odds',
             5: 'Away Odds'})
bets = bets.reset_index().drop('index', axis=1)

bets['Date'] = datetime.datetime.today().strftime('%Y-%m-%d')

savebets = pd.read_csv('SaveBets.csv')

savebets = savebets[
    ['Over-Under Odds', 'Over-Under Scores', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date', 'Difference']]

# This should track daily activity
if max(savebets['Date']) < datetime.datetime.today().strftime('%Y-%m-%d'):
    savebets = savebets.append(bets)
else:
    pass

savebets = savebets[
    ['Over-Under Odds', 'Over-Under Scores', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date', 'Difference']]

savebets['Home Team'] = savebets['Home Team'].str.replace(" ", "_").str.upper()
savebets['Away Team'] = savebets['Away Team'].str.replace(" ", "_").str.upper()

simapping = pd.DataFrame(data={'Team': ['TORONTO_RAPTORS',
                                        'BOSTON_CELTICS', 'LA_CLIPPERS', 'DENVER_NUGGETS', 'MIAMI_HEAT',
                                        'GOLDEN_STATE_WARRIORS', 'PHOENIX_SUNS', 'CHARLOTTE_HORNETS',
                                        'DALLAS_MAVERICKS', 'CHICAGO_BULLS', 'OKLAHOMA_CITY', 'HOUSTON_ROCKETS',
                                        'CLEVELAND_CAVALIERS', 'UTAH_JAZZ', 'BROOKLYN_NETS', 'DETROIT_PISTONS',
                                        'MINNESOTA_TIMBERWOLVES', 'NEW_ORLEANS_PELICANS', 'NEW_YORK_KNICKS',
                                        'WASHINGTON_WIZARDS', 'ORLANDO_MAGIC', 'PHILADELPHIA_76ERS', 'SACRAMENTO_KINGS',
                                        'INDIANA_PACERS', 'MILWAUKEE_BUCKS', 'DENVER_NUGGETS', 'ATLANTA_HAWKS',
                                        'MEMPHIS_GRIZZLIES', 'PORTLAND_TRAIL_BLAZERS', 'LA_LAKERS',
                                        'SAN_ANTONIO_SPURS'],
                               'ABBREVIATION': ['TOR', 'BOS', 'LAC', 'DEN', 'MIA', 'GSW', 'PHX', 'CHA', 'DAL', 'CHI',
                                                'OKC', 'HOU', 'CLE', 'UTA', 'BKN', 'DET', 'MIN', 'NOP', 'NYK', 'WAS',
                                                'ORL', 'PHI', 'SAC', 'IND', 'MIL', 'DEN', 'ATL', 'MEM', 'POR', 'LAL',
                                                'SAS']})

savebets = pd.merge(savebets, simapping, left_on=savebets['Home Team'], right_on=simapping['Team'], how='left')

savebets['ID'] = savebets['Date'] + str("H") + savebets['ABBREVIATION']

schedulesave = schedule.filter(['HomeScheduleID', 'TotalScore'])

savebets = savebets.drop('key_0', axis=1)

savebets = pd.merge(savebets, schedulesave, left_on=savebets['ID'], right_on=schedulesave['HomeScheduleID'], how='left')

savebets = savebets[['Over-Under Odds', 'Over-Under Scores',
                     'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date',
                     'Difference', 'Team', 'ABBREVIATION', 'ID', 'HomeScheduleID',
                     'TotalScore']]

savebets.to_csv('SaveBets.csv')

# Cosmetics
bets['Home Team'] = bets['Home Team'].str.replace(" ", "_").str.upper()
bets['Away Team'] = bets['Away Team'].str.replace(" ", "_").str.upper()

bets = pd.merge(bets, simapping, left_on=bets['Home Team'], right_on=simapping['Team'], how='left').drop('key_0',
                                                                                                         axis=1).drop(
    'Team', axis=1)
bets = pd.merge(bets, simapping, left_on=bets['Away Team'], right_on=simapping['Team'], how='left').drop('key_0',
                                                                                                         axis=1).drop(
    'Team', axis=1)
bets = bets.drop_duplicates().reset_index()
bets = bets.rename(columns={'ABBREVIATION_x': 'Home Abbreviation', 'ABBREVIATION_y': 'Away Abbreviation'})

# BetClick DataFrame with the command to the button in question to click and a few conversions
betclick = pd.DataFrame(betgameclick).rename(
    columns={0: 'Over Click', 1: 'Under Click', 2: 'Home Click', 3: 'Away Click', 4: 'Cut'})
bets.loc[bets['Home Odds'] == 'Closed', 'Home Odds'] = 1
bets['Home Odds'] = bets['Home Odds'].astype(float)
bets.loc[bets['Away Odds'] == 'Closed', 'Away Odds'] = 1
bets['Away Odds'] = bets['Away Odds'].astype(float)
bets['Over-Under Odds'] = bets['Over-Under Odds'].astype(float)

# Bet Differences of Odds, used to sort which games to bet on first
# Rationale is the closer the odds the more even the teams and the higher probability of a realized upset.
bets['Difference'] = abs(bets['Home Odds'] - bets['Away Odds'])
betclick['Difference'] = abs(bets['Home Odds'] - bets['Away Odds'])

# Cosmetics
betclick['Home Abbreviation'] = bets['Home Abbreviation']

# Sorts differences highest to lowest
bets.sort_values('Difference', inplace=True, ascending=True)
betclick.sort_values('Difference', inplace=True, ascending=True)
bets.reset_index().drop('index', axis=1)
betclick.reset_index().drop('index', axis=1)

# Save cutpoints for model testing later
cutpoints = list((set(bets['Over-Under Scores'])))

# Betting Strategy 1
favorite = []
underdog = []
favoriteclick = []
underdogclick = []
teamadd = []

# Create dataframe for the odds and a list for the buttons to click
for j in range(0, len(bets)):
    a = min(bets.iloc[j, :][['Away Odds', 'Home Odds']])
    favorite.insert(j, a)
    teamadd.insert(j, betclick.loc[j, 'Home Abbreviation'])
    if a == bets.iloc[j, :]['Away Odds']:
        favoriteclick.insert(j, betclick.loc[j, 'Away Click'])
    else:
        favoriteclick.insert(j, betclick.loc[j, 'Home Click'])

    a = max(bets.iloc[j, :][['Away Odds', 'Home Odds']])
    underdog.insert(j, a)
    if a == bets.iloc[j, :]['Away Odds']:
        underdogclick.insert(j, betclick.loc[j, 'Away Click'])
    else:
        underdogclick.insert(j, betclick.loc[j, 'Home Click'])

# Items to Click combined dataframe
click = [favoriteclick, underdogclick, teamadd]

# Set Bet amount and the command for website, will start with $1 per position
betamount = 1
bet = """driver.find_element_by_xpath('//*[@id="betcard-container"]/div/div/div/div[2]/div[1]/div/form/div/div/div[2]/div[1]/input').send_keys(betamount)"""
placebet = """driver.find_element_by_xpath('//*[@id="betcard-container"]/div/div/div/div[2]/div[1]/div/form/div/div/div[2]/div[1]/input').send_keys(betamount)"""
# loadmodelfromfile
modelsonly = []
with open("/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/models.pckl", "rb") as f:
    while True:
        try:
            modelsonly.append(pickle.load(f))
        except EOFError:
            break

models = [(modelsonly, item) for modelsonly, item in enumerate(modelsonly, start=210)]

completetesting = pd.DataFrame()
predicted = []

for j in range(0, len(bets)):

    testing = pd.DataFrame()

    home = bets['Home Abbreviation'][j]
    opponent = bets['Away Abbreviation'][j]

    teamstanding1 = standing[(standing['Index'] == home) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    teamstanding2 = standing[(standing['Index'] == opponent) & (standing['Season Year'] == currentyear)][
        'Team Rank'].reset_index()
    testing = pd.concat([testing, teamstanding2], axis=1).drop('index', axis=1)
    testing = testing.rename(columns={'Team Rank': 'Opponent Rank'})
    testing = pd.concat([testing, teamstanding1], axis=1).drop('index', axis=1)
    a = datetime.datetime.today()
    date_format = "%Y-%m-%d"
    b = datetime.datetime.strptime(max(nbadata[nbadata['TEAM_ABBREVIATION'] == home]['Gamedate']), date_format)
    c = datetime.datetime.strptime(max(nbadata[nbadata['TEAM_ABBREVIATION'] == opponent]['Gamedate']), date_format)
    delta = a - b
    delta2 = a - c
    testing['HomeDaysRest'] = min(delta.days, 5)
    testing['AwayDaysRest'] = min(delta2.days, 5)

    if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':

        testing['ConferenceBinary'] = 1
    else:
        testing['ConferenceBinary'] = 0

    if conference[conference['Team'] == opponent]['Conference'].reset_index()['Conference'][0] == 'E':
        testing['OpponentConferenceBinary'] = 1
    else:
        testing['OpponentConferenceBinary'] = 0

    testing['S2015'] = 0
    testing['S2016'] = 0
    testing['S2017'] = 0
    testing['S2018'] = 0
    testing['S2019'] = 0
    testing['S2020'] = 0
    testing['S2021'] = 0
    testing['S2022'] = 1

    testteamscoring = teamscoring[
        (teamscoring['TEAM_ABBREVIATION'] == home) & ((teamscoring['Season Year'] == currentyear))].reset_index()
    testing['Average_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[len(testteamscoring) - 5:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == home) & (
            totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)[
        'TotalScore']
    testing['Average_Team_Total_Score'] = np.mean(testscoring.loc[len(testscoring) - 5:len(testscoring)])
    testteamscoring = teamscoring[(teamscoring['TEAM_ABBREVIATION'] == opponent) & (
        (teamscoring['Season Year'] == currentyear))].reset_index()
    testing['Average_Opp_Team_Score'] = np.mean(
        testteamscoring['Team_Score'].iloc[len(testteamscoring) - 5:len(testteamscoring)])
    testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == opponent) & (
            totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)[
        'TotalScore']
    testing['Average_Opp_Total_Score'] = np.mean(testscoring.loc[len(testscoring) - 5:len(testscoring)])

    testing['Team_Last_Game_Score'] = \
        schedule[(schedule['HOME_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
            ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
            columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
            schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == home) & (schedule['Season Year'] == currentyear)].filter(
                ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
                columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
            ['start_time'], ascending=False)['team_score'].iloc[0]
    testing['Opponent_Last_Game_Score'] = \
        schedule[(schedule['HOME_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
            ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
            columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(schedule[
            (schedule['AWAY_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
            ['start_time'], ascending=False)['team_score'].iloc[0]
    testinghome = avgHomeStats[avgHomeStats['Team'] == home][
        ['AvgHPF', 'AvgHFGM', 'AvgHBLK', 'AvgHFG3M', 'AvgHOREB', 'AvgHAST', 'AvgHREB', 'AvgHSTL', 'AvgHAST_PCT',
         'AvgHAST_TO', 'AvgHAST_RATIO', 'AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT', 'AvgHTM_TOV_PCT', 'AvgHEFG_PCT',
         'AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE']].tail(1).reset_index()
    testingaway = avgAwayStats[avgAwayStats['Team'] == opponent][
        ['AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgAAST_PCT',
         'AvgAAST_TO', 'AvgAAST_RATIO', 'AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT', 'AvgATM_TOV_PCT', 'AvgAEFG_PCT',
         'AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE']].tail(1).reset_index()
    testing = pd.concat([testing, testingaway], axis=1).drop('index', axis=1)
    testing = pd.concat([testing, testinghome], axis=1).drop('index', axis=1)
    testing['Playoffs'] = 0
    completetesting = completetesting.append(testing)
    completetesing = completetesting[
        ['HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016',
         'S2018',
         'S2019', 'S2020', 'S2021', 'S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score',
         'Team_Last_Game_Score',
         'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
         'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
         'AvgHBLK', 'AvgHFG3M', 'AvgHOREB', 'AvgHAST', 'AvgHREB', 'AvgHSTL', 'Playoffs', 'AvgHAST_PCT', 'AvgHAST_TO',
         'AvgHAST_RATIO', 'AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
         'AvgHTM_TOV_PCT', 'AvgHEFG_PCT', 'AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS',
         'AvgHPIE', 'AvgAAST_PCT', 'AvgAAST_TO', 'AvgAAST_RATIO', 'AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
         'AvgATM_TOV_PCT', 'AvgAEFG_PCT', 'AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS',
         'AvgAPIE']]

    benchmark = [item[0] for item in models]
    if float(math.floor(float(list(bets[bets['Home Abbreviation'] == home]['Over-Under Scores'])[0]))) in benchmark:
        modelindex = benchmark.index(float(math.floor(float(
            bets[bets['Home Abbreviation'] == home]['Over-Under Scores'].reset_index().drop('index', axis=1)[
                'Over-Under Scores'][0]))))  # need to put the abbreviation.
        prob = models[modelindex][1].predict_proba(completetesting)[0][1]
        predicted.insert(j, [home, prob])
    else:
        predicted.insert(j, [home, "skip"])

predicted = pd.DataFrame(predicted).rename(columns={0: 'Team', 1: 'Prob'})
predicted.sort_values('Prob', inplace=True, ascending=False)

betamount = 1
bet = """driver.find_element_by_xpath('//*[@id="betcard-container"]/div/div/div/div[2]/div[1]/div/form/div/div/div[2]/div[1]/input').send_keys(betamount)"""
placebet = """driver.find_element_by_xpath('//*[@id="betcard-container"]/div/div/div/div[2]/div[2]/div/button').click()"""
balance = driver.find_element_by_xpath('//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
j = 0
numgames = len(bets)
gamesbet = []
betcount = 1
maxbets = 24
betcounter = 0
# possiblebets = math.comb(numgames, 2)

while betcounter <= maxbets:
    if numgames == 1:
        if predicted['Prob'][0] >= 0.5:
            exec(betclick['Over Click'][0])
            time.sleep(random.uniform(2, 5))
            exec(click[1][0])
            time.sleep(random.uniform(2, 5))
            exec(bet)
            time.sleep(random.uniform(2, 5))
            for i in range(0, 5):
                try:
                    exec(placebet)  # accept
                    time.sleep(random.uniform(1, 3))
                    newbalance = driver.find_element_by_xpath(
                        '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                except:
                    pass
        else:
            exec(betclick['Under Click'][0])
            time.sleep(random.uniform(2, 5))
            # Underdog-Favorite
            exec(click[1][0])
            time.sleep(random.uniform(2, 5))
            exec(bet)
            time.sleep(random.uniform(2, 5))
            for i in range(0, 5):
                try:
                    exec(placebet)  # accept # accept
                    time.sleep(random.uniform(1, 3))
                    newbalance = driver.find_element_by_xpath(
                        '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                except:
                    pass
        if predicted['Prob'][0] >= 0.5:
            exec(betclick['Over Click'][0])
            exec(click[0][0])
            time.sleep(random.uniform(2, 5))
            exec(bet)
            time.sleep(random.uniform(2, 5))
            for i in range(0, 5):
                try:
                    exec(placebet)  # accept # accept
                    time.sleep(random.uniform(1, 3))
                    newbalance = driver.find_element_by_xpath(
                        '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                except:
                    pass
        else:
            exec(betclick['Under Click'][0])
            time.sleep(random.uniform(2, 5))
            exec(click[0][0])
            time.sleep(random.uniform(2, 5))
            exec(bet)
            time.sleep(random.uniform(2, 5))
            for i in range(0, 5):
                try:
                    exec(placebet)  # accept  # accept
                    time.sleep(random.uniform(1, 3))
                    newbalance = driver.find_element_by_xpath(
                        '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                except:
                    pass
    else:
        if (numgames == 2):
            if predicted['Prob'][0] >= 0.5:
                # Underdog-Favorite
                exec(click[1][j])
                for i in range(0, numgames):
                    if i != j:
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][i])
                    else:
                        pass
            time.sleep(random.uniform(2, 5))
            # if model says bet over bet over
            if predicted['Prob'][0] >= 0.5:
                exec(betclick['Over Click'][0])
                time.sleep(random.uniform(2, 5))
                # if model says bet over bet over
                exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                exec(placebet)  # accept  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_b5_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            # Underdogs
            for k in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[1][k])
            exec(betclick['Under Click'][0])
            time.sleep(random.uniform(2, 5))
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                exec(placebet)  # accept  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

                # Favorites
            for i in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[0][i])
            exec(betclick['Under Click'][0])
            time.sleep(random.uniform(2, 5))
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                exec(placebet)  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

            # Favorite - Underdog
            exec(click[0][j])
            for i in range(0, numgames):
                if i != j:
                    time.sleep(random.uniform(2, 5))
                    exec(click[1][i])
                else:
                    pass
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                exec(placebet)  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            j = j + 1

        else:
            for a in (predicted.index):
                betcount = 0

                if len(gamesbet) < 4:
                    gamesbet = gamesbet
                else:
                    gamesbet = []

                gamesbet = []
                gamesindex = []
                while betcount <= 1:
                    team = predicted.loc[a]['Team']
                    teamposition = click[2].index(team)
                    if predicted['Prob'][a] >= 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                    if predicted['Prob'][a] < 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))  # Underdog-Favorite
                    if (predicted['Prob'][a] < 0.50) | (predicted['Prob'][a] >= 0.50):
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        gamesindex = list(range(0, numgames))
                        gamesindex = [item for item in gamesindex if item not in gamesbet]
                        secondgame = list(bets[bets['Home Abbreviation'] != team].index)
                        for l in secondgame:
                            secondgameteam = bets.loc[random.choice(list(bets.index))]['Home Abbreviation']
                            if (team != secondgameteam) & (l not in gamesbet) & (l != a):
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                time.sleep(random.uniform(2, 5))
                                exec(click[0][l])
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                gamesbet.insert(0, l)
                                break
                            else:
                                revisedsecondgame = list(bets[bets['Home Abbreviation'] != secondgameteam].index)
                                revisedsecondteam = bets.loc[random.choice(list(bets.index))]['Home Abbreviation']

                        # if model says bet over bet over
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                exec(placebet)  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(10, 15))
                    # Underdogs
                    if predicted['Prob'][a] >= 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] < 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(5, 7))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 7))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        #                                bets[bets['Home Abbreviation']==team]['']
                        for i in range(0, 5):
                            try:
                                exec(placebet)  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(10, 15))
                        # Favorites
                    if predicted['Prob'][a] >= 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] < 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    time.sleep(random.uniform(2, 5))
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][gamesbet[0]])
                        time.sleep(random.uniform(5, 6))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 6))
                        exec(click[0][teamposition])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(1, 3))
                        for i in range(0, 5):
                            try:
                                exec(placebet)  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(10, 15))

                    # Favorite - Underdog
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[0][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                exec(placebet)  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(10, 15))
                    betcount = betcount + 1
                    betcounter = betcounter + 1
    driver.close()


def matched(str):
    count = 0


# for i in str:
#     if i == "[":
#         count += 1
#     elif i == "]":
#         count -= 1
#     if count < 0:
#         return False
# return count == 0

# nbadatamodel = pd.read_csv(directory1+'nbadatamodel.csv',low_memory=False)

teamavgH = []
teamavgHdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'H') & (nbadata['MIN'] > 0)]
teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
for team in teamtemp:
    a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
    gamesplayed = len(a['Gamedate'].unique())
    a = a.groupby(['TEAM_ABBREVIATION']).agg(
        {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
         'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum'})  # 3PTs made may be important, FG as well
    a = a.div(gamesplayed)
    a = a.add_prefix('AvgH')
    a['Season'] = s
    a['Team'] = team
    teamavgHdf = teamavgHdf.append(a)

teamavgH = []
teamavgHdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'H') & (nbadata['MIN'] > 0)]
teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
for team in teamtemp:
    a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
    a = a.groupby(['TEAM_ABBREVIATION', 'Gamedate']).agg(
        {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
         'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum'})  # 3PTs made may be important, FG as well
    a = a.add_prefix('AvgH').reset_index()
    a['Season'] = s
    a['Team'] = team
    teamavgHdf = teamavgHdf.append(a)

teamavgHdf = teamavgHdf.reset_index()

avgHomeStats = []
avgHomeStats = pd.DataFrame()

for s in list(teamavgHdf['Season'].unique()):
    for i in list(teamavgHdf['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamavgHdf[
                (teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgHdf['Season'] == s))].reset_index()
        except:
            tempscoring = teamavgHdf[(teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgAHf['Season'] == s))].drop(
                'level_0', axis=1).reset_index()
        for j in range(6, len(tempscoring)):
            tempmean = tempscoring.loc[j - 3:j - 1]
            gamedate = tempscoring['Gamedate'][j]
            tempmean = tempmean.groupby("Team", as_index=True)[
                'AvgHPF', 'AvgHFGM', 'AvgHBLK', 'AvgHFG3M', 'AvgHOREB', 'AvgHAST', 'AvgHREB', 'AvgHSTL'].mean()
            tempmean = tempmean.reset_index()
            tempmean['Gamedate'] = gamedate
            tempmean['Season'] = s
            avgHomeStats = avgHomeStats.append(tempmean)

avgHomeStats = avgHomeStats.reset_index()

teamavgAdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'A') & (nbadata['MIN'] > 0)]
teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
for team in teamtemp:
    a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
    a = a.groupby(['TEAM_ABBREVIATION', 'Gamedate']).agg(
        {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
         'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum'})  # 3PTs made may be important, FG as well
    a = a.add_prefix('AvgA')
    a['Season'] = s
    a['Team'] = team
    teamavgAdf = teamavgAdf.append(a)

teamavgAdf = teamavgAdf.reset_index()

avgAwayStats = []
avgAwayStats = pd.DataFrame()

for s in list(teamavgAdf['Season'].unique()):
    for i in list(teamavgAdf['TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = teamavgAdf[
                (teamavgAdf['TEAM_ABBREVIATION'] == i) & ((teamavgAdf['Season'] == s))].reset_index()
        except:
            tempscoring = teamavgAdf[(teamavgAdf['TEAM_ABBREVIATION'] == i) & ((teamavgAdf['Season'] == s))].drop(
                'level_0', axis=1).reset_index()
        for j in range(6, len(tempscoring)):
            tempmean = tempscoring.loc[j - 3:j - 1]
            gamedate = tempscoring['Gamedate'][j]
            tempmean = tempmean.groupby("Team", as_index=True)[
                'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL'].mean()
            tempmean = tempmean.reset_index()
            tempmean['Gamedate'] = gamedate
            tempmean['Season'] = s
            avgAwayStats = avgAwayStats.append(tempmean)

avgAwayStats = avgAwayStats.reset_index()

avgAwayStats['avgID'] = avgAwayStats['Team'] + avgAwayStats['Gamedate']
avgHomeStats['avgID'] = avgHomeStats['Team'] + avgHomeStats['Gamedate']

avgAwayStats = avgAwayStats.drop(['Season', 'Gamedate'], axis=1)
avgHomeStats = avgHomeStats.drop(['Season', 'Gamedate'], axis=1)

#
teamavgA = []
teamavgAdf = pd.DataFrame()
for s in list(nbadata['Season Year'].unique()):
    tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'A') & (nbadata['MIN'] > 0)]
teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
for team in teamtemp:
    a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
    gamesplayed = len(a['Gamedate'].unique())
    a = a.groupby(['TEAM_ABBREVIATION']).agg(
        {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
         'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum',
         'FGM': 'sum'})  # 3PTs made may be important, FG as well}) #3PTs made may be important, FG as well
    a = a.div(gamesplayed)
    a = a.add_prefix('AvgA')
    a['Season'] = s
    a['Team'] = team
    teamavgAdf = teamavgAdf.append(a)

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
            tempscoring = teamscoring[
                (teamscoring['TEAM_ABBREVIATION'] == i) & (teamscoring['Season Year'] == s)].reset_index()
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
                (teamscoring['TEAM_ABBREVIATION'] == i) & ((teamscoring['Season Year'] == s))].drop('level_0',
                                                                                                    axis=1).reset_index()
        for j in range(6, len(tempscoring)):
            tempmean = tempscoring.loc[j - 3:j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['Team_Score'])
            avgrollteamscore.insert(0, [i, s, gamedate, tempmean])

avgrollteamscore = pd.DataFrame(avgrollteamscore).rename(
    columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

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
        for j in range(6, len(tempscoring)):
            tempmean = tempscoring.loc[j - 3:j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['TotalScore'])
            avgrolltotalscore.insert(0, [i, s, gamedate, tempmean])

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(
    columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

avgrolltotalscore['Month'] = avgrolltotalscore['Date'].str[5:7].apply(int)  # use for gamedate

avgrolltotalscore['avgID'] = avgrolltotalscore['TEAM_ABBREVIATION'] + avgrolltotalscore['Date']

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score', 'Month']).drop_duplicates()

# players = list(nbadatamodel['PLAYER_ID'].unique())
# opponents = list(nbadatamodel['TEAM_ABBREVIATION'].unique())

# nbadata2=nbadata[['PLAYER_ID','Opponent','H/A','MIN','PTS','Gamedate']]
# nbadata2=nbadata2.drop_duplicates()

# averageadjust=[]
# playeraverages=pd.DataFrame()
#
# for i in players:
#     for j in opponents:
#         try:
#             temp=nbadata[(nbadata2["Opponent"]==j)&(nbadata2["PLAYER_ID"]==i) & (nbadata2["H/A"]=="H") &(nbadata2['MIN']>0)]
#             H= np.mean(temp["PTS"])
#             temp=nbadata[(nbadata2["Opponent"]==j)&(nbadata2["PLAYER_ID"]==i) & (nbadata2["H/A"]=="A") &(nbadata2['MIN']>0)]
#             A=np.mean(temp["PTS"])
#             averageadjust.insert(players.index(i),[i,j,H,A])
#         except:
#             pass

# playeraverages=pd.DataFrame(averageadjust)
# playeraverages.to_csv('/Users/Quaye/Desktop/Work In Progress/PlayerAverages.csv')
# playeraverages = pd.read_excel('/Users/Quaye/Desktop/Work in Progress/PlayerAverages.csv',low_memory=False)
# playeraverages = playeraverages.rename(columns={0: 'PLAYER_ID', 1: 'Opponent', 2: 'HomeAvg', 3: 'AwayAvg'})
# playeraverages['ModelID'] = playeraverages['PLAYER_ID'].astype(str) + playeraverages['Opponent'].astype(str)
# nbadatamodel['ID'] = nbadatamodel['PLAYER_ID'].astype(str) + nbadatamodel['Opponent'].astype(str)
# playeraverages = playeraverages.drop('Opponent', axis=1).drop('PLAYER_ID', axis=1)

nbadatamodel = nbadata
nbadata = nbadata.drop('key_0', axis=1)

# nbadatamodel['HomeAvg'] = nbadatamodel['HomeAvg'].fillna(
#     np.mean(nbadata[(nbadata["H/A"] == "H") & (nbadata['MIN'] > 10)]['PTS'] / 2))
# nbadatamodel['AwayAvg'] = nbadatamodel['AwayAvg'].fillna(
#     np.mean(nbadata[(nbadata["H/A"] == "A") & (nbadata['MIN'] > 10)]['PTS']) / 2)

tfdata = pd.DataFrame()

nbadatamodel = nbadatamodel.rename(
    columns={'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'Gamedate_x': 'Gamedate', 'Season Year_x': 'Season Year'})

for team in nbadatamodel['TEAM_ABBREVIATION'].unique():
    traindatatemp = pd.DataFrame()
traindata = nbadatamodel[(nbadatamodel['TEAM_ABBREVIATION'] == team) & (nbadatamodel['H/A'] == 'H')]
for game in traindata['Gamedate'].unique():
    teamdata = traindata[(traindata['Gamedate'] == game)]
    totscore = teamdata['TotalScore'].iloc[0]
    Opponent = traindata[(traindata['Gamedate'] == game)]['Opponent'].unique()[0]
    Opponentdf = nbadatamodel[
        (nbadatamodel['TEAM_ABBREVIATION'] == Opponent) & (nbadatamodel['Gamedate'] == game)]
    players = pd.DataFrame()
    # for k in teamdata['PLAYER_ID'].unique():
    #     playerstemp = (teamdata[teamdata['PLAYER_ID'] == k])['HomeAvg'].reset_index().drop('index', axis=1)
    #     playerstemp = playerstemp.rename(
    #         columns={'HomeAvg': 'HomeAvg' + str(list(teamdata['PLAYER_ID'].unique()).index(k))})
    #     players = pd.concat([players, playerstemp], axis=1)
    # players = players.sort_values(by=0, ascending=False, axis=1)
    # for j in Opponentdf['PLAYER_ID'].unique():
    #     oplayerstemp = Opponentdf[Opponentdf['PLAYER_ID'] == j]['AwayAvg'].reset_index().drop('index', axis=1)
    #     oplayerstemp = oplayerstemp.rename(
    #         columns={'AwayAvg': 'AwayAvg' + str(list(Opponentdf['PLAYER_ID'].unique()).index(j))})
    #     players = pd.concat([players, oplayerstemp, ], axis=1)
    # playerstemp = players.values
    # playerstemp = np.sort(playerstemp)
    # playerstemp = playerstemp[:, ::-1]
    # players = pd.DataFrame(playerstemp, players.index,
    #                        columns=['HomeAvg0', 'HomeAvg1', 'HomeAvg2', 'HomeAvg3', 'HomeAvg4', 'AwayAvg0',
    #                                 'AwayAvg1', 'AwayAvg2', 'AwayAvg3', 'AwayAvg4'])
    players[['HomeDaysRest', 'Team Rank', 'ConferenceBinary', 'Opponent Rank', 'OpponentConferenceBinary',
             'Season Year']] = teamdata.reset_index()[
        ['HomeDaysRest', 'Team Rank', 'ConferenceBinary', 'Opponent Rank', 'OpponentConferenceBinary',
         'Season Year']]  # Add ,'AST','REB','TOV','STL', 'BLK','PF'
    players['AwayDaysRest'] = Opponentdf.reset_index()['HomeDaysRest'][0]
    players['Gamedate'] = game
    players['Team'] = team
    players['Opponent'] = Opponent
    players['TotalScore'] = totscore
    # if len(players.columns) < 18:
    #     pass
    # else:
    tfdata = tfdata.append(players)
# except:
#     pass

tfdata = tfdata.drop_duplicates()
tfdata = tfdata.reset_index()
tfdatakeep = tfdata

tfdata = tfdatakeep
tfdata = tfdata.drop('index', axis=1)

tfdata['avgIDmain'] = tfdata['Team'] + tfdata['Gamedate']

tfdata['avgID2main'] = tfdata['Opponent'] + tfdata['Gamedate']

teamscoring = teamscoring.reset_index()

teamscoring = teamscoring.filter(['avgID', 'Last_Game_Score']).drop_duplicates()

tfdata2 = pd.merge(tfdata, teamscoring, left_on=tfdata['avgIDmain'], right_on=teamscoring['avgID'], how='left').rename(
    columns={'avgID_x': 'avgID', 'Last_Game_Score': 'Team_Last_Game_Score'})

tfdata2 = tfdata2.drop('key_0', axis=1)

tfdata3 = pd.merge(tfdata2, teamscoring, left_on=tfdata2['avgID2main'], right_on=teamscoring['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Last_Game_Score': 'Opponent_Last_Game_Score'}).drop(
    'avgID_y', axis=1)

tfdata3 = tfdata3.drop_duplicates()

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score']).drop_duplicates()

tfdata3 = tfdata3.drop('key_0', axis=1)

tfdata4 = pd.merge(tfdata3, avgrollteamscore, left_on=tfdata3['avgIDmain'], right_on=avgrollteamscore['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Team_Score'}).dropna()

tfdata4 = tfdata4.drop('key_0', axis=1)

tfdata5 = pd.merge(tfdata4, avgrollteamscore, left_on=tfdata4['avgID2main'], right_on=avgrollteamscore['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Team_Score'}).dropna()

tfdata5 = tfdata5.drop('key_0', axis=1)

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score']).drop_duplicates()

tfdata6 = pd.merge(tfdata5, avgrolltotalscore, left_on=tfdata5['avgIDmain'], right_on=avgrolltotalscore['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Team_Total_Score'}).dropna()

tfdata6 = tfdata6.drop('key_0', axis=1)

tfdata7 = pd.merge(tfdata6, avgrolltotalscore, left_on=tfdata6['avgID2main'], right_on=avgrolltotalscore['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

tfdata7 = tfdata7.drop('key_0', axis=1)

tfdata8 = pd.merge(tfdata7, avgHomeStats, left_on=tfdata7['avgIDmain'], right_on=avgHomeStats['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

tfdata8 = tfdata8.drop('key_0', axis=1)

tfdata9 = pd.merge(tfdata8, avgAwayStats, left_on=tfdata8['avgID2main'], right_on=avgAwayStats['avgID'],
                   how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

traindata = tfdata9
testdata = traindata[traindata['Gamedate'] > '2021-02-28']
traindata = traindata[(traindata['Gamedate'] <= '2021-02-28')]
TrainScore = traindata['TotalScore']
TestScore = testdata['TotalScore']

traindata = traindata.drop('TotalScore', axis=1)
testdata = testdata.drop('TotalScore', axis=1)

# TotalScore=tfdata7['TotalScore']
# tfdata=tfdata7.drop('TotalScore',axis=1)
# TotalScore.loc[TotalScore.TotalScore<200,'TotalScore']=0
# TotalScore.loc[(TotalScore.TotalScore>200)&(TotalScore.TotalScore<225),'TotalScore']=1
# TotalScore.loc[(TotalScore.TotalScore>225)&(TotalScore.TotalScore<245),'TotalScore']=2
# TotalScore.loc[(TotalScore.TotalScore>245),'TotalScore']=3

traindata['S2015'] = (np.where(traindata['Season Year'] == '2015', 1, 0))
traindata['S2016'] = (np.where(traindata['Season Year'] == '2016', 1, 0))
traindata['S2017'] = (np.where(traindata['Season Year'] == '2017', 1, 0))
traindata['S2018'] = (np.where(traindata['Season Year'] == '2018', 1, 0))
traindata['S2019'] = (np.where(traindata['Season Year'] == '2019', 1, 0))
traindata['S2020'] = (np.where(traindata['Season Year'] == '2020', 1, 0))
traindata['S2021'] = (np.where(traindata['Season Year'] == '2021', 1, 0))
traindata = traindata.drop('Season Year', axis=1)

testdata['S2015'] = (np.where(testdata['Season Year'] == '2015', 1, 0))
testdata['S2016'] = (np.where(testdata['Season Year'] == '2016', 1, 0))
testdata['S2017'] = (np.where(testdata['Season Year'] == '2017', 1, 0))
testdata['S2018'] = (np.where(testdata['Season Year'] == '2018', 1, 0))
testdata['S2019'] = (np.where(testdata['Season Year'] == '2019', 1, 0))
testdata['S2020'] = (np.where(testdata['Season Year'] == '2020', 1, 0))
testdata['S2021'] = (np.where(testdata['Season Year'] == '2021', 1, 0))
testdata = testdata.drop('Season Year', axis=1)

tfdata = traindata[[
    'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
    'S2019', 'S2020', 'S2021', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
    'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
    'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
    'AvgHBLK', 'AvgHFG3M', 'AvgHOREB',
    'AvgHAST', 'AvgHREB', 'AvgHSTL', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
    'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS',
    'PIE']]  # Season Year can remove due to taking average of scores.

tftest = testdata[[
    'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
    'S2019', 'S2020', 'S2021', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
    'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
    'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
    'AvgHBLK', 'AvgHFG3M', 'AvgHOREB',
    'AvgHAST', 'AvgHREB', 'AvgHSTL', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
    'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS',
    'PIE']]  # Season Year can remove due to taking average of scores.]]  # Season Year can remove due to taking average of scores.

models = []
scores = []

cutpoints = list(range(200, 255))
cutpoints = [math.floor(float(x)) for x in cutpoints]

# cutpoints = [226,227,231,220]

for cut in cutpoints:
Target = np.where(TrainScore > float(cut), 1, 0)
TestTarget = np.where(TestScore > float(cut), 1, 0)
# X_train, X_test, y_train, y_test = train_test_split(tfdata, Target, test_size=0.3, random_state=1)
X_train = tfdata
y_train = Target
estimators = 4000
param_test = {'max_depth': (1, 5, 10, 15), 'subsample': (0.5, 0.6, 0.75, 0.8)}
gsearch = GridSearchCV(
    estimator=GradientBoostingClassifier(learning_rate=0.01, n_estimators=estimators, max_features='sqrt',
                                         subsample=0.8, random_state=10, validation_fraction=0.3, n_iter_no_change=5,
                                         tol=0.01),  # sqrt
    param_grid=param_test, scoring='accuracy', n_jobs=4, iid=False, cv=5)
# est = GradientBoostingClassifier(n_estimators=estimators,max_features='sqrt', learning_rate=0.001,random_state=0).fit(X_train,y_train)
fitmodel = gsearch.fit(X_train, y_train)

score = fitmodel.score(tftest, TestTarget)
models.insert(cutpoints.index(cut), [cut, gsearch.best_estimator_])
scores.insert(cutpoints.index(cut), [score, cut])
#    scores = [item for item in scores if item[0] >= 0.65]

fitmodel.best_estimator_.predict(tftest)
fitmodel.best_estimator_.predict(X_train)
score = fitmodel.score(X_train, y_train)
sum((fitmodel.predict_proba(tftest)[:, 1] >= 0.05).astype(bool))
list(fitmodel.predict_proba(tftest))
sum(fitmodel.predict(tftest))
TestTarget

# 'AST':'sum','OREB':'sum', 'DREB':'sum', 'REB':'sum','TOV':'sum','STL':'sum', 'BLK':'sum','PF':'sum'

# Access to Website and then information pull of daily games
driver = webdriver.Chrome(
    '/Users/kabariquaye/Desktop/chromedriver')  # need to have the right version of chromedriver installed on computer.
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
# The gamedate to check if the games are today
gamedatesi = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/h4').text
gamedatesi = gamedatesi + ' ' + str(datetime.datetime.today().year)
gamedatesi = datetime.datetime.strptime(gamedatesi, '%a %b %d %Y')
today = datetime.datetime.today().date()
today = datetime.datetime(today.year, today.month, today.day)
betgamesodds = []
betgameclick = []

# Actual Site Log-in
time.sleep(random.uniform(10, 20))
driver.find_element_by_xpath('//*[@id="header-container"]/span/div/div/div/span').click()
driver.find_element_by_xpath('//*[@id="LoginForm__account-name"]').send_keys('kabariq@gmail.com')
driver.find_element_by_xpath('//*[@id="LoginForm__password"]').send_keys('############')
driver.find_element_by_xpath(
    '//*[@id="portals-container"]/div/div[1]/div/div[2]/div/content/div/div/div/form/button').click()
time.sleep(random.uniform(10, 20))
try:
    driver.find_element_by_xpath('//*[@id="portals-container"]/div/div[1]/div[1]/div[2]/span').click()
time.sleep(random.uniform(5, 10))
except:
pass
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
time.sleep(random.uniform(2, 5))
balance = driver.find_element_by_xpath('//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
balances = []
balances.insert(0, balance)
newbalance = balance
account = []
winslosses = []
account.insert(len(account), [balance, datetime.datetime.today().strftime('%Y-%m-%d')])
bot = pd.DataFrame(account).rename(columns={0: 'Account_Balance', 1: 'Date'})  # balance over time
botfile = pd.read_excel('BalanceOverTime.csv')

botfile = botfile[['Account_Balance', 'Date']]

# This should track daily activity
if max(botfile['Date']) < datetime.datetime.today().strftime('%Y-%m-%d'):
    botfile = botfile.append(bot)
else:
    pass

botfile = botfile[['Account_Balance', 'Date']]

botfile.to_csv('BalanceOverTime.csv')

betgamesodds = []
betgameclick = []

if gamedatesi == today:

for i in range(0, 20):
    try:
        driver.maximize_window()
        driver.implicitly_wait(random.uniform(6, 9))
        # Over Under Away Team, generally this is set at 1.91
        overunderodds = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + ']/div[2]/a/div/div[3]/div[2]/div/span[2]').text
        underid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + "]/div[2]/a/div/div[3]/div[2]/div').click()"
        overid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + "]/div[2]/a/div/div[3]/div[1]/div').click()"
        # TotalScore
        score = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + ']/div[2]/a/div/div[3]/div[1]/div/span[1]/span[2]/span').text
        # HomeTeam
        hometeam = driver.find_element_by_xpath(
            '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/a/ul/li[2]').text
        # AwayTeam
        awayteam = driver.find_element_by_xpath(
            '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/a/ul/li[1]').text
        # Away Team Money Line
        awayline = driver.find_element_by_xpath(
            '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/a/div/div[2]/div[1]/div/span').text
        awaylineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + "]/div[2]/a/div/div[2]/div[1]/div').click()"
        # Home Team Money Line
        homeline = driver.find_element_by_xpath(
            '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/a/div/div[2]/div[2]/div/span').text
        homelineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
            i + 2) + "]/div[2]/a/div/div[2]/div[2]/div').click()"
        betgamesodds.insert(i, ([overunderodds, score, hometeam, awayteam, homeline, awayline]))
        betgameclick.insert(i, ([overid, underid, homelineid, awaylineid]))
    except:
        try:
            driver.maximize_window()
            driver.implicitly_wait(random.uniform(6, 9))
            # Over Under Away Team, generally this is set at 1.91
            overunderodds = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/div/ul/li[1]/div[1]/div[2]/span/div/div[2]/div/div[1]/div').text
            driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/div/ul/li[1]/div[2]/div[2]/span/div/div[3]/div[1]/div/span[2]').text
            underid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + "]/div[2]/span/div/div[3]/div[2]/div').click()"
            overid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + "]/div[2]/span/div/div[3]/div[1]/div').click()"
            # TotalScore
            score = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + ']/div[2]/span/div/div[3]/div[1]/div/span[1]/span[2]/span').text
            # HomeTeam
            hometeam = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/span/a/ul/li[2]').text
            # AwayTeam
            awayteam = driver.find_element_by_xpath(
                '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 2) + ']/div[2]/span/a/ul/li[1]').text
            # Away Team Money Line
            awayline = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + ']/div[2]/span/div/div[2]/div[1]/div/span').text
            awaylineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + "]/div[2]/span/div/div[2]/div[1]/div').click()"
            # Home Team Money Line
            homeline = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + ']/div[2]/span/div/div[2]/div[2]/div/span').text
            homelineid = "driver.find_element_by_xpath(" + "'//*[@id=" + '"page"]/div[2]/div/ul/li[1]/div[' + str(
                i + 2) + "]/div[2]/span/div/div[2]/div[2]/div').click()"
            betgamesodds.insert(i, ([overunderodds, score, hometeam, awayteam, homeline, awayline]))
            betgameclick.insert(i, ([overid, underid, homelineid, awaylineid, score]))
        except:
            try:
                driver.maximize_window()
                driver.implicitly_wait(random.uniform(6, 9))
                # Over Under Away Team, generally this is set at 1.91
                overunderodds = '1.91'
                overid = """driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li/div[""" + str(
                    i + 1) + """]/div[2]/span/div/div[3]/div/div[1]/div').click()"""
                underid = """driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li/div[""" + str(
                    i + 1) + """]/div[2]/span/div/div[3]/div/div[2]/div').click()"""
                # TotalScore
                score = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li/div[' + str(
                    i + 1) + ']/div[2]/span/div/div[3]/div/div[1]/div/span/span[2]').text
                # HomeTeam
                hometeam = driver.find_element_by_xpath(
                    '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 1) + ']/div[2]/span/a/ul/li[2]/p').text
                # AwayTeam
                awayteam = driver.find_element_by_xpath(
                    '//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(i + 1) + ']/div[2]/span/a/ul/li[1]/p').text
                # Away Team Money Line
                awayline = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
                    i + 1) + ']/div[2]/span/div/div[2]/div/div[1]/div').text
                awaylineid = """driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li/div[""" + str(
                    i + 1) + """]/div[2]/span/div/div[2]/div/div[2]/div').click()"""
                # Home Team Money Line
                homeline = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/div[' + str(
                    i + 1) + ']/div[2]/span/div/div[2]/div/div[2]/div').text
                homelineid = """driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li/div[""" + str(
                    i + 1) + """]/div[2]/span/div/div[2]/div/div[1]/div').click()"""
                betgamesodds.insert(i, ([overunderodds, score, hometeam, awayteam, homeline, awayline]))
                betgameclick.insert(i, ([overid, underid, homelineid, awaylineid, score]))
            except:
                pass
            pass
        pass

# DataFrame of Betgameodds
bets = pd.DataFrame(betgamesodds)

# Rename Columns
bets = bets.rename(
    columns={0: 'Over-Under Odds', 1: 'Over-Under Scores', 2: 'Home Team', 3: 'Away Team', 4: 'Home Odds',
             5: 'Away Odds'})
bets = bets.reset_index().drop('index', axis=1)

if sum(bets['Over-Under Scores'] < 150) > 0:
    client = Client('AC468cf84e2201587130c09aece683217d', 'e6af6bf9426b51ff9cfb1eb17ddb1a5f')
message = client.messages.create(body="OPPORTUNITY BETTING RIGHT NOW", from_='+14154944379', to='+13065916701')

bets['Date'] = datetime.datetime.today().strftime('%Y-%m-%d')

savebets = pd.read_csv('SaveBets.csv')

savebets = savebets[
    ['Over-Under Odds', 'Over-Under Scores', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date', 'Difference']]

# This should track daily activity
if max(savebets['Date']) < datetime.datetime.today().strftime('%Y-%m-%d'):
    savebets = savebets.append(bets)
else:
    pass

savebets = savebets[
    ['Over-Under Odds', 'Over-Under Scores', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date', 'Difference']]

savebets['Home Team'] = savebets['Home Team'].str.replace(" ", "_").str.upper()
savebets['Away Team'] = savebets['Away Team'].str.replace(" ", "_").str.upper()

simapping = pd.DataFrame(data={'Team': ['TORONTO_RAPTORS',
                                        'BOSTON_CELTICS', 'LA_CLIPPERS', 'DENVER_NUGGETS', 'MIAMI_HEAT',
                                        'GOLDEN_STATE_WARRIORS', 'PHOENIX_SUNS', 'CHARLOTTE_HORNETS',
                                        'DALLAS_MAVERICKS', 'CHICAGO_BULLS', 'OKLAHOMA_CITY', 'HOUSTON_ROCKETS',
                                        'CLEVELAND_CAVALIERS', 'UTAH_JAZZ', 'BROOKLYN_NETS', 'DETROIT_PISTONS',
                                        'MINNESOTA_TIMBERWOLVES', 'NEW_ORLEANS_PELICANS', 'NEW_YORK_KNICKS',
                                        'WASHINGTON_WIZARDS', 'ORLANDO_MAGIC', 'PHILADELPHIA_76ERS', 'SACRAMENTO_KINGS',
                                        'INDIANA_PACERS', 'MILWAUKEE_BUCKS', 'DENVER_NUGGETS', 'ATLANTA_HAWKS',
                                        'MEMPHIS_GRIZZLIES', 'PORTLAND_TRAIL_BLAZERS', 'LA_LAKERS',
                                        'SAN_ANTONIO_SPURS'],
                               'ABBREVIATION': ['TOR', 'BOS', 'LAC', 'DEN', 'MIA', 'GSW', 'PHX', 'CHA', 'DAL', 'CHI',
                                                'OKC', 'HOU', 'CLE', 'UTA', 'BKN', 'DET', 'MIN', 'NOP', 'NYK', 'WAS',
                                                'ORL', 'PHI', 'SAC', 'IND', 'MIL', 'DEN', 'ATL', 'MEM', 'POR', 'LAL',
                                                'SAS']})

savebets = pd.merge(savebets, simapping, left_on=savebets['Home Team'], right_on=simapping['Team'], how='left')

savebets['ID'] = savebets['Date'] + str("H") + savebets['ABBREVIATION']

schedulesave = schedule.filter(['HomeScheduleID', 'TotalScore'])

savebets = savebets.drop('key_0', axis=1)

savebets = pd.merge(savebets, schedulesave, left_on=savebets['ID'], right_on=schedulesave['HomeScheduleID'], how='left')

savebets = savebets[['Over-Under Odds', 'Over-Under Scores',
                     'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Date',
                     'Difference', 'Team', 'ABBREVIATION', 'ID', 'HomeScheduleID',
                     'TotalScore']]

savebets = savebets.to_csv('SaveBets.csv')

# Cosmetics
bets['Home Team'] = bets['Home Team'].str.replace(" ", "_").str.upper()
bets['Away Team'] = bets['Away Team'].str.replace(" ", "_").str.upper()

bets = pd.merge(bets, simapping, left_on=bets['Home Team'], right_on=simapping['Team'], how='left').drop('key_0',
                                                                                                         axis=1).drop(
    'Team', axis=1)
bets = pd.merge(bets, simapping, left_on=bets['Away Team'], right_on=simapping['Team'], how='left').drop('key_0',
                                                                                                         axis=1).drop(
    'Team', axis=1)
bets = bets.drop_duplicates().reset_index()
bets = bets.rename(columns={'ABBREVIATION_x': 'Home Abbreviation', 'ABBREVIATION_y': 'Away Abbreviation'})

# BetClick DataFrame with the command to the button in question to click and a few conversions
betclick = pd.DataFrame(betgameclick).rename(
    columns={0: 'Over Click', 1: 'Under Click', 2: 'Home Click', 3: 'Away Click', 4: 'Cut'})
bets['Home Odds'] = bets['Home Odds'].astype(float)
bets['Away Odds'] = bets['Away Odds'].astype(float)
bets['Over-Under Odds'] = bets['Over-Under Odds'].astype(float)

# Bet Differences of Odds, used to sort which games to bet on first
# Rationale is the closer the odds the more even the teams and the higher probability of a realized upset.
bets['Difference'] = abs(bets['Home Odds'] - bets['Away Odds'])
betclick['Difference'] = abs(bets['Home Odds'] - bets['Away Odds'])

# Cosmetics
betclick['Home Abbreviation'] = bets['Home Abbreviation']

# Sorts differences highest to lowest
bets.sort_values('Difference', inplace=True, ascending=True)
betclick.sort_values('Difference', inplace=True, ascending=True)
bets.reset_index().drop('index', axis=1)
betclick.reset_index().drop('index', axis=1)

# Save cutpoints for model testing later
cutpoints = list((set(bets['Over-Under Scores'])))

# Betting Strategy 1
favorite = []
underdog = []
favoriteclick = []
underdogclick = []
teamadd = []

# Create dataframe for the odds and a list for the buttons to click
for j in range(0, len(bets)):
    a = min(bets.iloc[j, :][['Away Odds', 'Home Odds']])
favorite.insert(j, a)
teamadd.insert(j, betclick.loc[j, 'Home Abbreviation'])
if a == bets.iloc[j, :]['Away Odds']:
    favoriteclick.insert(j, betclick.loc[j, 'Away Click'])
else:
    favoriteclick.insert(j, betclick.loc[j, 'Home Click'])

a = max(bets.iloc[j, :][['Away Odds', 'Home Odds']])
underdog.insert(j, a)
if a == bets.iloc[j, :]['Away Odds']:
    underdogclick.insert(j, betclick.loc[j, 'Away Click'])
else:
    underdogclick.insert(j, betclick.loc[j, 'Home Click'])

# Items to Click combined dataframe
click = [favoriteclick, underdogclick, teamadd]

# Set Bet amount and the command for website, will start with $1 per position
betamount = 1
bet = """driver.find_element_by_xpath('//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[1]/div/form/div/div/div/div[1]/input').send_keys(betamount)"""

completetesting = pd.DataFrame()

for j in range(0, len(bets)):

testing = pd.DataFrame()

home = bets['Home Abbreviation'][j]
opponent = bets['Away Abbreviation'][j]
a = datetime.datetime.today()
date_format = "%Y-%m-%d"
b = datetime.datetime.strptime(max(nbadata[nbadata['TEAM_ABBREVIATION'] == home]['Gamedate']), date_format)
c = datetime.datetime.strptime(max(nbadata[nbadata['TEAM_ABBREVIATION'] == opponent]['Gamedate']), date_format)
delta = a - b
delta2 = a - c
testing['HomeDaysRest'] = min(delta.days, 5)
testing['AwayDaysRest'] = min(delta2.days, 5)

if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':

    testing['ConferenceBinary'] = 1
else:
    testing['ConferenceBinary'] = 0

if conference[conference['Team'] == home]['Conference'].reset_index()['Conference'][0] == 'E':
    testing['OpponentConferenceBinary'] = 1
else:
    testing['OpponentConferenceBinary'] = 0

teamstanding1 = standing[(standing['Index'] == home) & (standing['Season Year'] == currentyear)][
    'Team Rank'].reset_index()
teamstanding2 = standing[(standing['Index'] == opponent) & (standing['Season Year'] == currentyear)][
    'Team Rank'].reset_index()
testing = pd.concat([testing, teamstanding1], axis=1).drop('index', axis=1)
testing = pd.concat([testing, teamstanding2], axis=1).drop('index', axis=1)

testing['S2015'] = 0
testing['S2016'] = 0
testing['S2017'] = 0
testing['S2018'] = 0
testing['S2019'] = 0
testing['S2020'] = 0
testing['S2021'] = 1

testteamscoring = teamscoring[
    (teamscoring['TEAM_ABBREVIATION'] == home) & ((teamscoring['Season Year'] == currentyear))].reset_index()
testing['Average_Team_Score'] = np.mean(
    testteamscoring['Team_Score'].iloc[len(testteamscoring) - 5:len(testteamscoring)])
testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == home) & (
        totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)[
    'TotalScore']
testing['Average_Team_Total_Score'] = np.mean(testscoring.loc[len(testscoring) - 5:len(testscoring)])
testteamscoring = teamscoring[
    (teamscoring['TEAM_ABBREVIATION'] == opponent) & ((teamscoring['Season Year'] == currentyear))].reset_index()
testing['Average_Opp_Team_Score'] = np.mean(
    testteamscoring['TotalScore'].iloc[len(testteamscoring) - 5:len(testteamscoring)])
testscoring = totalscoring[(totalscoring['TEAM_ABBREVIATION'] == opponent) & (
        totalscoring['Season Year'] == currentyear)].reset_index().drop_duplicates().drop('index', axis=1)
testing['Average_Opp_Total_Score'] = np.mean(testscoring.loc[len(testscoring) - 5:len(testscoring)])

testing['Team_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == team) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
        schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == team) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].iloc[0]
testing['Opponent_Last_Game_Score'] = \
    schedule[(schedule['HOME_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
        ['HOME_TEAM_ABBREVIATION', 'home_team_score', 'start_time']).rename(
        columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'home_team_score': 'team_score'}).append(
        schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == opponent) & (schedule['Season Year'] == currentyear)].filter(
            ['AWAY_TEAM_ABBREVIATION', 'away_team_score', 'start_time']).rename(
            columns={'AWAY_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'away_team_score': 'team_score'})).sort_values(
        ['start_time'], ascending=False)['team_score'].iloc[0]
completetesting = completetesting.append(testing)

# loadmodelfromfile
modelsonly = []
with open("/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/models.pckl", "rb") as f:
    while True:
        try:
            modelsonly.append(pickle.load(f))
        except EOFError:
            break

models2 = models

models = [(modelsonly, item) for modelsonly, item in enumerate(modelsonly, start=210)]

benchmark = [item[0] for item in models]
modelindex = benchmark.index(float(math.floor(float(
    bets[bets['Home Abbreviation'] == home]['Over-Under Scores'].reset_index().drop('index', axis=1)[
        'Over-Under Scores'][0]))))  # need to put the abbreviation.
prob = models[modelindex][1].predict_proba(completetesting.iloc[:, 0:17])[0][1]
predicted.insert(j, [home, prob])

predicted = pd.DataFrame(predicted).rename(columns={0: 'Team', 1: 'Prob'})
predicted.sort_values('Prob', inplace=True, ascending=False)
# predicted=predicted[predicted['Prob']<=0.50]
# Betting Selection Loop:Therefore this will go through the games for the day making up to 15 bets or at the max $60 worth of bets based on
# combinations of two games at a time.

# sign-in
# straight to betting
# Access to Website and then information pull of daily games

# for range in (1,len(prairieaccounts)):
driver = webdriver.Chrome(
    '/Users/kabariquaye/Desktop/chromedriver')  # need to have the right version of chromedriver installed on computer.
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
# The gamedate to check if the games are today
gamedatesi = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/h4').text
gamedatesi = gamedatesi + ' ' + str(datetime.datetime.today().year)
gamedatesi = datetime.datetime.strptime(gamedatesi, '%a %b %d %Y')
today = datetime.datetime.today().date()
today = datetime.datetime(today.year, today.month, today.day)
betgamesodds = []
betgameclick = []

# Actual Site Log-in
time.sleep(random.uniform(10, 20))
driver.find_element_by_xpath('//*[@id="header-container"]/span/div/div/div/span').click()
driver.find_element_by_xpath('//*[@id="LoginForm__account-name"]').send_keys(prairieaccounts[i][])
driver.find_element_by_xpath('//*[@id="LoginForm__password"]').send_keys(prairieaccounts[i][1])
driver.find_element_by_xpath(
    '//*[@id="portals-container"]/div/div[1]/div/div[2]/div/content/div/div/div/form/button').click()
time.sleep(random.uniform(10, 20))

j = 0
numgames = len(bets)
betcount = 1
maxbets = 24
bets = 0
possiblebets = comb(numgames, 2)

while bets <= maxbets:
    if numgames == 1:
        # if model says bet over bet over
        if predicted['Prob'][0] >= 0.7:
            exec(betclick['Over Click'][0])
            time.sleep(random.uniform(2, 5))
            # Underdog-Favorite
            exec(click[1][0])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
        else:
            pass
        if predicted['Prob'][0] >= 0.7:
            exec(betclick['Over Click'][0])
            # Bet Favorite
            exec(click[0][0])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

            else:
                pass
    else:
        if (numgames == 2):
            if predicted['Prob'][0] >= 0.6:
                # Underdog-Favorite
                exec(click[1][j])
                for i in range(0, numgames):
                    if i != j:
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][i])
                    else:
                        pass
            time.sleep(random.uniform(2, 5))
            # if model says bet over bet over
            if predicted['Prob'][0] >= 0.6:
                exec(betclick['Over Click'][0])
                time.sleep(random.uniform(2, 5))
                # if model says bet over bet over
                exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            # Underdogs
            for k in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[1][k])
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

                # Favorites
            for i in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[0][i])
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

            # Favorite - Underdog
            exec(click[0][j])
            for i in range(0, numgames):
                if i != j:
                    time.sleep(random.uniform(2, 5))
                    exec(click[1][i])
                else:
                    pass
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            j = j + 1

        else:
            for a in (predicted.index):
                betcount = 0
                try:
                    if len(gamesbet) < 2:
                        gamesbet = gamesbet
                    else:
                        gamesbet = []
                except:
                    gamesbet = []
                gamesindex = []
                while betcount <= 1:
                    team = predicted.loc[a]['Team']
                    teamposition = click[2].index(team)
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))  # Underdog-Favorite
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        gamesindex = list(range(0, numgames))
                        gamesindex = [item for item in gamesindex if item not in gamesbet]
                        gameschoose = list(bets[bets['Home Abbreviation'] != team].index)
                        for i in gameschoose:
                            if (team != bets.loc[random.choice(list(bets.index))]['Home Abbreviation']) & (
                                    i not in gamesbet) & (i != a):
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                time.sleep(random.uniform(2, 5))
                                exec(click[0][i])
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                gamesbet.insert(0, i)
                                break
                            else:
                                pass
                        # if model says bet over bet over
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                    # Underdogs
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(5, 7))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 7))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        #                                bets[bets['Home Abbreviation']==team]['']
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                        # Favorites
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    time.sleep(random.uniform(2, 5))
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][gamesbet[0]])
                        time.sleep(random.uniform(5, 6))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 6))
                        exec(click[0][teamposition])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(1, 3))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))

                    # Favorite - Underdog
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[0][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                    betcount = betcount + 1
                    bets = bets + 1
driver.close()

for range in (1, len(prairieaccounts)):
    driver = webdriver.Chrome(
        '/Users/kabariquaye/Desktop/chromedriver')  # need to have the right version of chromedriver installed on computer.
driver.get("https://www.sportsinteraction.com/basketball/nba-betting-lines/")
# The gamedate to check if the games are today
gamedatesi = driver.find_element_by_xpath('//*[@id="page"]/div[2]/div/ul/li[1]/h4').text
gamedatesi = gamedatesi + ' ' + str(datetime.datetime.today().year)
gamedatesi = datetime.datetime.strptime(gamedatesi, '%a %b %d %Y')
today = datetime.datetime.today().date()
today = datetime.datetime(today.year, today.month, today.day)
betgamesodds = []
betgameclick = []

# Actual Site Log-in
time.sleep(random.uniform(10, 20))
driver.find_element_by_xpath('//*[@id="header-container"]/span/div/div/div/span').click()
driver.find_element_by_xpath('//*[@id="LoginForm__account-name"]').send_keys(prairieaccounts[i][0])
driver.find_element_by_xpath('//*[@id="LoginForm__password"]').send_keys(prairieaccounts[i][1])
driver.find_element_by_xpath(
    '//*[@id="portals-container"]/div/div[1]/div/div[2]/div/content/div/div/div/form/button').click()
time.sleep(random.uniform(10, 20))
j = 0
numgames = len(bets)
betcount = 1
maxbets = 24
bets = 0
possiblebets = comb(numgames, 2)

while bets <= maxbets:
    if numgames == 1:
        # if model says bet over bet over
        if predicted['Prob'][0] >= 0.7:
            exec(betclick['Over Click'][0])
            time.sleep(random.uniform(2, 5))
            # Underdog-Favorite
            exec(click[1][0])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
        else:
            pass
        if predicted['Prob'][0] >= 0.7:
            exec(betclick['Over Click'][0])
            # Bet Favorite
            exec(click[0][0])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

            else:
                pass
    else:
        if (numgames == 2):
            if predicted['Prob'][0] >= 0.6:
                # Underdog-Favorite
                exec(click[1][j])
                for i in range(0, numgames):
                    if i != j:
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][i])
                    else:
                        pass
            time.sleep(random.uniform(2, 5))
            # if model says bet over bet over
            if predicted['Prob'][0] >= 0.6:
                exec(betclick['Over Click'][0])
                time.sleep(random.uniform(2, 5))
                # if model says bet over bet over
                exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            # Underdogs
            for k in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[1][k])
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

                # Favorites
            for i in range(0, numgames):
                time.sleep(random.uniform(2, 5))
                exec(click[0][i])
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass

            # Favorite - Underdog
            exec(click[0][j])
            for i in range(0, numgames):
                if i != j:
                    time.sleep(random.uniform(2, 5))
                    exec(click[1][i])
                else:
                    pass
            exec(betclick['Under Click'][0])
            exec(betclick['Under Click'][1])
            exec(bet)
            time.sleep(random.uniform(2, 5))
            try:
                driver.find_element_by_xpath(
                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                time.sleep(random.uniform(2, 5))
                try:
                    driver.find_element_by_xpath(
                        '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button/span').click()  # placebet
                except:
                    pass
            except:
                pass
            j = j + 1

        else:
            for a in (predicted.index):
                betcount = 0
                try:
                    if len(gamesbet) < 2:
                        gamesbet = gamesbet
                    else:
                        gamesbet = []
                except:
                    gamesbet = []
                gamesindex = []
                while betcount <= 1:
                    team = predicted.loc[a]['Team']
                    teamposition = click[2].index(team)
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))  # Underdog-Favorite
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        gamesindex = list(range(0, numgames))
                        gamesindex = [item for item in gamesindex if item not in gamesbet]
                        gamechoose = list(bets[bets['Home Abbreviation'] != team].index)

                        for i in gameschoose:
                            if (team != bets.loc[random.choice(list(bets.index))]['Home Abbreviation']) & (
                                    i not in gamesbet) & (i != a):
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                time.sleep(random.uniform(2, 5))
                                exec(click[0][i])
                                time.sleep(random.uniform(2, 5))
                                driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                                gamesbet.insert(0, i)
                                break
                            else:
                                pass
                        # if model says bet over bet over
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                    # Underdogs
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[1][teamposition])
                        time.sleep(random.uniform(5, 7))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 7))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        #                                bets[bets['Home Abbreviation']==team]['']
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                        # Favorites
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    time.sleep(random.uniform(2, 5))
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[0][gamesbet[0]])
                        time.sleep(random.uniform(5, 6))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(5, 6))
                        exec(click[0][teamposition])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(bet)
                        time.sleep(random.uniform(1, 3))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))

                    # Favorite - Underdog
                    if predicted['Prob'][a] > 0.50:
                        exec(betclick['Over Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if predicted['Prob'][a] <= 0.50:
                        exec(betclick['Under Click'][betclick[betclick['Home Abbreviation'] == team].index[0]])
                    if (predicted['Prob'][a] <= 0.50) | (predicted['Prob'][a] > 0.50):
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        exec(click[1][gamesbet[0]])
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(click[0][teamposition])
                        time.sleep(random.uniform(2, 5))
                        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
                        exec(bet)
                        time.sleep(random.uniform(2, 5))
                        time.sleep(random.uniform(2, 5))
                        for i in range(0, 5):
                            try:
                                driver.find_element_by_xpath(
                                    '//*[@id="content-right"]/div/div[2]/div[2]/div[2]/div[2]/div/button').click()  # accept
                                time.sleep(random.uniform(1, 3))
                                newbalance = driver.find_element_by_xpath(
                                    '//*[@id="header-container"]/span/div/div/div/div/div[1]/div/span/h4').text
                            except:
                                pass
                        balance = newbalance
                        time.sleep(random.uniform(5, 10))
                    betcount = betcount + 1
                    bets = bets + 1
    driver.close()
    time.sleep(random.uniform(10, 20))

schedule2 = schedule

schedule2 = pd.merge(schedule2, conference, left_on='HOME_TEAM_ABBREVIATION', right_on='Team', how='left').drop('Team',
                                                                                                                axis=1).rename(
    columns={'Conference': 'Conference Home'})
schedule2 = pd.merge(schedule2, conference, left_on='AWAY_TEAM_ABBREVIATION', right_on='Team', how='left').drop('Team',
                                                                                                                axis=1).rename(
    columns={'Conference': 'Conference Away'})
tempanalysis = schedule2.groupby(['Season Year', 'Month', 'Conference Home']).agg({'TotalScore': 'mean'})

tempanalysisH = tempanalysis.reset_index(level=['Season Year', 'Month', 'Conference Home'])

tempanalysis[tempanalysisH['Season Year'] == '2021']

tempanalysis = schedule2.groupby(['Season Year', 'Month', 'Conference Away']).agg({'TotalScore': 'mean'})

tempanalysis[tempanalysis['Season Year'] == '2021']

conference = pd.DataFrame(data={'Team': ['BOS', 'GSW', 'HOU', 'CLE', 'ORL', 'UTA', 'BKN', 'DET', 'MIN',
                                         'NOP', 'PHI', 'WAS', 'MEM', 'SAC', 'CHA', 'POR', 'SAS', 'IND',
                                         'ATL', 'DAL', 'PHX', 'MIA', 'MIL', 'DEN', 'OKC', 'LAL', 'TOR',
                                         'NYK', 'LAC', 'CHI'],
                                'Conference': ['E', 'W', 'W', 'E', 'E', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W',
                                               'W', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'W',
                                               'E', 'E', 'W', 'E']})
