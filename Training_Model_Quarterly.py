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


#Save all Files here - this is where files will be updated
#Save your directory files here and then run code.
directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'

# Starts importing model data
nbadata = pd.read_csv(directory1+'quarterdata.csv', low_memory=False)

nbadata = nbadata[['TEAM_ID', 'TEAM_NAME','GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK',
                   'CFID', 'CFPARAMS', 'Gamedate','Quarter']]

nbadata['Gamedate'] = nbadata['Gamedate'].astype(str)
nbadata['TEAM_NAME'] = nbadata['TEAM_NAME'].str.replace(" ", "_").str.upper()
teammapping = pd.read_csv(directory1+'dataframe.csv')
nbadata = pd.merge(nbadata, teammapping, left_on=nbadata['TEAM_NAME'], right_on=teammapping['Team_Names'],how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'TEAM_ABBREVIATION'})

# We want to set ranks for the teams based on wins and get the end result for each game.
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
schedulelist = [Schedule2022,Schedule2021, Schedule2020, Schedule2019, Schedule2018, Schedule2017, Schedule2016, Schedule2015,
                Schedule2014, Schedule2013, Schedule2012, Schedule2011, Schedule2010]
Schedule2020.filter(['start_time', 'home_team_score'])

schedule = pd.DataFrame()

# Append each schedule and do appropriate conversions
schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014).append(Schedule2013).append(Schedule2012).append(Schedule2011).append(Schedule2010)


#Adjust Date so it is correct, adjust
schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]

#This is the Full Team Names to Abbreviation mapping
teammapping = pd.read_csv(directory1+'dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})

#Unique NBA Teams
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())

#Seasons
seasonyears = ['2022','2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']


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


#We don't want to re-add old data so we filter on dates that are larger than the maximum date in our data file
captureddates = pd.DataFrame(nbadata['Gamedate'].unique())
datelist = [x for x in dates[0].tolist() if x not in captureddates[0].tolist()]
datelist = [x for x in datelist if x > max(captureddates[0])]

for period in range(0,4):
    # The following for loop reads the new data from nba.com into a list
    for e in range(0, len(datelist)):
        try:
            year1 = datelist[e][:4]
            month1 = datelist[e][5:7]
            day1 = datelist[e][8:10]
            a = 'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
            if datelist[e] in dataytd[0].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[1].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[2].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[3].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[4].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[5].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[6].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[7].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[8].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            c = '&DateTo='
            d = '%2F'
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36','x-nba-stats-origin':'stats','x-nba-stats-token':'true','Referer':'https://www.nba.com/','Host': 'stats.nba.com', 'Origin': 'https://www.nba.com'}
            url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
            r = requests.get(url, headers=headers)
            numrecords = len(r.json()['resultSets'][0]['rowSet'])
            fields = r.json()['resultSets'][0]['headers']
            data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
            for i in range(0, numrecords):
                records = r.json()['resultSets'][0]['rowSet'][i]
                for j in range(0, len(records)):
                    data.iloc[[i], [j]] = records[j]
            nbadatalist.insert(e, [data, datelist[e],period+1])
        except:
            pass

nbadataworking = pd.DataFrame()
nbadatatemp = pd.DataFrame()

# Read the data from the website into a dataframe
for i in range(0, len(nbadatalist)):
    nbadatatemp = pd.DataFrame(nbadatalist[i][0])
    nbadatatemp['Gamedate'] = nbadatalist[i][1]
    nbadatatemp['Quarter'] = nbadatalist[i][2]
    nbadataworking = nbadataworking.append(nbadatatemp)

nbadataworking['Playoffs'] = 0

if not (nbadataworking.empty):
    nbadataworking = nbadataworking[['TEAM_ID', 'TEAM_NAME','GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK',
                   'CFID', 'CFPARAMS', 'Gamedate','Playoffs','Quarter']]

if (nbadataworking.empty):
    nbadatatworking = pd.DataFrame()

#Same process but this is for Playoffs Data
for period in range(0,4):
    # The following for loop reads the new data from nba.com into a list
    for e in range(0, len(datelist)):
        try:
            year1 = datelist[e][:4]
            month1 = datelist[e][5:7]
            day1 = datelist[e][8:10]
            a = 'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
            if datelist[e] in dataytd[0].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[1].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[2].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[3].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[4].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[5].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[6].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[7].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            elif datelist[e] in dataytd[8].tolist():
                b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
            c = '&DateTo='
            d = '%2F'
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36','x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/','Host': 'stats.nba.com', 'Origin': 'https://www.nba.com'}
            url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
            r = requests.get(url, headers=headers)
            numrecords = len(r.json()['resultSets'][0]['rowSet'])
            fields = r.json()['resultSets'][0]['headers']
            data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
            for i in range(0, numrecords):
                records = r.json()['resultSets'][0]['rowSet'][i]
                for j in range(0, len(records)):
                    data.iloc[[i], [j]] = records[j]
            playoffslist.insert(e, [data, datelist[e],period+1])
        except:
            pass

playoffsworking = pd.DataFrame()
playoffstemp = pd.DataFrame()

# Read the data from the website into a dataframe
for i in range(0, len(playoffslist)):
    playoffstemp = pd.DataFrame(playoffslist[i][0])
    playoffstemp['Gamedate'] = playoffslist[i][1]
    playoffstemp['Quarter'] = playoffslist[i][1]
    playoffsworking = playoffsworking.append(playoffstemp)

playoffsworking['Playoffs'] = 1

if not (playoffsworking.empty):
    playoffsdata = playoffsworking[['TEAM_ID', 'TEAM_NAME','GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK',
                   'CFID', 'CFPARAMS', 'Gamedate','Playoffs','Quarter']]
if (playoffsworking.empty):
    playoffsdata = pd.DataFrame()

playoffsdata = playoffsdata.append(playoffsworking)
playoffsdata = playoffsdata.reset_index()

#Append data into final dataframe
nbadata = nbadata.append(nbadataworking)
nbadataworking = nbadataworking.append(playoffsdata)

#SAVE DATA
nbadata.to_csv(directory1+'nbadataytd_quarterly.csv')

#READ DATA
nbadata = pd.read_csv(directory1+'nbadataytd_quarterly.csv', low_memory=False)
nbadata = nbadata[['TEAM_ID', 'TEAM_NAME','GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK',
                   'CFID', 'CFPARAMS', 'Gamedate','Playoffs','Quarter']]

#This secondary files adds in extra statistics for each game
execfile('/Users/kabariquaye/PycharmProjects/pythonProject1/AdditionalData_Quarterly.py')


#passing['ID']=passing['Gamedate'].astype(str) + passing['TEAM_ID'].astype(str).str[:-2]
#touches['ID']=touches['Gamedate'].astype(str) + touches['TEAM_ID'].astype(str).str[:-2]


#The above file generates the extra stats and then we read it in.
nbadataaddon = pd.read_csv(directory1+'nbadataytdaddon_quarterly.csv',low_memory=False)

#Create IDs to merge nbadata with the additional statistics
nbadata['TEAM_ID'] = nbadata['TEAM_ID'].astype(str)
nbadataaddon['ID'] = nbadataaddon['Gamedate'].astype(str) + nbadataaddon['TEAM_ID'].astype(str).str[:-2]
#Additional statistics to be merged
nbadataaddon = nbadataaddon[['ID', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT','E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']]

#Another ID
nbadata['AddonID'] = nbadata['Gamedate'] + nbadata['TEAM_ID'].astype(str).str[:-2]
#Merge
nbadata = pd.merge(nbadata, nbadataaddon, left_on=nbadata['AddonID'], right_on=nbadataaddon['ID'], how='left')
#nbadata = pd.merge(nbadata, passing, left_on=nbadata['AddonID'], right_on=passing['ID'], how='left')
#nbadata = pd.merge(nbadata, touches, left_on=nbadata['AddonID'], right_on=touches['ID'], how='left')

#Drop any error rows and duplicates
nbadata = nbadata.drop_duplicates()
nbadata = nbadata.dropna()

#Reset index and drop IDs
nbadata.reset_index().drop(['index', 'key_0'], axis=1)
nbadata = nbadata.drop(['AddonID', 'ID'], axis=1)

#Keep Rows we want
nbadata = nbadata[['TEAM_ID', 'TEAM_NAME','GP',
                   'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                   'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
                   'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                   'GP_RANK', 'W_RANK', 'L_RANK',
                   'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
                   'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
                   'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                   'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK',
                   'PTS_RANK', 'PLUS_MINUS_RANK',
                   'CFID', 'CFPARAMS', 'Gamedate','Playoffs','Quarter']]

#I think these commands are redundant
nbadata = nbadata.infer_objects()
nbadata['Gamedate'] = nbadata['Gamedate'].astype(str)
nbadata['TEAM_ABBREVIATION'] = nbadata['TEAM_ABBREVIATION'].astype(str)

#We need to re merge all our seasonal information so we redo the season process
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


schedulelist = [Schedule2022,Schedule2021, Schedule2020, Schedule2019, Schedule2018, Schedule2017, Schedule2016, Schedule2015,Schedule2014]

schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)

schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
teammapping = pd.read_csv(directory1+'dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())

seasonyears = ['2022','2021', '2020', '2019', '2018', '2017', '2016', '2015']

seasonmerge = schedule.filter(['Season Year', 'start_time'])
seasonmerge = seasonmerge.drop_duplicates()
nbadata = pd.merge(nbadata, seasonmerge, left_on=nbadata['Gamedate'], right_on=seasonmerge['start_time'],
                   how='left').drop('start_time', axis=1)

# The following for loop develops the standings for each team and assigns H or A for home or away games.
from pandas.core.common import \
    SettingWithCopyWarning  # I need this to not get a billion warnings while completing the loop.

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

for s in seasonyears:
    scheduletemp = schedule[(schedule['Season Year'] == s) & (schedule['start_time'] < (
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
# This will likely need to be included later, some form of ranking so we can incorporate starter information into the modelling.
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
opponentrank = pd.merge(rankingtemp, opponentrank, left_on=rankingtemp['SeasonID'], right_on=opponentrank['SeasonID'],
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
nbadata = pd.merge(nbadata, conference, left_on='TEAM_ABBREVIATION', right_on='Team', how='left').drop('Team', axis=1)
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

#In between seasons this will give extra large numbers and in-between February break
nbadata['HomeDaysRest'] = nbadata['HomeDaysRest'].clip(upper=5)
nbadata['away_team_score'][np.isnan(nbadata['away_team_score'])] = 0
nbadata['home_team_score'][np.isnan(nbadata['home_team_score'])] = 0
nbadata = nbadata.drop('key_0', axis=1)


#We set the number of games for averaging to use in our model, 5 is the ideal number
# averaging=5
#
# #We first filter variable and take sums for games played @ home
# teamavgH = []
# teamavgHdf = pd.DataFrame()
# for s in list(nbadata['Season Year'].unique()):
#     tempseason = nbadata[(nbadata['Season Year'] == s) & (nbadata['H/A'] == 'H') & (nbadata['MIN'] > 0)]
#     teamtemp = list(tempseason['TEAM_ABBREVIATION'].unique())
#     for team in teamtemp:
#         a = tempseason[(tempseason['TEAM_ABBREVIATION'] == team)]
#         a = a.groupby(['TEAM_ABBREVIATION', 'Gamedate']).agg(
#             {'PTS': 'sum', 'AST': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'TOV': 'sum', 'STL': 'sum',
#              'BLK': 'sum', 'PF': 'sum', 'FG3M': 'sum', 'FGM': 'sum', 'AST_PCT': 'sum', 'AST_TO': 'sum',
#              'AST_RATIO': 'sum', 'OREB_PCT': 'sum', 'DREB_PCT': 'sum', 'REB_PCT': 'sum',
#              'TM_TOV_PCT': 'sum', 'EFG_PCT': 'sum', 'TS_PCT': 'sum', 'E_PACE': 'sum', 'PACE': 'sum',
#              'PACE_PER40': 'sum', 'POSS': 'sum', 'PIE': 'sum'})  # 3PTs made may be important, FG as well
#         a = a.add_prefix('AvgH').reset_index()
#         a['Season'] = s
#         a['Team'] = team
#         teamavgHdf = teamavgHdf.append(a)

# teamavgHdf = teamavgHdf.reset_index()

#Take Average for a given season and a given team for games at home
# avgHomeStats = pd.DataFrame()
# for s in list(teamavgHdf['Season'].unique()):
#     for i in list(teamavgHdf['TEAM_ABBREVIATION'].unique()):
#         try:
#             tempscoring = teamavgHdf[
#                 (teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgHdf['Season'] == s))].reset_index()
#         except:
#             tempscoring = teamavgHdf[(teamavgHdf['TEAM_ABBREVIATION'] == i) & ((teamavgHdf['Season'] == s))].drop(
#                 'level_0', axis=1).reset_index()
#         for j in range(averaging+1, len(tempscoring)):
#             tempmean = tempscoring.loc[j - (averaging+1):j - 1]
#             gamedate = tempscoring['Gamedate'][j]
#             tempmean = tempmean.groupby("Team", as_index=True)[
#                 'AvgHPF', 'AvgHFGM', 'AvgHBLK', 'AvgHFG3M', 'AvgHOREB', 'AvgHAST', 'AvgHREB', 'AvgHSTL', 'AvgHAST_PCT', 'AvgHAST_TO', 'AvgHAST_RATIO', 'AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
#                 'AvgHTM_TOV_PCT', 'AvgHEFG_PCT', 'AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE'].mean()
#             tempmean = tempmean.reset_index()
#             tempmean['Gamedate'] = gamedate
#             tempmean['Season'] = s
#             avgHomeStats = avgHomeStats.append(tempmean)
# avgHomeStats = avgHomeStats.reset_index()


#Do the same thing for away games
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
        for j in range(averaging+1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging+1):j - 1]
            gamedate = tempscoring['Gamedate'][j]
            tempmean = tempmean.groupby("Team", as_index=True)[
                'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgAAST_PCT', 'AvgAAST_TO', 'AvgAAST_RATIO', 'AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
                'AvgATM_TOV_PCT', 'AvgAEFG_PCT', 'AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE'].mean()
            tempmean = tempmean.reset_index()
            tempmean['Gamedate'] = gamedate
            tempmean['Season'] = s
            avgAwayStats = avgAwayStats.append(tempmean)
avgAwayStats = avgAwayStats.reset_index()


#This is going to be used for merging to our final training data file.
avgAwayStats['avgID'] = avgAwayStats['Team'] + avgAwayStats['Gamedate']
avgHomeStats['avgID'] = avgHomeStats['Team'] + avgHomeStats['Gamedate']

avgAwayStats = avgAwayStats.drop(['Season', 'Gamedate'], axis=1)
avgHomeStats = avgHomeStats.drop(['Season', 'Gamedate'], axis=1)


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
        for j in range(averaging+1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging+1):j - 1]
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
        for j in range(averaging+1, len(tempscoring)):
            tempmean = tempscoring.loc[j - (averaging+1):j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmean['TotalScore'])
            avgrolltotalscore.insert(0, [i, s, gamedate, tempmean])

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

avgrolltotalscore['Month'] = avgrolltotalscore['Date'].str[5:7].apply(int)  # use for gamedate

avgrolltotalscore['avgID'] = avgrolltotalscore['TEAM_ABBREVIATION'] + avgrolltotalscore['Date']

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score', 'Month']).drop_duplicates()

avgrolltotalscore = pd.DataFrame(avgrolltotalscore).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg Score'}).drop_duplicates()

execfile('/Users/kabariquaye/PycharmProjects/pythonProject1/Quarterly_Data.py')
quarterdata = pd.read_csv(directory1+'quarterdata.csv')
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
quarterdata = pd.merge(quarterdata, teammapping, left_on=quarterdata['TEAM_NAME'], right_on=teammapping['Team_Names'],
                       how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'TEAM_ABBREVIATION'})
qw = quarterdata
qw = qw.dropna()
qw['ID'] = qw['TEAM_ABBREVIATION'] + qw['Gamedate'].astype(str)
IDs = list(qw['ID'].unique())
firstquartersummary = []
halfsummary = []
threequartersummary = []
fourthquartersummary = []

for ID in IDs:
    qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 1)]
    qwtemp2 = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 2)]
    qwtemp3 = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 3)]
    qwtemp4 = qw[(qw['ID'] == ID) & (qw['Quarter'] ==4)]
    qwtemp(['ID']).agg({'PTS': 'sum','FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum',
    'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'AST': 'sum', 'TOV': 'sum', 'STL': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'PF': 'sum',
    'PFA': 'sum', 'PFD': 'sum'})
    a = qwtemp(['ID']).agg({'PTS': 'sum','FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum',
    'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'AST': 'sum', 'TOV': 'sum', 'STL': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'PF': 'sum',
    'PFA': 'sum', 'PFD': 'sum'})
    b = (qwtemp['Gamedate']).unique()
    c = qwtemp2(['ID']).agg({'PTS': 'sum','FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum',
    'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'AST': 'sum', 'TOV': 'sum', 'STL': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'PF': 'sum',
    'PFA': 'sum', 'PFD': 'sum'})
    d = qwtemp3(['ID']).agg({'PTS': 'sum','FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum',
    'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'AST': 'sum', 'TOV': 'sum', 'STL': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'PF': 'sum',
    'PFA': 'sum', 'PFD': 'sum'})
    e = qwtemp4(['ID']).agg({'PTS': 'sum','FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum',
    'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum', 'AST': 'sum', 'TOV': 'sum', 'STL': 'sum', 'BLK': 'sum', 'BLKA': 'sum', 'PF': 'sum',
    'PFA': 'sum', 'PFD': 'sum'})
    quartersummary.insert(IDs.index(ID), [ID,a,b,c,d,e])

# for ID in IDs:
#     qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 2)]
#     a = sum(qwtemp['PTS'])
#     b = (qwtemp['Gamedate']).unique()
#     halfsummary.insert(IDs.index(ID), [ID, a, b])
#
# for ID in IDs:
#     qwtemp = qw[(qw['ID'] == ID) & (qw['Quarter'] <= 3)]
#     a = sum(qwtemp['PTS'])
#     b = (qwtemp['Gamedate']).unique()
#     threequartersummary.insert(IDs.index(ID), [ID, a, b])

qw = pd.DataFrame(quartersummary)
qw = qw.rename(columns={0: 'ID', 1: 'PTS', 2: 'Gamedate',3:'PTS2',4:'PTS3',5:'PTS4'})
qwfilterhome = qw.filter(['ID', 'PTS','PTS2','PTS3','PTS4'])
qwfilteraway = qw.filter(['ID', 'PTS','PTS2','PTS3','PTS4'])
schedule['home_teamID'] = schedule['HOME_TEAM_ABBREVIATION'] + schedule['start_time']
schedule['away_teamID'] = schedule['AWAY_TEAM_ABBREVIATION'] + schedule['start_time']
schedule['TotalScore'] = schedule['home_team_score'] + schedule['away_team_score']
opponent = schedule.filter(['home_teamID', 'Season Year', 'TotalScore', 'away_teamID'])
probdata = pd.merge(qwfilterhome, opponent, left_on='ID', right_on='home_teamID', how='left')
probdata = pd.merge(probdata, qwfilteraway, left_on='away_teamID', right_on='ID', how='left')
probdata = probdata.dropna().drop_duplicates()
#probdata = probdata.rename(columns={'ID_x': 'HomeID', 'PTS_x': 'HomeQ2', 'PTS_y': 'AwayQ2'})
probdata = probdata.rename(columns={'ID_x': 'HomeID', 'PTS_x': 'HomeQ1', 'PTS_y': 'AwayQ1','PTS2_x': 'HomeQ2', 'PTS2_y': 'AwayQ2','PTS3_x': 'HomeQ3', 'PTS3_y': 'AwayQ3','PTS4_x': 'HomeQ4', 'PTS4_y': 'AwayQ4'})
#probdata = probdata.filter(['HomeID', 'HomeQ2', 'AwayQ2', 'TotalQ2'])
probdata = probdata.filter(['HomeID','HomeQ1', 'AwayQ1','HomeQ2', 'AwayQ2','HomeQ3', 'AwayQ3','HomeQ4', 'AwayQ4'])
probdata['TotalQ2'] = probdata['HomeQ2']+probdata['AwayQ2']
probdata['TotalQ3'] = probdata['HomeQ3']+probdata['AwayQ3']
probdata['TotalQ4'] = probdata['HomeQ4']+probdata['AwayQ4']
schedule = schedule.drop('key_0', axis=1).drop_duplicates()
schedule=pd.merge(schedule,probdata,left_on=schedule['home_teamID'],right_on=probdata['HomeID'],how='left')

# q2scoring = schedule.sort_values(['AWAY_TEAM_ABBREVIATION', 'start_time']).filter(
#     ['AwayQ2', 'AWAY_TEAM_ABBREVIATION', 'start_time', 'Season Year']).rename(
#     columns={'AWAY_TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}).append(
#     schedule.sort_values(['HOME_TEAM_ABBREVIATION', 'start_time']).filter(
#         ['HomeQ2', 'HOME_TEAM_ABBREVIATION', 'start_time', 'Season Year'])).sort_values(
#     ['HOME_TEAM_ABBREVIATION', 'start_time']).rename(
#     columns={'HOME_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'}).drop_duplicates()

# q2scoring=q2scoring[['HomeScheduleID','HomeQ2','AwayQ2']]
#q2scoring=schedule[['home_teamID','TotalQ2']]

qscoring=schedule[['home_teamID','HomeQ1', 'AwayQ1','HomeQ2', 'AwayQ2','HomeQ3', 'AwayQ3','TotalQ3','AwayQ4','TotalQ4']]
offdefrating = pd.read_csv(directory1+'offdefrating.csv',low_memory=False)
offdefrating['TEAM_NAME'] = offdefrating['TEAM_NAME'].str.replace(" ", "_").str.upper()
offdefrating = pd.merge(offdefrating, teammapping, left_on=offdefrating['TEAM_NAME'], right_on=teammapping['Team_Names'],how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'TEAM_ABBREVIATION'})
offdefrating['ID']= offdefrating['TEAM_ABBREVIATION'].astype(str)+offdefrating['Gamedate'].astype(str)
offdefratingH=offdefrating[['ID','OFF_RATING', 'DEF_RATING']]
offdefratingA=offdefrating[['ID','OFF_RATING', 'DEF_RATING']]
schedule = schedule.drop('key_0', axis=1)
schedule= pd.merge(schedule,offdefratingH,left_on=schedule['home_teamID'],right_on=offdefrating['ID'],how='left').rename(columns={'OFF_RATING':'HOFF_RATING','DEF_RATING':'HDEF_RATING'}).drop('ID',axis=1)
schedule = schedule.drop('key_0', axis=1)
schedule= pd.merge(schedule,offdefratingA,left_on=schedule['away_teamID'],right_on=offdefrating['ID'],how='left').rename(columns={'OFF_RATING':'AOFF_RATING','DEF_RATING':'ADEF_RATING'}).drop('ID',axis=1)
schedule=schedule.drop_duplicates()

scheduleoffdefH = []
tempscoring=pd.DataFrame()
for s in list(schedule['Season Year'].unique()):
    for i in list(schedule['HOME_TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = schedule[(schedule['HOME_TEAM_ABBREVIATION'] == i) & ((schedule['Season Year'] == s))].dropna().reset_index()
        except:
            tempscoring = schedule[(schedule['HOME_TEAM_ABBREVIATION'] == i) & ((schedule['Season Year'] == s))].drop('level_0',axis=1).dropna().reset_index()
        for j in range(averaging+1, len(tempscoring)):
            tempmeanframe = tempscoring.loc[j - (averaging+1):j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmeanframe['HOFF_RATING'])
            tempmean2 = np.mean(tempmeanframe['HDEF_RATING'])
            scheduleoffdefH.insert(0, [i, s, gamedate, tempmean,tempmean2])

scheduleoffdefH = pd.DataFrame(scheduleoffdefH).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg_HOFFRATING', 4: 'Avg_HDEFRATING'}).drop_duplicates()
scheduleoffdefH['ID']=scheduleoffdefH['TEAM_ABBREVIATION']+scheduleoffdefH['Date']
scheduleoffdefH=scheduleoffdefH.reset_index()

scheduleoffdefA = []
tempscoring=pd.DataFrame()

for s in list(schedule['Season Year'].unique()):
    for i in list(schedule['AWAY_TEAM_ABBREVIATION'].unique()):
        try:
            tempscoring = schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == i) & ((schedule['Season Year'] == s))].dropna().reset_index()
        except:
            tempscoring = schedule[(schedule['AWAY_TEAM_ABBREVIATION'] == i) & ((schedule['Season Year'] == s))].drop('level_0',axis=1).dropna().reset_index()
        for j in range(averaging+1, len(tempscoring)):
            tempmeanframe = tempscoring.loc[j - (averaging+1):j - 1]
            gamedate = tempscoring['start_time'][j]
            tempmean = np.mean(tempmeanframe['AOFF_RATING'])
            tempmean2 = np.mean(tempmeanframe['ADEF_RATING'])
            scheduleoffdefA.insert(0, [i, s, gamedate, tempmean,tempmean2])

scheduleoffdefA = pd.DataFrame(scheduleoffdefA).rename(columns={0: 'TEAM_ABBREVIATION', 1: 'Season Year', 2: 'Date', 3: 'Avg_AOFFRATING', 4: 'Avg_ADEFRATING'}).drop_duplicates()
scheduleoffdefA['ID']=scheduleoffdefA['TEAM_ABBREVIATION']+scheduleoffdefA['Date']
scheduleoffdefA=scheduleoffdefA.reset_index()

#Create new data file as interim step to training data
nbadatamodel = nbadata

#This will be the training data
tfdata = pd.DataFrame()

nbadatamodel = nbadatamodel.rename(columns={'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION', 'Gamedate_x': 'Gamedate', 'Season Year_x': 'Season Year'})
# nbadatamodel=nbadatamodel.drop('key_0',axis=1)

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
            players[['HomeDaysRest', 'Team Rank', 'ConferenceBinary', 'Opponent Rank', 'OpponentConferenceBinary',
                     'Season Year','Playoffs']] = teamdata.reset_index()[
                ['HomeDaysRest', 'Team Rank', 'ConferenceBinary', 'Opponent Rank', 'OpponentConferenceBinary',
                 'Season Year','Playoffs']]  # Add ,'AST','REB','TOV','STL', 'BLK','PF'
            players['AwayDaysRest'] = Opponentdf.reset_index()['HomeDaysRest'][0]
            players['Gamedate'] = game
            players['Team'] = team
            players['Opponent'] = Opponent
            players['TotalScore'] = totscore
            tfdata = tfdata.append(players)


tfdata=tfdata.drop_duplicates()
tfdata = tfdata.reset_index()
tfdatakeep = tfdata

tfdata = tfdatakeep

tfdata = tfdata.drop('index', axis=1)

tfdata['avgIDmain'] = tfdata['Team'] + tfdata['Gamedate']

tfdata['avgID2main'] = tfdata['Opponent'] + tfdata['Gamedate']

tfdata['QID'] = tfdata['Team']+tfdata['Gamedate']

teamscoring = teamscoring.reset_index()

teamscoring = teamscoring.filter(['avgID', 'Last_Game_Score']).drop_duplicates()

tfdata2 = pd.merge(tfdata, teamscoring, left_on=tfdata['avgIDmain'], right_on=teamscoring['avgID'], how='left').rename(columns={'avgID_x': 'avgID', 'Last_Game_Score': 'Team_Last_Game_Score'})

tfdata2 = tfdata2.drop('key_0', axis=1)

tfdata3 = pd.merge(tfdata2, teamscoring, left_on=tfdata2['avgID2main'], right_on=teamscoring['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Last_Game_Score': 'Opponent_Last_Game_Score'}).drop('avgID_y', axis=1)

tfdata3 = tfdata3.drop_duplicates()

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score']).drop_duplicates()

tfdata3 = tfdata3.drop('key_0', axis=1)

tfdata4 = pd.merge(tfdata3, avgrollteamscore, left_on=tfdata3['avgIDmain'], right_on=avgrollteamscore['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Team_Score'}).dropna()

tfdata4 = tfdata4.drop('key_0', axis=1)

tfdata5 = pd.merge(tfdata4, avgrollteamscore, left_on=tfdata4['avgID2main'], right_on=avgrollteamscore['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Team_Score'}).dropna()

tfdata5 = tfdata5.drop('key_0', axis=1)

avgrolltotalscore = avgrolltotalscore.filter(['avgID', 'Avg Score']).drop_duplicates()

tfdata6 = pd.merge(tfdata5, avgrolltotalscore, left_on=tfdata5['avgIDmain'], right_on=avgrolltotalscore['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Team_Total_Score'}).dropna()

tfdata6 = tfdata6.drop('key_0', axis=1)

tfdata7 = pd.merge(tfdata6, avgrolltotalscore, left_on=tfdata6['avgID2main'], right_on=avgrolltotalscore['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

tfdata7 = tfdata7.drop('key_0', axis=1)

tfdata8 = pd.merge(tfdata7, avgHomeStats, left_on=tfdata7['avgIDmain'], right_on=avgHomeStats['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

tfdata8 = tfdata8.drop('key_0', axis=1)

tfdata9 = pd.merge(tfdata8, avgAwayStats, left_on=tfdata8['avgID2main'], right_on=avgAwayStats['avgID'],how='left').rename(columns={'avgID_x': 'avgID', 'Avg Score': 'Average_Opp_Total_Score'}).dropna()

tfdata9 = tfdata9.drop('key_0', axis=1)

tfdata10 = pd.merge(tfdata9, qscoring, left_on=tfdata9['QID'], right_on=qscoring['home_teamID'],how='left').dropna().drop_duplicates()

tfdata10 = tfdata10.drop('key_0', axis=1)

tfdata11 = pd.merge(tfdata10, scheduleoffdefH, left_on=tfdata10['avgIDmain'], right_on=scheduleoffdefH['ID'],how='left').dropna().drop_duplicates().drop('ID',axis=1)

tfdata11 = tfdata11.drop('key_0', axis=1)

tfdata12 = pd.merge(tfdata11, scheduleoffdefA, left_on=tfdata11['avgID2main'], right_on=scheduleoffdefA['ID'],how='left').dropna().drop_duplicates().drop('ID',axis=1)

traindata = tfdata12
testdata=tfdata12

#traindata = tfdata10
#testdata=tfdata10
#traindata = traindata[(traindata['Gamedate'] <= '2021-02-01')]
#testdata = traindata[traindata['Gamedate'] > '2021-02-01']  #Comment out after checking model output

#Use this after checking results of testdata, train on all data available
traindata = traindata[(traindata['Gamedate'] <= '2021-11-01')]
testdata = testdata[testdata['Gamedate'] > '2021-11-01']  #Comment out after checking model output
TrainScore = traindata['TotalScore']
TestScore = testdata['TotalScore']


traindata = traindata.drop('TotalScore', axis=1)
testdata = testdata.drop('TotalScore', axis=1)

#traindata = tfdata12
traindata['S2015'] = (np.where(traindata['Season Year'] == '2015', 1, 0))
traindata['S2016'] = (np.where(traindata['Season Year'] == '2016', 1, 0))
traindata['S2017'] = (np.where(traindata['Season Year'] == '2017', 1, 0))
traindata['S2018'] = (np.where(traindata['Season Year'] == '2018', 1, 0))
traindata['S2019'] = (np.where(traindata['Season Year'] == '2019', 1, 0))
traindata['S2020'] = (np.where(traindata['Season Year'] == '2020', 1, 0))
traindata['S2021'] = (np.where(traindata['Season Year'] == '2021', 1, 0))
traindata['S2022'] = (np.where(traindata['Season Year'] == '2022', 1, 0))
traindata = traindata.drop('Season Year', axis=1)

testdata['S2015'] = (np.where(testdata['Season Year'] == '2015', 1, 0))
testdata['S2016'] = (np.where(testdata['Season Year'] == '2016', 1, 0))
testdata['S2017'] = (np.where(testdata['Season Year'] == '2017', 1, 0))
testdata['S2018'] = (np.where(testdata['Season Year'] == '2018', 1, 0))
testdata['S2019'] = (np.where(testdata['Season Year'] == '2019', 1, 0))
testdata['S2020'] = (np.where(testdata['Season Year'] == '2020', 1, 0))
testdata['S2021'] = (np.where(testdata['Season Year'] == '2021', 1, 0))
testdata['S2022'] = (np.where(testdata['Season Year'] == '2022', 1, 0))
testdata = testdata.drop('Season Year', axis=1)

tfdata = traindata[[
    'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
     'S2019', 'S2020', 'S2021','S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
     'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
     'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
     'AvgHBLK', 'AvgHFG3M', 'AvgHOREB','AvgHAST', 'AvgHREB', 'AvgHSTL','Playoffs','AvgHAST_PCT','AvgHAST_TO', 'AvgHAST_RATIO','AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
    'AvgHTM_TOV_PCT', 'AvgHEFG_PCT','AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE','AvgAAST_PCT','AvgAAST_TO', 'AvgAAST_RATIO','AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
    'AvgATM_TOV_PCT', 'AvgAEFG_PCT','AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE','TotalQ3','Avg_HOFFRATING', 'Avg_HDEFRATING','Avg_AOFFRATING','Avg_ADEFRATING']]

testdata = testdata[[
    'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
     'S2019', 'S2020', 'S2021','S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
     'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
     'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
     'AvgHBLK', 'AvgHFG3M', 'AvgHOREB','AvgHAST', 'AvgHREB', 'AvgHSTL','Playoffs','AvgHAST_PCT','AvgHAST_TO', 'AvgHAST_RATIO','AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
    'AvgHTM_TOV_PCT', 'AvgHEFG_PCT','AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE','AvgAAST_PCT','AvgAAST_TO', 'AvgAAST_RATIO','AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
    'AvgATM_TOV_PCT', 'AvgAEFG_PCT','AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE','TotalQ3','Avg_HOFFRATING', 'Avg_HDEFRATING','Avg_AOFFRATING','Avg_ADEFRATING']]

#tfdata.to_csv(directory1+'trainingset_offdefrating.csv')

# tfdata = traindata[[
#     'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2016', 'S2017','S2018',
#      'S2019', 'S2020', 'S2021','S2022','HomeQ3', 'AwayQ3','Avg_HOFFRATING', 'Avg_HDEFRATING','Avg_AOFFRATING','Avg_ADEFRATING']]  # Season Year can remove due to taking average of scores.


# tfdata = traindata[[
#     'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2016', 'S2017','S2018',
#      'S2019', 'S2020', 'S2021','S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
#      'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
#      'Playoffs','HomeQ1', 'AwayQ1','HomeQ2', 'AwayQ2','HomeQ3', 'AwayQ3']]  # Season Year can remove due to taking average of scores.
#
# testdata=testdata[['HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2016', 'S2017', 'S2018',
# 'S2019', 'S2020', 'S2021', 'S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
# 'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
# 'Playoffs','HomeQ1', 'AwayQ1','HomeQ2', 'AwayQ2','HomeQ3', 'AwayQ3']]

#
# tfdata = traindata[[
#     'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
#      'S2019', 'S2020', 'S2021','S2022', 'Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
#      'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
#      'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
#      'AvgHBLK', 'AvgHFG3M', 'AvgHOREB','AvgHAST', 'AvgHREB', 'AvgHSTL','Playoffs','AvgHAST_PCT','AvgHAST_TO', 'AvgHAST_RATIO','AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
#     'AvgHTM_TOV_PCT', 'AvgHEFG_PCT','AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE','AvgAAST_PCT','AvgAAST_TO', 'AvgAAST_RATIO','AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
#                         'AvgATM_TOV_PCT', 'AvgAEFG_PCT','AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE','HomeQ3', 'AwayQ3','Avg_HOFFRATING', 'Avg_HDEFRATING','Avg_AOFFRATING','Avg_ADEFRATING']]  # Season Year can remove due to taking average of scores.

# tftest = testdata[[
#     'HomeDaysRest', 'AwayDaysRest', 'ConferenceBinary', 'OpponentConferenceBinary', 'S2015', 'S2018', 'S2016', 'S2018',
#      'S2019', 'S2020', 'S2021', 'S2022','Team Rank', 'Opponent Rank', 'Opponent_Last_Game_Score', 'Team_Last_Game_Score',
#      'Average_Team_Score', 'Average_Opp_Team_Score', 'Average_Team_Total_Score', 'Average_Opp_Total_Score',
#      'AvgAPF', 'AvgAFGM', 'AvgABLK', 'AvgAFG3M', 'AvgAOREB', 'AvgAAST', 'AvgAREB', 'AvgASTL', 'AvgHPF', 'AvgHFGM',
#      'AvgHBLK', 'AvgHFG3M', 'AvgHOREB','AvgHAST', 'AvgHREB', 'AvgHSTL', 'Playoffs','AvgHAST_PCT','AvgHAST_TO', 'AvgHAST_RATIO','AvgHOREB_PCT', 'AvgHDREB_PCT', 'AvgHREB_PCT',
#     'AvgHTM_TOV_PCT', 'AvgHEFG_PCT','AvgHTS_PCT', 'AvgHE_PACE', 'AvgHPACE', 'AvgHPACE_PER40', 'AvgHPOSS', 'AvgHPIE','AvgAAST_PCT','AvgAAST_TO', 'AvgAAST_RATIO','AvgAOREB_PCT', 'AvgADREB_PCT', 'AvgAREB_PCT',
#                         'AvgATM_TOV_PCT', 'AvgAEFG_PCT','AvgATS_PCT', 'AvgAE_PACE', 'AvgAPACE', 'AvgAPACE_PER40', 'AvgAPOSS', 'AvgAPIE']]  # Season Year can remove due to taking average of scores.
#

tfdata = tfdata.reset_index()
tfdata = tfdata.drop('index',axis=1)

models = []
scores = []

cutpoints = list(range(190, 250))
cutpoints = [math.floor(float(x)) for x in cutpoints]

for cut in cutpoints:
    try:
        Target = np.where(TrainScore > float(cut), 1, 0)
        TestTarget = np.where(TestScore > float(cut), 1, 0) #Use for training and checking scores.
        X_train = tfdata
        X_test=testdata
        y_train = Target
        estimators = 1000
        param_test = {'max_depth': (1,2,3,4,5,6,7,8,9,10,15,20,25), 'subsample': (0.5, 0.6, 0.75, 0.8,0.9)}
        gsearch = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.01, n_estimators=estimators, max_features='sqrt',subsample=0.8, random_state=10, validation_fraction=0.3),param_grid=param_test, scoring='accuracy', n_jobs=4, cv=5)
        fitmodel = gsearch.fit(X_train, y_train) #Use for training and checking scores, use on complete data when done.
        score = fitmodel.score(X_test, TestTarget)
        scores.insert(cutpoints.index(cut), [cut, score])
        models.insert(cutpoints.index(cut), [cut, gsearch.best_estimator_])
    except:
        pass

#savemodeltofile
cutsonly=[item[0] for item in models]
modelsonly=[item[1] for item in models]
with open("/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/modelsQ3kq.pckl", "wb") as f:
    for model in modelsonly:
         pickle.dump(model, f)

#savemodeltofile
cutsonly=[item[0] for item in models]
modelsonly=[item[1] for item in models]
with open("/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/modelsQ2.pckl", "wb") as f:
    for model in modelsonly:
         pickle.dump(model, f)



traindata.to_csv(directory1+'traindataq3.csv')
