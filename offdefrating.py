# Import Libraries
import requests
import random
import json
import math
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from datetime import timedelta, date
import basketball_reference_web_scraper
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import schedule


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
schedule = pd.DataFrame()
schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(
    Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)
schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'
teammapping = pd.read_csv(directory1+'dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())
seasonyears = ['2021', '2020', '2019', '2018', '2017', '2016', '2015']

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
schedule = pd.DataFrame()
schedule = schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(
    Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)
schedule['start_time'] = schedule['start_time'] - timedelta(hours=12)
schedule['start_time'] = schedule['start_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
schedule['away_team'] = schedule['away_team'].astype('S')
awayteams = schedule['away_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['away_team'] = [x[1] for x in awayteams]
schedule['home_team'] = schedule['home_team'].astype('S')
hometeams = schedule['home_team'].apply(lambda x: x.decode("utf-8")).str.split('Team.').tolist()
schedule['home_team'] = [x[1] for x in hometeams]
# gamehist = schedule.loc[schedule['Season Year']!=currentyear]['start_time']
teammapping = pd.read_csv(directory1+'dataframe.csv')
schedule = pd.merge(schedule, teammapping, left_on=schedule['home_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'HOME_TEAM_ABBREVIATION'})
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, teammapping, left_on=schedule['away_team'], right_on=teammapping['Team_Names'],
                    how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'AWAY_TEAM_ABBREVIATION'})
conference = pd.DataFrame(data={'Team': ['BOS', 'GSW', 'HOU', 'CLE', 'ORL', 'UTA', 'BKN', 'DET', 'MIN',
                                         'NOP', 'PHI', 'WAS', 'MEM', 'SAC', 'CHA', 'POR', 'SAS', 'IND',
                                         'ATL', 'DAL', 'PHX', 'MIA', 'MIL', 'DEN', 'OKC', 'LAL', 'TOR',
                                         'NYK', 'LAC', 'CHI'],
                                'Conference': ['E', 'W', 'W', 'E', 'E', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W',
                                               'W', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'E', 'E', 'W', 'W', 'W',
                                               'E', 'E', 'W', 'E']})
seasonyears = ['2022','2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, conference, left_on=schedule['HOME_TEAM_ABBREVIATION'], right_on=conference['Team'],
                    how='left').drop('Team', axis=1).rename(columns={'Conference': 'Conference_Home'})

schedule = schedule.drop('key_0', axis=1)
schedule = pd.merge(schedule, conference, left_on=schedule['AWAY_TEAM_ABBREVIATION'], right_on=conference['Team'],
                    how='left').drop('key_0', axis=1).drop('Team', axis=1).rename(
    columns={'Conference': 'Conference_Away'})
nbateams = pd.DataFrame(schedule['HOME_TEAM_ABBREVIATION'].unique())
seasonyears = ['2022','2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
# We want to add the new data to our original file, therefore we filter on the new season and compare the dates in our file to the games that have been played.
offdefratinglist = []
notcaptured = []
dataytd = []
dates = pd.DataFrame()
datestemp = pd.DataFrame()
offdefrating = pd.DataFrame()

for s in seasonyears:
    scheduletemp = schedule[(schedule['Season Year'] == s)]
    dataytdtemp = scheduletemp.filter(['start_time', 'home_team_score']).dropna()['start_time'].drop_duplicates()
    dataytd.insert(seasonyears.index(s), dataytdtemp)
    dates = dates.append(dataytd[seasonyears.index(s)].tolist())
    # dates = dataytd[0].tolist()+dataytd[1].tolist()+dataytd[2].tolist()+dataytd[3].tolist()+dataytd[4].tolist()+dataytd[5].tolist()+dataytd[6].tolist()

offdefratingdata = pd.read_csv(directory1+'offdefrating.csv')
captureddates = pd.DataFrame(offdefratingdata['Gamedate'].unique())
datelist=list(dates[0])
datelist = [x for x in datelist if x > max(captureddates[0])]

# leftover games are generally the playoff games.
# datelist = [x for x in datelist if x > max(captureddates[0])]
#datelist = [x for x in dates[0].tolist() if x not in captureddates[0].tolist()]
# The following for loop reads the new data from nba.com into a list
for e in range(0, len(datelist)):
    try:
        year1 = datelist[e][:4]
        month1 = datelist[e][5:7]
        day1 = datelist[e][8:10]
        a = 'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
        if datelist[e] in dataytd[0].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[1].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[2].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[3].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[4].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[5].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[6].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[7].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[8].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2013-14&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        c = '&DateTo='
        d = '%2F'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
            'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/',
            'Host': 'stats.nba.com', 'Origin': 'https://www.nba.com'}
        url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
        r = requests.get(url, headers=headers)
        numrecords = len(r.json()['resultSets'][0]['rowSet'])
        fields = r.json()['resultSets'][0]['headers']
        data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
        for i in range(0, numrecords):
            records = r.json()['resultSets'][0]['rowSet'][i]
            for j in range(0, len(records)):
                data.iloc[[i], [j]] = records[j]
        offdefratinglist.insert(e, [data, datelist[e]])
    except:
        pass
offdefratingworking = pd.DataFrame()
offdefratingtemp = pd.DataFrame()
# Read the data from the website into a dataframe
for i in range(0, len(offdefratinglist)):
    offdefratingtemp = pd.DataFrame(offdefratinglist[i][0])
    offdefratingtemp['Gamedate'] = offdefratinglist[i][1]
    offdefratingworking = offdefratingworking.append(offdefratingtemp)

offdefratingplayoffslist = []

for e in range(0, len(datelist)):
    try:
        year1 = datelist[e][:4]
        month1 = datelist[e][5:7]
        day1 = datelist[e][8:10]
        a = 'https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom='
        if datelist[e] in dataytd[0].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2021-22&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[1].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[2].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[3].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[4].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[5].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[6].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[7].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        elif datelist[e] in dataytd[8].tolist():
            b = '&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2013-14&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
        c = '&DateTo='
        d = '%2F'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
            'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true', 'Referer': 'https://www.nba.com/',
            'Host': 'stats.nba.com', 'Origin': 'https://www.nba.com'}
        url = a + month1 + d + day1 + d + year1 + c + month1 + d + day1 + d + year1 + b
        r = requests.get(url, headers=headers)

        numrecords = len(r.json()['resultSets'][0]['rowSet'])
        fields = r.json()['resultSets'][0]['headers']
        data = pd.DataFrame(index=np.arange(numrecords), columns=fields)
        for i in range(0, numrecords):
            records = r.json()['resultSets'][0]['rowSet'][i]
            for j in range(0, len(records)):
                data.iloc[[i], [j]] = records[j]
        offdefratingplayoffslist.insert(e, [data, datelist[e]])
    except:
        pass
offdefratingplayoffsworking = pd.DataFrame()
offdefratingplayoffstemp = pd.DataFrame()
# Read the data from the website into a dataframe
for i in range(0, len(offdefratingplayoffslist)):
    offdefratingplayoffstemp = pd.DataFrame(offdefratingplayoffslist[i][0])
    offdefratingplayoffstemp['Gamedate'] = offdefratingplayoffslist[i][1]
    offdefratingplayoffsworking = offdefratingplayoffsworking.append(offdefratingplayoffstemp)

offdefrating = offdefrating.append(offdefratingworking).append(offdefratingplayoffsworking)

offdefrating = offdefrating.drop_duplicates()

offdefratingdata = offdefratingdata.append(offdefrating)

offdefratingdata.to_csv(directory1+'offdefrating2.csv')

