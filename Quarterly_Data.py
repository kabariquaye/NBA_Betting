# Quarterly Betting
# The goal of this model will be to bet on games in the 3rd quarter.
# Timing is going to be the biggest issue for this modelling.


#Import Libraries
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta,date
from basketball_reference_web_scraper import client
directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'
from basketball_reference_web_scraper.data import OutputType

standing=pd.DataFrame()

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
schedulelist = [Schedule2020,Schedule2019,Schedule2018,Schedule2017,Schedule2016,Schedule2015,Schedule2014]
Schedule2020.filter(['start_time','home_team_score'])

schedule=pd.DataFrame()

#Append each schedule and do appropriate conversions
schedule=schedule.append(Schedule2022).append(Schedule2021).append(Schedule2020).append(Schedule2019).append(Schedule2018).append(Schedule2017).append(Schedule2017).append(Schedule2016).append(Schedule2015).append(Schedule2014)
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

seasonyears = ['2022','2021','2020','2019','2018','2017','2016','2015','2014']

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
quarterdata=pd.read_csv(directory1+'quarterdata.csv',low_memory=False)
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
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[2].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[3].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[4].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[5].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[6].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[7].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[8].tolist():
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
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2020-21&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[2].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2019-20&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[3].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2018-19&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[4].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2017-18&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[5].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2016-17&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[6].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2015-16&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[7].tolist():
                    b='&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period='+str(period+1)+'&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2014-15&SeasonSegment=&SeasonType=Playoffs&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
                elif datelist[e] in dataytd[8].tolist():
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
    quarterworkingplayoffs=quarterworkingplayoffs.append(quartertemp)


quarterdata=quarterdata.append(quarterworking)
quarterdata=quarterdata.append(quarterworkingplayoffs)
directory1='/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/'
quarterdata.to_csv(directory1+'quarterdata.csv')

# quarterdata = pd.read_csv('/Users/kabariquaye/PycharmProjects/pythonProject1/venv/data/quarterdata.csv')
# quarterdata['TEAM_NAME'] = quarterdata['TEAM_NAME'].str.replace(" ", "_").str.upper()
#
# quarterdata = pd.merge(quarterdata, teammapping, left_on=quarterdata['TEAM_NAME'], right_on=teammapping['Team_Names'],
#                        how='left').drop('Team_Names', axis=1).rename(columns={'Mapping': 'TEAM_ABBREVIATION'})
# qw = quarterdata
# qw = qw.dropna()
# qw['ID'] = qw['TEAM_ABBREVIATION'] + qw['Gamedate']
# IDs = list(qw['ID'].unique())
# halfsummary = []
# threequartersummary = []
#
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
# qw = pd.DataFrame(halfsummary)
# qw2 = pd.DataFrame(threequartersummary)
# qw = qw.rename(columns={0: 'ID', 1: 'PTS', 2: 'Gamedate'})
# qwfilterhome = qw.filter(['ID', 'PTS'])
# qwfilteraway = qw.filter(['ID', 'PTS'])
# schedule['home_teamID'] = schedule['HOME_TEAM_ABBREVIATION'] + schedule['start_time']
# schedule['away_teamID'] = schedule['AWAY_TEAM_ABBREVIATION'] + schedule['start_time']
# schedule['TotalScore'] = schedule['home_team_score'] + schedule['away_team_score']
# opponent = schedule.filter(['home_teamID', 'Season Year', 'TotalScore', 'away_teamID'])
# probdata = pd.merge(qwfilterhome, opponent, left_on='ID', right_on='home_teamID', how='left')
# probdata = pd.merge(probdata, qwfilteraway, left_on='away_teamID', right_on='ID', how='left')
# probdata = probdata.dropna()
# probdata = probdata.rename(columns={'ID_x': 'HomeID', 'PTS_x': 'HomeQ2', 'PTS_y': 'AwayQ2'})
# # probdata['TotalQ2']=probdata['HomeQ2']+probdata['AwayQ2']
# probdata = probdata.filter(['HomeID', 'HomeQ2', 'AwayQ2', 'TotalQ2', 'TotalScore', 'Season Year'])
# #
# # uniquetotscore = list(unique(probdata['TotalScore']))
# # unique3score = list(unique(probdata['TotalQ3']))
# #
# # distribution2 = []
# # distribution3 = []
# #
# # for i in uniquetotscore:
# #     for j in unique3score:
# #         distribution2.insert(unique3score.index(j), [i, j, len(
# #             probdata[(probdata['TotalQ2'] <= j) & (probdata['TotalQ2'] >= j) & (probdata['TotalScore'] >= i)]), len(
# #             probdata[(probdata['TotalQ3'] <= j + 5) & (probdata['TotalQ3'] >= j - 5)]), len(probdata[(probdata[
# #                                                                                                           'TotalQ3'] <= j + 5) & (
# #                                                                                                                  probdata[
# #                                                                                                                      'TotalQ3'] >= j - 5) & (
# #                                                                                                                  probdata[
# #                                                                                                                      'TotalScore'] >= i)]) / len(
# #             probdata[(probdata['TotalQ3'] <= j) & (probdata['TotalQ3'] >= j)])])
# #         distribution3.insert(unique3score.index(j), [i, j, len(
# #             probdata[(probdata['TotalQ2'] <= j) & (probdata['TotalQ2'] >= j) & (probdata['TotalScore'] <= i)]), len(
# #             probdata[(probdata['TotalQ3'] <= j + 5) & (probdata['TotalQ3'] >= j - 5)]), len(probdata[(probdata[
# #                                                                                                           'TotalQ3'] <= j + 5) & (
# #                                                                                                                  probdata[
# #                                                                                                                      'TotalQ3'] >= j - 5) & (
# #                                                                                                                  probdata[
# #                                                                                                                      'TotalScore'] <= i)]) / len(
# #             probdata[(probdata['TotalQ3'] <= j) & (probdata['TotalQ3'] >= j)])])
# #
# # distrover = pd.DataFrame(distribution2).rename(
# #     columns={0: 'TotalScore', 1: 'TotalQ3', 2: 'Freq', 3: 'Total', 4: 'Probability'})
# # distrunder = pd.DataFrame(distribution3).rename(
# #     columns={0: 'TotalScore', 1: 'TotalQ3', 2: 'Freq', 3: 'Total', 4: 'Probability'})
# #
