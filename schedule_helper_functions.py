# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:24:16 2018

@author: mjpbb
"""
import pandas as pd
import datetime as dt
#from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def strip(text):
    # remove spaces from string
    try:
        return text.strip()
    except AttributeError:
        return text

def checkTeamList(s_df,d_df,t_df):
    # verify that teams in schedule are in distance/time matrices and viceversa
    teams_sched=sorted(pd.concat([s_df['Team1'],s_df['Team2']]).unique())
    teams_loc=sorted(s_df['Location'].unique())
    teams_dist=sorted(d_df.index.values)
    teams_time=sorted(t_df.index)
    diff_dist1=np.setdiff1d(teams_sched,teams_dist)
    diff_dist2=np.setdiff1d(teams_dist,teams_sched)
    diff_time1=np.setdiff1d(teams_sched,teams_time)
    diff_time2=np.setdiff1d(teams_time,teams_sched)
    diff_loc1=np.setdiff1d(teams_loc,teams_dist)
    diff_loc2=np.setdiff1d(teams_loc,teams_time)
    if len(diff_loc1)>0:
        print('The location(s) below were in the schedule but not the distance matrix:')
        print(diff_loc1)
        raise NameError('Add location to distance matrix!')
    if len(diff_loc2)>0:
        print('The location(s) below were in the schedule but not the time matrix:')
        print(diff_loc2)
        raise NameError('Add location to time matrix!')
    if len(diff_dist1)>0:
        print('The team(s) below were in the schedule but not the distance matrix:')
        print(diff_dist1)
        raise NameError('Add team to distance matrix!')    
    if len(diff_dist2)>0:
        d_df.drop(diff_dist2,inplace=True)
        d_df.drop(diff_dist2,inplace=True,axis=1)
    if len(diff_time1)>0:
        print('The team(s) below were in the schedule but not the time matrix:')
        print(diff_time1)
        raise NameError('Add team to time matrix!')    
    if len(diff_time2)>0:
        t_df.drop(diff_time2,inplace=True)
        t_df.drop(diff_time2,inplace=True,axis=1)
    print("Teams from distance/time matrices NOT active in the schedule:")
    print(list(set(diff_dist2).union(set(diff_time2))))
    print("Teams active in the schedule:")
    print(teams_sched)

    return teams_sched,d_df,t_df


def format_timedelta(x):
    # remove seconds from travel time
    x=str(x)
    x=x[:-3]
    return x

def get_team_schedule_dataframes(fullsched_df,teams,params,time_df,dist_df):
    # generate dataframe containing dataframes with each team's schedule
    team_dfs={}
    for team in teams: #teams
        team_df=GetTeamSchedule(fullsched_df,
                                             team,
                                             params,
                                             time_df,
                                             dist_df)
        team_df['Total travel time for trip']=team_df['Total travel time for trip'].apply(format_timedelta)
        team_df['Total time off campus']=team_df['Total time off campus'].apply(format_timedelta)
        team_df['Total missed class time']=team_df['Total missed class time'].apply(format_timedelta)
        team_dfs[team]=team_df.copy()
    return team_dfs


def GetTeamSchedule(df,team,params,time_df,dist_df):
    s_df=df[["Datetime","Location"]].copy()
    s_df['Opponent']=df.apply(lambda x: GetOpponent(x,team),axis=1)
    s_df['Day of week']=GetDay(df['Datetime'])
    s_df.dropna(inplace=True)
    s_df.reset_index(inplace=True) 
    s_df.drop('index',inplace=True,axis=1)
    s_df=GetTravelInfo(s_df,team,params,time_df,dist_df)
    s_df=FixSchedule(s_df,team,params)
    s_df=AddDistTimes(s_df,team,time_df,dist_df)
    s_df=AddTotals(s_df)
    s_df=ReorderRenameColumns(s_df)
    return s_df

def GetDay(dates):
    daylist=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    res=[]
    for day in dates:
        daynum=day.weekday()
        res.append(daylist[daynum])
    return res


def AddTotals(s_df):
    tot=pd.DataFrame(index=['Totals'])
    for col in ['totalTripDistance',
                'totalTripTime',
                'nightsAway',
                'offCampusTime',
                'missedClassTime']:
        tot[col]=s_df[col].sum()
    #tot.rename(index={0:'Totals'},inplace=True)
    s_df=pd.concat([s_df,tot],sort=False)
    #print(s_df)
    return s_df

def ReorderRenameColumns(s_df):
    new_names={'Datetime': 'Event time', 
               'Day of week': 'Day of week', 
               'Opponent': 'Opponent', 
               'Location': 'Event location', 
               'nextLocation': 'Next stop',
               'beforeDepTimes': 'Departure Time (from home)',
               'afterArrTimes': 'Arrival Time (at home)',
               'totalTripDistance': 'Total distance for trip', 
               'totalTripTime': 'Total travel time for trip',
               'nightsAway': 'Number of nights off campus', 
               'offCampusTime': 'Total time off campus',
               'missedClassTime': 'Total missed class time'}
    s_df.rename(columns=new_names,inplace=True)
    new_col_order=list(new_names.values())
    s_df=s_df[new_col_order]
    return s_df

def AddDistTimes(s_df,team,time_df,dist_df):
    distances=[]
    times=[]
    first=True
    curtripdist=0
    curtriptime=dt.timedelta(hours=0)
    for lastgame in s_df.index: 
        if not first:
            prevloc=s_df.loc[lastgame-1,'nextLocation']
        else:
            prevloc=team
            first=False    
        nextloc=s_df.loc[lastgame,'nextLocation']
        curtripdist=curtripdist+dist_df.loc[prevloc,s_df.loc[lastgame,'Location']]
        curtripdist=curtripdist+dist_df.loc[s_df.loc[lastgame,'Location'],nextloc]
        curtriptime=curtriptime+time_df.loc[prevloc,s_df.loc[lastgame,'Location']]
        curtriptime=curtriptime+time_df.loc[s_df.loc[lastgame,'Location'],nextloc]
        if nextloc==team: # normal trip
            distances.append(curtripdist)
            times.append(curtriptime)
            curtripdist=0
            curtriptime=dt.timedelta(hours=0)
        elif nextloc!=team: # one leg of multicity
            distances.append(0)
            times.append(dt.timedelta(hours=0))                    
    s_df['totalTripDistance']=distances
    s_df['totalTripTime']=times
    return s_df

# add columns for overnight trip, long trip, missed class time
def GetOpponent(df,team):
    if df['Team1']==team:
        return df['Team2']
    elif df['Team2']==team:
        return df['Team1']
    else:
        return None

def GetBeforeTravelInfo(gametime,traveltime,params):
    departureLimit=dt.datetime.combine(gametime,params['morning_depart'])
    overnight=0
    if traveltime==dt.timedelta(hours=0): #home game
        depTime=None
        arrTime=None
        return depTime,arrTime,overnight
    else: # away game
        depTime=gametime+params['arrival_buffer']-traveltime
        arrTime=gametime+params['arrival_buffer']
        if depTime<departureLimit: # departure too early
            depTime=dt.datetime.combine(gametime,params['evening_depart'])+params["prevday"]
            arrTime=depTime+traveltime
            overnight=1
        return depTime,arrTime,overnight
    
    
def GetAfterTravelInfo(gametime,traveltime,params):
    if traveltime==dt.timedelta(hours=0): #home game
        depTime=None
        arrTime=None
    else:
        depTime=gametime+params['event_duration']
        arrTime=depTime+traveltime
    return depTime,arrTime


def GetTravelInfo(s_df,team,params,time_df,dist_df):
    tripDuration=[]
    beforeDepTimes=[]
    beforeArrTimes=[]
    afterDepTimes=[]
    afterArrTimes=[]
    overnights=[]
    tripDist=[]
    offCampusTimes=[]
    missedClassTimes=[]
    for row in s_df.index:
        loc=s_df.iloc[row]['Location']
        gametime=s_df.iloc[row]['Datetime']
        traveltime=time_df[team][loc]
        dist=dist_df[team][loc]
        bdt,bat,on=GetBeforeTravelInfo(gametime,traveltime,params)
        adt,aat=GetAfterTravelInfo(gametime,traveltime,params)
        tripDuration.append(traveltime)
        beforeDepTimes.append(bdt)
        beforeArrTimes.append(bat)
        afterDepTimes.append(adt)
        afterArrTimes.append(aat)
        overnights.append(on)
        if bdt is None:
            offCampusTimes.append(dt.timedelta(hours=0))
        else:
            offCampusTimes.append(aat-bdt)
        missedClassTimes.append(dt.timedelta(hours=0))
        tripDist.append(dist)
    
    #tripDuration=pd.DataFrame(data=tripDuration,columns=["TravelTime"])
    s_df['timeFromHome']=tripDuration
    s_df['distFromHome']=tripDist
    s_df['beforeDepTimes']=beforeDepTimes
    s_df['beforeArrTimes']=beforeArrTimes
    s_df['afterDepTimes']=afterDepTimes
    s_df['afterArrTimes']=afterArrTimes
    s_df['offCampusTime']=offCampusTimes
    s_df['nightsAway']=overnights
    s_df['nextLocation']=team
    s_df['missedClassTime']=missedClassTimes
    return s_df


def FixSchedule(s_df,team,params):
    #get next locations
    nextList=[]
    #overnight=0
    for lastgame in s_df.index:
        if (lastgame==s_df.index[-1]): 
            nextList.append(team)
        elif (s_df.iloc[lastgame+1]['Location']==team) |(s_df.iloc[lastgame]['Location']==team):
            nextList.append(team)
        elif s_df.iloc[lastgame]['afterArrTimes']+params['home_stay']<s_df.iloc[lastgame+1]['beforeDepTimes']:
            nextList.append(team)
        else:
            nextList.append(s_df.iloc[lastgame+1]['Location'])
            s_df.at[lastgame,'afterDepTimes']=None
            s_df.at[lastgame,'afterArrTimes']=None
            s_df.at[lastgame+1,'beforeDepTimes']=None
            s_df.at[lastgame+1,'beforeArrTimes']=None
    s_df['nextLocation']=nextList
      
        
    #compute missed class time and off campus time
    ontrip=False
    for lastgame in s_df.index:    
        nextloc=s_df.iloc[lastgame]['nextLocation']
        if (nextloc!=team) & (ontrip==False):
            firstgame=lastgame
            #print('case 1:',lastgame)
            depTime=s_df.iloc[lastgame]['beforeDepTimes']
            s_df.at[lastgame,'offCampusTime']=dt.timedelta(hours=0)
            ontrip=True
        elif (nextloc!=team) & (ontrip==True):
            #print('case 2:',lastgame)
            s_df.at[lastgame,'offCampusTime']=dt.timedelta(hours=0)
            ontrip=True
        elif (nextloc==team) & (ontrip==True):
            #print('case 3:',lastgame)
            arrTime=s_df.iloc[lastgame]['afterArrTimes']
            misstime=GetMissedClassTime(depTime,arrTime,params)
            s_df.at[lastgame,'offCampusTime']=arrTime-depTime
            s_df.at[lastgame,'missedClassTime']=misstime
            for j in range(firstgame,lastgame):
                s_df.at[j,'nightsAway']=0
            timeaway=(arrTime-depTime)
            s_df.at[lastgame,'nightsAway']=timeaway.days
            ontrip=False
        else:
            depTime=s_df.iloc[lastgame]['beforeDepTimes']
            arrTime=s_df.iloc[lastgame]['afterArrTimes']
            misstime=GetMissedClassTime(depTime,arrTime,params)
            s_df.at[lastgame,'missedClassTime']=misstime
    return s_df

def GetMissedClassTime(depTime,arrTime,params):
    if (depTime=='None') & (arrTime=='None'):
            return dt.timedelta(hours=0)
    else:
        k=0;
        theTime=depTime
        while theTime<arrTime:
            if ClassTime(theTime,params):
                k+=1
            theTime=theTime+dt.timedelta(hours=1)
        return dt.timedelta(hours=k)

def ClassTime(theTime,params):
    starttime=dt.datetime.combine(theTime.date(),params['class_start'])
    endtime=dt.datetime.combine(theTime.date(),params['class_end'])
    theDay=theTime.weekday()
    if (theTime >=starttime)&(theTime<endtime)&(theDay<=4):
        return True
    else:
        return False

def GetNormalizationFactors(dist_df,time_df,params):
    numTeams=len(dist_df.index)
    games=(numTeams-1)*2
    distNorm=dist_df.sum()*2/games
    travelTimeNorm=time_df.sum()*2/games
    offCampusTime=(time_df.sum()*2+(-params['arrival_buffer']+params['event_duration'])*games/2)/games
    norm=pd.DataFrame()
    norm['distance']=distNorm
    norm['traveltime']=travelTimeNorm
    norm['offCampusTime']=offCampusTime
    return norm#.transpose()

def GetScheduleStatistics(s_df,team,dist_df,statparams,norm):
    print(s_df)
    # need to modify tripDist Column so that it reflects multicity trips
    nameslist=[]
    statslist=[]
    nameslist.append('totGames')
    statslist.append(len(s_df))
    nameslist.append('homeGames')
    homeGames=(s_df['location']==team).sum()
    statslist.append(homeGames)
    nameslist.append('awayGames')
    awayGames=statslist[0]-statslist[1]
    statslist.append(awayGames) 
    
    nameslist.append('totDist')
    totDist=GetTravelDistTotal(s_df,team,dist_df)
    statslist.append(totDist)
    nameslist.append('mean distance (away)')
    statslist.append(totDist/(awayGames*1.0))
    nameslist.append('normalized distance')
    statslist.append(statslist[-1]/norm['distance'][team])
    nameslist.append('median distance (away)')
    statslist.append(s_df[s_df['tripDist']>0]['tripDist'].median())
    nameslist.append('maximum distance')
    statslist.append(s_df['tripDist'].max())
    string='# of long trips (distance>' + str(statparams['TripDistanceThreshold']) + ' miles)'
    nameslist.append(string)
    statslist.append((s_df['tripDist']>=statparams['TripDistanceThreshold']).sum())
    
    nameslist.append('mean travel time (away)')
    statslist.append(s_df[s_df['tripTime']>dt.timedelta(hours=0)]['tripTime'].mean())
    nameslist.append('normalized time')
    statslist.append(statslist[-1]/norm['traveltime'][team])
    nameslist.append('median travel time (away)')
    statslist.append(s_df[s_df['tripTime']>dt.timedelta(hours=0)]['tripTime'].median())
    nameslist.append('maximum travel time')
    statslist.append(s_df['tripTime'].max())
    string='# of long trips (time>' + str(statparams['TripTimeThreshold']) + ' hours)'
    nameslist.append(string)
    statslist.append((s_df['tripTime']>=statparams['TripTimeThreshold']).sum())
    
    nameslist.append('total off campus time')
    statslist.append(s_df['offCampusTime'].sum())
    nameslist.append('average off campus trip')
    statslist.append(s_df['offCampusTime'].mean())
    nameslist.append('normalized off campus time')
    statslist.append(statslist[-1]/norm['offCampusTime'][team])
    nameslist.append('longest off campus trip')
    statslist.append(s_df['offCampusTime'].max())
    string='# of long trips (off campus time >' + str(statparams['OffCampusThreshold']) + ' hours)'
    nameslist.append(string)
    statslist.append((s_df['offCampusTime']>=statparams['OffCampusThreshold']).sum())
    
    nameslist.append('total missed class hours')
    statslist.append(s_df['missedClassTime'].sum())
    nameslist.append('average missed class hours (per game)')
    statslist.append(s_df['missedClassTime'].sum()/len(s_df))
    nameslist.append('longest trip (missed class hours)')
    statslist.append(s_df['missedClassTime'].max())
    
    stats=pd.DataFrame(data=[statslist],columns=nameslist,index=[team])
    return stats

def GetPath(s_df,team):
    list=[team]
    k=0;
    for k in s_df.index:
        nextgame=s_df.iloc[k]['location']
        nextstop=s_df.iloc[k]['nextLocation']
        if nextgame!=list[-1]:
            list.append(nextgame)
        if nextstop!=list[-1]:
            list.append(nextstop)      
    return list

def GetTravelDistTotal(s_df,team,dist_df): 
    path=GetPath(s_df,team)
    print(path)
    tot=0
    for k in range(0,len(path)-1):
        tot+=dist_df.loc[path[k],path[k+1]]
    return tot


def get_summary_stats(df,team,dist_df):
    # compute summary statistics for a single team
    summary=pd.DataFrame(index=[team])
    """
    'Event time', 'Opponent', 'Event location', 'Next stop',
       'Departure Time (from home)', 'Arrival Time (at home)',
       'Total distance for trip', 'Total travel time for trip',
       'Number of nights off campus', 'Total time off campus',
       'Total missed class time'
    """
    summary['Number of nights off campus']=df.loc['Totals','Number of nights off campus']
    summary['Total distance traveled']=df.loc['Totals','Total distance for trip']
    summary['Total travel time'] = df.loc['Totals','Total travel time for trip']
    summary['Total time off campus'] =df.loc['Totals','Total time off campus']
    summary['Total missed class time']=df.loc['Totals','Total missed class time']
    df=df.drop('Totals').copy()
    # get number of games for each day
    days=(df['Event time'].dt.dayofweek).astype(int)
    games_by_day=days.groupby(days).count()
    for k in range(0,7):
        if k not in games_by_day.index:
            games_by_day[k]=0        
    summary['# of Monday games']=games_by_day[0]
    summary['# of Tuesday games']=games_by_day[1]
    summary['# of Wednesday games']=games_by_day[2]
    summary['# of Thursday games']=games_by_day[3]
    summary['# of Friday games']=games_by_day[4]
    summary['# of Saturday games']=games_by_day[5]
    summary['# of Sunday games']=games_by_day[6]
    # total games
    summary['# of games']=len(df['Event location'])
    # home games
    summary['# of home games']=(df['Event location']==team).sum()
    # away games
    summary['# of away games']=(df['Event location']==df['Opponent']).sum()
    # neutral games
    summary['# of neutral games']=summary['# of games']-summary['# of home games']-summary['# of away games']
    summary['Average distance (per trip)']=summary['Total distance traveled']/summary['# of games']
    summary['Average distance (round robin)']=dist_df[team].mean()
    summary['Normalized average distance (lower is better)']=summary['Average distance (per trip)']/summary['Average distance (round robin)']
    # longest road trip (by games)
    # normalized distance
    """
    'Number of nights off campus', 'Total distance traveled',
       'Total travel time', 'Total time off campus', 'Total missed class time',
       '# of Monday games', '# of Tuesday games', '# of Wednesday games',
       '# of Thursday games', '# of Friday games', '# of Saturday games',
       '# of Sunday games', '# of games', '# of home games', '# of away games',
       '# of neutral games', 'Average distance (per trip)'
     """
    new_names={'# of games':'# of games', 
     '# of home games': '# of home games', 
     '# of away games': '# of away games',
     '# of neutral games': '# of neutral games', 
     'Average distance (per trip)': 'Average distance (per trip)',
     'Average distance (round robin)': 'Average distance (round robin)',
     'Normalized average distance (lower is better)': 'Normalized average distance (lower is better)',
     'Total distance traveled': 'Total distance traveled',
     'Number of nights off campus': 'Number of nights off campus', 
     'Total time off campus': 'Total time off campus', 
     'Total missed class time': 'Total missed class time',
     'Total travel time': 'Total travel time',
      '# of Monday games':'# of Monday games', 
      '# of Tuesday games': '# of Tuesday games', 
      '# of Wednesday games': '# of Wednesday games',
      '# of Thursday games': '# of Thursday games', 
      '# of Friday games': '# of Friday games', 
      '# of Saturday games': '# of Saturday games',
      '# of Sunday games': '# of Sunday games'} 
    new_col_order=list(new_names.values())
    summary=summary[new_col_order]
    return summary

def str2td(df):
    # convert string to time delta
    res=[]
    for time in df:
        day,other =time.split(" days ")
        hour,minute=other.split(":")
        day=int(day)
        hour=int(hour)
        minute=int(minute)
        delta = dt.timedelta(days=day,hours=hour, minutes=minute)
        res.append(delta) 
    res=pd.DataFrame(res)
    return res

def get_summary_stats_df(teams,team_dfs,dist_df):
    # compute summary statistics for a all teams team
    summary_df=pd.DataFrame()
    for team in teams: #teams
        teamsum=get_summary_stats(team_dfs[team],team,dist_df)
        summary_df=pd.concat([summary_df,teamsum])
    total_summary=pd.DataFrame(index=['Totals'])
    for col in summary_df.columns:    
        total_summary[col]=summary_df[col].sum()
    
    
    
    total_summary['Total time off campus']=str(str2td(summary_df['Total time off campus']).sum()[0])[:-3]
    total_summary['Total missed class time']=str(str2td(summary_df['Total missed class time']).sum()[0])[:-3]
    total_summary['Total travel time']=str(str2td(summary_df['Total travel time']).sum()[0])[:-3]
    summary_df=pd.concat([summary_df,total_summary])
    
    summary_df.at['Totals','Normalized average distance (lower is better)']=summary_df.loc['Totals','Average distance (per trip)']/summary_df.loc['Totals','Average distance (round robin)']
  
    
    for col in ['Total time off campus','Total missed class time','Total travel time']:
        summary_df[col]=str2td(summary_df[col])[0].values
        summary_df[col]=summary_df[col].apply(lambda x: x/dt.timedelta(hours=1))
    return summary_df
        
def ComputeScheduleStats(sched_df,dist_df,time_df,params,statparams,norm):
    # compute stats for a single team schedule
    stat_df=pd.DataFrame()
    for team in dist_df.index:
        s_df=GetTeamSchedule(sched_df,team,params,time_df,dist_df)
        stat_df=stat_df.append(GetScheduleStatistics(s_df,team,dist_df,statparams,norm))
    return stat_df  

def plot_results_summaries(summary_df):
    # plot bar charts summarizing the schedule properties (number of games, average distance, nights off campus, games by day)
    s_df=summary_df.drop('Totals')
    
    
    f, axarr = plt.subplots(nrows=4,ncols=1,figsize=(7,16))
    
    games=s_df[['# of home games','# of away games','# of neutral games']].plot(kind='bar',width=0.8,ax=axarr[0])
    patches, labels = games.get_legend_handles_labels()
    games.legend(patches, labels, bbox_to_anchor=(1, -0.15))
    
    distances=s_df[['Average distance (per trip)', 'Average distance (round robin)']].plot(kind='bar',width=0.8,ax=axarr[1])
    patches, labels = distances.get_legend_handles_labels()
    distances.legend(patches, labels, bbox_to_anchor=(1, -0.15))
    
    nights=s_df[['Number of nights off campus']].plot(kind='bar',width=0.8,ax=axarr[2])
    patches, labels = nights.get_legend_handles_labels()
    nights.legend(patches, labels, bbox_to_anchor=(1, -0.15))
    
    days=s_df[[ '# of Monday games', 
          '# of Tuesday games', 
          '# of Wednesday games', 
          '# of Thursday games', 
          '# of Friday games', 
          '# of Saturday games',
          '# of Sunday games']].plot(kind='bar',width=0.8,color=['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494'],ax=axarr[3])
    patches, labels = days.get_legend_handles_labels()
    days.legend(patches, labels, loc='center left',bbox_to_anchor=(1, 0.5))
    
    f.subplots_adjust(hspace =0.5)
    plt.savefig('results_charts.png',bbox_inches='tight',pad_inches=0)

def td2hours(td):
    # convert time delta to hours
    return td.seconds/3600
def dt2time(td):
    #convert datetime to time string
    return td.strftime('%H:%M')

def generate_results(sched_filename,summary_df,team_dfs,params,teams): 
    # generate results and save to csv files
    description={'Next stop':"The next destination on the trip. This will either be "
         "home or the location of the next game",
         '# of home games': 'Total number of games played at home', 
         '# of away games': 'Total number of games played at opponent''s home',
         '# of neutral games': 'Total number of games played at a neutral location', 
         'Average distance (per trip)': 'Average distance traveled per game (total distance traveled divided by number of games)',
         'Average distance (round robin)': "Average distance in a schedule in " 
         "which every team plays every other team. This is a useful point of " 
         "comparison that reflects the fact that some institutions will necessarily " 
         "travel further due to their distance from theother institutions.",
         'Normalized average distance (lower is better)': "This is the average "
         "average distance per trip divided by the average distance for a round " 
         "robin schedule.  A value of 1 indicates that the schedule requires the "
         "same amount of travel as a generic round robin schedule.  A value less "
         "than 1 indicates that the schedule requires less travel. A value greater "
         "than 1 indicates that the schedule requires more travel.",
         'Total distance traveled': 'Total distance traveled in the entire schedule',
         'Number of nights off campus': "The total number of nights off campus. "
         "This excludes partial nights due to the return home.",
         'Total time off campus': "This is an estimate for the total time spent "
         "off campus during the season. It is based on estimates for the departure " 
         "time and the arrival time for each trip. Given in hours.", 
         'Total missed class time': "This is an estimate for the missed class time "
         "based on the number of weekday hours during regular class periods. Given in hours.",
         'Total travel time': 'Total time spent on traveling to and from games. Given in hours.',
          '# of Monday games':'Total number of games played of Mondays', 
          '# of Tuesday games': 'Total number of games played of Tuesdays', 
          '# of Wednesday games': 'Total number of games played of Wednesdays', 
          '# of Thursday games': 'Total number of games played of Thursdays', 
          '# of Friday games': 'Total number of games played of Fridays', 
          '# of Saturday games': 'Total number of games played of Saturdays',
          '# of Sunday games': 'Total number of games played of Sundays'} 
    description=pd.DataFrame.from_dict(description,orient="index")
    description=description.rename(columns={0: 'Description'})
    # format assumptions 
    assumptions={'arrival_buffer': [24-td2hours(params['arrival_buffer']),"""Teams must arrive this many hours before the event begins."""], 
            'event_duration': [td2hours(params['event_duration']),"""The competition ends this many hours after the start time"""], 
            'morning_depart': [dt2time(params['morning_depart']),"""If a team must leave must leave earlier than this time, they will choose to leave the preceding day"""], 
            'evening_depart': [dt2time(params['evening_depart']),"""If a team leaves the day before an event, they will depart at this time."""],
            #'night_return': dt.datetime(year=1,month=1,day=1,hour=2).time(),
            'home_stay': [td2hours(params['home_stay']),"""If a team spends less than this amount of time at home between games, they will travel directly to their next game."""], 
            'class_start': [dt2time(params['class_start']), """Class hours begin at this time on weekdays. Used to calculate missed class time"""], 
            'class_end': [dt2time(params['class_end']),"""Class hours end at this time on weekdays. Used to calculate missed class time"""]}
    
    assumptions=pd.DataFrame.from_dict(assumptions,orient="index")
    assumptions=assumptions.rename(columns={0: 'Parameter', 1: 'Explanation'})
    
    
    # write to file
    results_filename=sched_filename + '_results.xlsx'
    writer = pd.ExcelWriter(results_filename, engine='xlsxwriter')
    
    summary_df.to_excel(writer,'Summary')   
    for team in teams: #teams
        df=team_dfs[team]
        newind=list(np.arange(1, len(df)))
        newind.append('Totals')
        df.index = newind
        df.to_excel(writer,team)     
    assumptions.to_excel(writer,'Assumptions')
    description.to_excel(writer,'Description')                                  
    #writer.save()
    
    # add coloring to summary sheet
    #workbook=writer.book
    worksheet=writer.sheets['Summary']
    
    firstrow=2
    lastrow=summary_df.shape[0]
    template='{0}{1}:{0}{2}'
    #template2='${0}${1}:${0}${2}'
    for col in list('BCDEFGHIJKLMNOPQRST'):
        colstr=template.format(col,firstrow,lastrow)
        worksheet.conditional_format(colstr,{'type': '2_color_scale',
                                             'max_color': '#FF6666',
                                             'min_color': '#FFFFFF'})
    return results_filename