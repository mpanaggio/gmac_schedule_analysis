#%%  import packages
import pandas as pd
import datetime as dt
import schedule_helper_functions as aux
import os
#%% define parameters for solution
params={'arrival_buffer': dt.timedelta(hours=-4), 
        'event_duration': dt.timedelta(hours=4), 
        'morning_depart': dt.datetime(year=1,month=1,day=1,hour=8).time(), 
        'evening_depart': dt.datetime(year=1,month=1,day=1,hour=17).time(),
        #'night_return': dt.datetime(year=1,month=1,day=1,hour=2).time(),
        'home_stay': dt.timedelta(hours=12), 
        'nextday': dt.timedelta(days=1),
        'prevday': dt.timedelta(days=-1),
        'class_start': dt.datetime(year=1,month=1,day=1,hour=8).time(), 
        'class_end': dt.datetime(year=1,month=1,day=1,hour=17).time()}
statparams={'TripDistanceThreshold': 250,
            'TripTimeThreshold': dt.timedelta(hours=5), 
            'OffCampusThreshold': dt.timedelta(hours=24)}
print("Current parameters:")
print(pd.DataFrame.from_dict(params,orient='index',columns=['value']))
print(pd.DataFrame.from_dict(statparams,orient='index',columns=['value']))
#%% select schedule to importclean up schedule

print('Available schedules in {}:'.format(os.getcwd()))
print()
file_list=os.listdir() # list files in current directory
k=0
skip=[]
for file in file_list:
    if file.endswith('.csv') |  file.endswith('.xlsx'):
        print("Schedule {}: {}".format(k,file))
        k+=1
    else:
        skip.append(file)
file_list=[x for x in file_list if x not in skip]


nofile=True
while nofile:
    try:
        ind= int(input("Enter schedule file number: "))
        sched_filename=file_list[ind].split('.')[0]
        sched_filetype=file_list[ind].split('.')[1]
        warningflag=False
        if ('results' in sched_filename): 
            print('Warning: Filename contains "results".  This may be an output file.')
            warningflag=True
        elif ('TravelDistances' in sched_filename): 
            print('Warning: Filename contains "TravelDistances".  This may be a distance file.')
            warningflag=True
        elif ('TravelTimes' in sched_filename): 
            print('Warning: Filename contains "TravelTimes".  This may be a time file.')
            warningflag=True
        if warningflag:
            cont= input("Continue (y/n)? ")
            if (cont=='y') | (cont=='Y'):
                nofile=False
            else:
                print('Trying again.')            
        else:
             nofile=False
    except:
        print('Invalid selection! Index must be an integer.')
print()
print('Current schedule file:')
print(sched_filename + '.' +sched_filetype)
print()
#%%  import schedule 

try: 
    if sched_filetype=='xlsx':
        fullsched_df=pd.read_excel(sched_filename + '.xlsx',header=0)
    elif sched_filetype=='csv':
        fullsched_df=pd.read_csv(sched_filename + '.csv',header=0)
except:
     print("Import Error: Check input file.")
     print("File must be a valid csv or xlsx file.")


#%% remove spaces from column names
fullsched_df.columns=aux.strip(fullsched_df.columns)
fullsched_df.index=aux.strip(fullsched_df.index)
  
#%% verify that column names are correct
cols=fullsched_df.columns
necessary_cols=['Date','Team1','Team2','Location']

for col in necessary_cols:
    if col not in cols:
        raise NameError('Input file must contain a column named {}.'.format(col))
        
#%% Incorporate times if available
if 'Time' in cols:
    try: 
        fullsched_df['Datetime']=pd.to_datetime(pd.to_datetime(fullsched_df['Date']).apply(lambda x: x.strftime('%Y-%m-%d')) + ' ' + fullsched_df['Time'].apply(lambda x: x.strftime('%H:%M')))
        fullsched_df.drop(columns=['Date','Time'],inplace=True)
    except:
        raise IOError('Unable to parse time column. Check formatting.')
else:
    day_of_week=aux.GetDay(fullsched_df['Date'])
    weekdaytime='5:00 PM'
    weekendtime='2:00 PM'
    times=[weekdaytime if x not in ['Saturday','Sunday'] else weekendtime for x in day_of_week]
    print(times)
    fullsched_df['Time']=times
    print('Warning! Times not provided.  Assuming default values.')
    print('Weekday time:',weekdaytime)
    print('Weekend time:',weekendtime)
    print('')
    try: 
        fullsched_df['Datetime']=pd.to_datetime(pd.to_datetime(fullsched_df['Date']).apply(lambda x: x.strftime('%Y-%m-%d')) + ' ' + fullsched_df['Time'])
        fullsched_df.drop(columns=['Date','Time'],inplace=True)
    except:
        raise IOError('Unable to parse time column. Check formatting.')
    
        
       

fullsched_df.rename(columns={'Date': 'Datetime'},inplace=True)
# Remove spaces from column names

#%% import time and distance dataframes
try:
    time_df=pd.read_csv('TravelTimes.csv',header=0,index_col=0,parse_dates=True)
    time_df=time_df.applymap(lambda x: dt.timedelta(minutes=x))
    time_df.columns=aux.strip(time_df.columns)
    time_df.index=aux.strip(time_df.index)
except:
    raise IOError('Unable to import travel time data. Check formatting of TravelTimes.csv.')
try:
    dist_df=pd.read_csv('TravelDistances.csv',header=0,index_col=0)
    dist_df.columns=aux.strip(dist_df.columns)
    dist_df.index=aux.strip(dist_df.index)
except:
    raise IOError('Unable to import distance data. Check formatting of TravelDistances.csv.')
#%% # check to see which teams have games in the schedule
teams,dist_df,time_df=aux.checkTeamList(fullsched_df,dist_df,time_df)

#%% get dataframe containing schedule dataframes for each team
team_dfs=aux.get_team_schedule_dataframes(fullsched_df,teams,params,time_df,dist_df)

#%% get summary stats
summary_df=aux.get_summary_stats_df(teams,team_dfs,dist_df)
#%%  view bar charts summarizing the schedule results   
#aux.plot_results_summaries(summary_df) # uncomment to view temporary results
#%% Generate output file
resname=aux.generate_results(sched_filename,summary_df,team_dfs,params,teams)
print("Results saved to {}".format(resname))