# coding: utf-8
#import libraries
import pandas as pd
import math
import time
import numpy as np

#main

train_log = pd.read_csv("dataset/train/log_train.csv")
course_date = pd.read_csv("dataset/date.csv")
object_data = pd.read_csv("dataset/object.csv")

def writetotxt(towrite):
    # write the data for the text file
    towrite.to_csv('table4.txt', index=True, sep=',', mode='w+', header=True)
    pass


#making event count table
def main() :


    # Initial data for span days and number days
    cols_name = []
    cols_name.append('enrollment_id')
    cols_name.append("lastlog-firstlog")
    cols_name.append("study days")
    cols_name.append("early events")
    cols_name.append("mid events")
    cols_name.append("late events")
    id_list = train_log.enrollment_id.unique() ########################



    # Initial data for events near deadline
    train_log_count = pd.read_csv("dataset/train/log_train.csv", usecols =["enrollment_id", "time", "object"])
    train_log_count.columns = ["enrollment_id", "log_time", "module_id"]
    train_log_count.rename(columns={"object":"module_id", "time":"log_time"})

    course_date_count = pd.read_csv("dataset/date.csv", usecols =["course_id", "from", "to"])
    course_date_count.columns = ["course_id", "course_start", "course_end"]
    course_date_count.rename(columns={"from":"course_start", "to":"course_end"})

    object_data_count = pd.read_csv("dataset/object.csv", usecols =["course_id", "module_id", "start"])
    object_data_count.columns = ["course_id", "module_id", "module_start"]
    object_data_count.rename(columns={"start":"module_start"})

    when_table = pd.merge(course_date_count, object_data_count, on="course_id")
    when_table=pd.merge(when_table, train_log_count, on="module_id")

    # reorder columns and sort rows
    columnsTitles = ['enrollment_id', 'course_id', 'course_start', 'course_end', 'module_id', "module_start", 'log_time']
    when_table = when_table.reindex(columns=columnsTitles)
    when_table = when_table.sort_values(by=['enrollment_id', 'log_time'])

    ##########
    ind = 0
    when_table.fillna(0, inplace=True)

    corrupt_ids = []

    start_timing = time.time()

    # delete erroneous record with log_time < module_start_time
    for index, row in when_table.iterrows():
        if (row[5] != 0) & (row[5] > row[6]):
            if row[0] not in corrupt_ids:
                corrupt_ids.append(row[0])
            when_table.drop(when_table.index[ind])
        ind += 1

#    when_table=  when_table.set_index('enrollment_id')
#    when_table.to_csv('dataset/features/when_table.csv', index=False)
#    print when_table.columns

    print corrupt_ids

    ev = []
    early=0
    mid=0
    late=0
    for id in id_list:
    	# getting the span of days	#################################### col 1
    	trained = train_log[train_log['enrollment_id'] == id]['time']
    	event_counts = pd.to_datetime(trained)
    	span_dates = (event_counts.max()-event_counts.min()).days
    	# number of unique days 	#################################### col 2
    	dates_number = event_counts.dt.normalize().nunique()



    	# number of events in early #################################### col 3
    	start_time= when_table[when_table['enrollment_id']==id]['course_start']
    	end_time= when_table[when_table['enrollment_id']==id]['course_end']

    	# for i in range(0, 2):
    	for i in range(0, len(when_table[when_table['enrollment_id']==id])):
    		# print(when_table[when_table['enrollment_id']==id].iloc[i,:])
    		start = pd.to_datetime(when_table[when_table['enrollment_id']==id].iloc[i,:]['course_start'])
    		end = pd.to_datetime(when_table[when_table['enrollment_id']==id].iloc[i,:]['course_end'])
    		current = pd.to_datetime(when_table[when_table['enrollment_id']==id].iloc[i,:]['log_time'])

    		if current <= start +pd.to_timedelta("10day") :
    			early+=1
    		elif current >= end -pd.to_timedelta("10day") :
    			late+=1
    		else :
    			mid+=1

    	temp = [id, span_dates, dates_number, early,  mid,  late]
    	ev.append(temp)
    	early=0
    	mid=0
    	late=0

    # expected table at the end : enrollment_id, time, course, start, end
    # merge of log_train, object, date

    end_timing = time.time()
    print end_timing - start_timing

    cols_df = pd.DataFrame(ev,columns=cols_name)
    cols_df.fillna(0, inplace=True)
    cols_df.to_csv('dataset/features/daycount_feature_train.csv', index=False)

if __name__ == "__main__":
    main()

