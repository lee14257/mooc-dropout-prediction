# coding: utf-8

# In[149]:


#import libraries
import pandas as pd

#main
#train_truth = pd.read_csv("dataset/train/truth_train.csv", names = ["enrollment_id", "truth"])
train_enrollment = pd.read_csv("dataset/test/enrollment_test.csv")
train_log = pd.read_csv("dataset/test/log_test.csv")
course_date = pd.read_csv("dataset/date.csv")
object_data = pd.read_csv("dataset/object.csv")


# In[193]:


event_list = []
event_list.append('enrollment_id')
event_list.extend(train_log.event.unique())
id_list = train_log.enrollment_id.unique()
print event_list
print id_list


# In[201]:


#making event count table
eventcount_df = pd.DataFrame(columns=event_list)
event_counts = train_log.groupby('enrollment_id')['event'].value_counts()
ev = []
for id in id_list:
    navigate_count = event_counts[id].get('navigate')
    access_count = event_counts[id].get('access')
    problem_count = event_counts[id].get('problem')
    page_close_count = event_counts[id].get('page_close')
    video_count = event_counts[id].get('video')
    discussion_count = event_counts[id].get('discussion')
    wiki_count = event_counts[id].get('wiki')
    temp = [id, navigate_count, access_count, problem_count, page_close_count, video_count, discussion_count, wiki_count]
    ev.append(temp)



# In[219]:


event_df = pd.DataFrame(ev,columns=event_list)
# remove id
event_df.drop(event_df.columns[0], axis=1, inplace=True)
event_df.fillna(0, inplace=True)
event_df.to_csv('dataset/features/eventcount_feature_test_wo_id.csv', index=False)

