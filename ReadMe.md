# MOOC Drop-out Prediction
## Problem Statement
> Students' high dropout rate on MOOC platforms has been heavily criticized, and predicting their likelihood of dropout would be useful for maintaining and encouraging students' learning activities

## Code Structure
<details>
   
#### *preprocessing_eventcount.py* 
    - Pre-processes  1st Feature Vector : Events Count 
    - Includes load data and produces Feature Table in CSV.

####  *preprocessing_daycount.py* 
    - Pre-processes  2nd Feature Vector : Day Count 
    - Includes load data and produces Feature Table in CSV.

#### *preprocessing_combined.py*  
    - Pre-processes & Combines both Feature Vectors 
    - Reads directly from CSV’s produced in a. and b.
    - Produces final Feature Table in CSV.

#### *Rf.py* 
    - Classifier #1 -> Outputs a .txt file and ROC curves

#### *mlp.py* 
    - Classifier #2 -> Outputs a .txt file and ROC curves

#### *xgb.py* 
    - Classifier #3 -> Outputs a .txt file and ROC curves.

#### *hard_voting.py*  
    - Calls Ensemble method

#### *dataset (folder)* 
    - Contains all the data provided by XueTang.
    - Contains a "feature" folder that stores the CSV's produced in a, b, and c.

#### *outputs (folder)* 
    - Contains all .txt files produced in d, e, and f.
    - Contains the ROC curves that are to be manually saved as PNG's.		      	
</details>

## Understanding the Data
<details>
   <summary> Read </summary>
   
  - `Date.csv` : Gives us more information about the timespan of each course
  - `Object.csv` : Gives us more information about each module in a course 
  - `Enrollment_(train/test).csv` : User enrollment records
  - `Log_(train/test).csv` : Behavior Records
  - `True_train.csv` : Ground Truth about a dropout
   
</details>

## Data-Based Hypotheses
<details>
   <summary> Read </summary>
      
>Any student who completes a course would have more counts for each event i.e. watching more course videos, solving problems, participating in discussions etc. 

>Besides events, there is also the information of whether the student accessed the course in a server or a browser. This information was useful in terms of how a browser is more frequently used to watch videos and other interactive events compared to a server. Thus, I hypothesized that a student who spends more time in a browser would have a lower chance of dropping out of a course as well.
   
>Finally, the greater the number of active days (days where the student had events on a particular enrollment), the smaller the chance of a dropout, and vice versa.
   
![upload_1](https://user-images.githubusercontent.com/15091955/128627874-dc1e87a4-f0c6-4326-b40b-a3bbc83af263.png)
   

</details>

## Pre-Processing & Feature Engineering
<details>
   <summary> Read </summary>

   ### Feature Vector 1: Event Counts
  * In order to build this feature set, the data was grouped by the *enrollmend_id*. And then for each *enrollment_id*, each of the event-category counts were added.
   
   | Index  | Variable |
   | ------------- | ------------- |
   | 1  | # of problems solved  |
   | 2  | # of videos watched  |
   | 3  | # of access  |
   | 4  | # of wiki  |
   | 5  | # of discussions  |
   | 6  | # of navigations  |
   | 7  | # of page close  |
   | 8  | # of server connections  |
   | 9  | # of browser connections  |
   
   ### Feature Vector 2: Time Series & Day Counts
   ##### Handling Corrupt Data
  * Before engineering the next feature vector, I noticed that the *Enrollment Log* showed that students accessed some modules before they were even posted i.e. `enrollment_log` time stamp was before the start-date of a course module in `object.csv`. 
  * In order to solve this, I deleted those log instances so as to ensure that it did not affect the calculation of active days. It turned out that roughly 2% of the Enrollment ID’s (1570 out of 72395) had such corrupt log instances.
   
   ##### Building the Vector
   
   | Index  | Variable |
   | ------------- | ------------- |
   | 1  | last log time - first log time  |
   | 2  | # of effective study days  |
   | 3  | # events in start time-period of course   |
   | 4  | # events in middle time-period of course  |
   | 5  | # events in end time-period of course  |

  * The dataset was grouped by enrollment_id, then converted the time element into Dates after which it was easy to find the first and last date of the event for a specific user.
  * The courses' duration was divided into 3 equal time-spans: beginning, middle, and end. This information was useful to determine key user behaviors (eg. users that had some log record during the end period of a course were more likely to complete the course without a dropout, etc.)
   
   > Both feature vectors proved to give useful insight into the users' behaviors of dropout. As a result, the two vectors were combined to form a final feature vector with 14 Variables.
   
</details>

## Classifers
## Results
## Ensemble: Hard Voting
## Conclusion



