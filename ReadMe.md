# MOOC Drop-out Prediction
## 1. Problem Statement
> Students' high dropout rate on MOOC platforms has been heavily criticized, and predicting their likelihood of dropout would be useful for maintaining and encouraging students' learning activities

## 2. Code Structure
<details>
   <summary> Open </summary>
   
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

## 3. Understanding the Data
<details>
   <summary> Open </summary>
   
  - `Date.csv` : Gives us more information about the timespan of each course
  - `Object.csv` : Gives us more information about each module in a course 
  - `Enrollment_(train/test).csv` : User enrollment records
  - `Log_(train/test).csv` : Behavior Records
  - `True_train.csv` : Ground Truth about a dropout
   
</details>

## 4. Data-Based Hypotheses

>My first assumption was that any student who completes a course would have more counts for each event i.e. watching more course videos, solving problems, participating in discussions etc. 

>Besides events, there is also the information of whether the student accessed the course in a server or a browser. This information was useful in terms of how a browser is more frequently used to watch videos and other interactive events compared to a server. Thus, I hypothesized that a student who spends more time in a browser would have a lower chance of dropping out of a course as well.

## 5. Pre-Processing
## 6. Classifers
## 7. Results
## 8. Ensemble: Hard Voting
## 9. Conclusion



