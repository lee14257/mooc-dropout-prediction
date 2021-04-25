# MOOC Drop-out Prediction
### Problem Statement
> Students' high dropout rate on MOOC platforms has been heavily criticized, and predicting their likelihood of dropout would be useful for maintaining and encouraging students' learning activities

### Structure
1. *preprocessing_eventcount.py* 
   - Pre-processes  1st Feature Vector : Events Count 
   - Includes load data and produces Feature Table in CSV.

2. *preprocessing_daycount.py* 
   - Pre-processes  2nd Feature Vector : Day Count 
   - Includes load data and produces Feature Table in CSV.

3. *preprocessing_combined.py*  
   - Pre-processes -> Combines both Feature Vectors 
   - Reads directly from CSV’s produced in a. and b.
   - Produces final Feature Table in CSV.

4. *Rf.py* 
   - Classifier #1 -> Outputs a .txt file and ROC curves

5. *mlp.py* 
   - Classifier #2 -> Outputs a .txt file and ROC curves

6. *xgb.py* 
   - Classifier #3 -> Outputs a .txt file and ROC curves.

7. *hard_voting.py*  
   - Calls Ensemble method

8. *dataset (folder)* 
   - Contains all the data provided by XueTang.
   - Contains a "feature" folder that stores the CSV's produced in a, b, and c.

9. *outputs (folder)* 
   - Contains all .txt files produced in d, e, and f.
   - Contains the ROC curves that are to be manually saved as PNG's.		      	

### Understanding the Data
### Data-Based Hypotheses
### Pre-Processing
### Classifers
### Results
### Ensemble: Hard Voting
### Conclusion



