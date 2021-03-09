I. Proj_4472_T1_Group6_report.pdf

II.Proj_4472_T1_Group6_code.zip 

   a. preprocessing_eventcount.py ——> Pre-processes  1st Feature Vector : Events Count 
				  ——> includes load data and produces Feature Table in CSV.

   b. preprocessing_daycount.py ————> Pre-processes  2nd Feature Vector : Day Count 
			        ————> includes load data and produces Feature Table in CSV.

   c. preprocessing_combined.py ————> Pre-processes -> Combines both Feature Vectors 
			        ————> reads directly from CSV’s produced in a. and b.
				————> produces final Feature Table in CSV.

   d. Rf.py ————————————————————————> Classifier #1 -> Outputs a .txt file and ROC curves

   e. mlp.py ———————————————————————> Classifier #2 -> Outputs a .txt file and ROC curves

   f. xgb.py ———————————————————————> Classifier #3 -> Outputs a .txt file and ROC curves.

   g. hard_voting.py ———————————————> Calls Ensemble method

   h. dataset (folder) —————————————> Contains all the data provided by XueTang.
		       —————————————> Contains a "feature" folder that stores the CSV's produced in a, b, and c.

   i. outputs (folder) —————————————> Contains all .txt files produced in d, e, and f.
		       —————————————> Contains the ROC curves that are to be manually saved as PNG's.		      	

III. ReadMe.txt


######################################################################################################
#  Note: I ran .py files in a, b, and c again to create the corresponding tables for the test data  #
#  by changing the file directories and output file names manually				     #
######################################################################################################



