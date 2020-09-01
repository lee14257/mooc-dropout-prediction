import pandas as pd

feature_event = pd.read_csv("dataset/features/event_features_train.csv")
feature_days = pd.read_csv("dataset/features/daycount_feature_train.csv")

# merge the two feature vectors into final feature vector
feature_vector = pd.merge(feature_event, feature_days, on="enrollment_id")
feature_vector.to_csv('dataset/features/combined_feature_train.csv', index=False)