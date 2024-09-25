import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path='A1\project10.csv'):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    datetime_columns = ['created_time', 'user_account_created_time', 'post_created_time']
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Combine post title and self-text
    df['post_text'] = df['post_title'] + ' ' + df['post_self_text'].fillna('')
    
    # Create additional features
    df['post_age'] = (df['created_time'] - df['post_created_time']).dt.total_seconds() / 3600  # age in hours
    df['user_account_age'] = (df['created_time'] - df['user_account_created_time']).dt.total_seconds() / (3600 * 24 * 365.25)  # age in years
    df['comment_to_post_score_ratio'] = df['score'] / df['post_score']
    
    # Assign labels based on subreddit
    df['label'] = df['subreddit'].apply(lambda x: 'action' if x in ['climateoffensive', 'climateactionplan'] else 'science')
    
    # Select features for modeling
    text_features = ['post_text', 'self_text']
    numeric_features = ['score', 'controversiality', 'ups', 'downs', 
                        'user_awardee_karma', 'user_awarder_karma', 'user_link_karma', 
                        'user_comment_karma', 'user_total_karma', 'post_score', 
                        'post_upvote_ratio', 'post_thumbs_ups', 'post_total_awards_received',
                        'post_age', 'user_account_age', 'comment_to_post_score_ratio']
    categorical_features = ['user_is_verified']
    target = 'label'
    
    # Handle categorical features
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_features + numeric_features + [col for col in df.columns if col.startswith('user_is_verified_')]],
        df[target], 
        test_size=0.2, 
        random_state=42,
        stratify=df[target]
    )
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train['post_text'] + ' ' + X_train['self_text'].fillna(''))
    X_test_tfidf = tfidf.transform(X_test['post_text'] + ' ' + X_test['self_text'].fillna(''))
    
    # Handle NaN and infinite values
    imputer = SimpleImputer(strategy='mean')
    print(df)
    X_train_numeric = imputer.fit_transform(X_train[numeric_features + [col for col in X_train.columns if col.startswith('user_is_verified_')]])
    X_test_numeric = imputer.transform(X_test[numeric_features + [col for col in X_test.columns if col.startswith('user_is_verified_')]])
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
    X_test_numeric_scaled = scaler.transform(X_test_numeric)
    
    # Combine TF-IDF and numeric features
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_numeric_scaled))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_numeric_scaled))
    
    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)
    
    return X_train_balanced, X_test_combined, y_train_balanced, y_test, tfidf, scaler, imputer, df

# Usage
X_train, X_test, y_train, y_test, tfidf_vectorizer, numeric_scaler, numeric_imputer, original_df = load_and_preprocess_data()

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Number of 'action' posts in training:", sum(y_train == 'action'))
print("Number of 'science' posts in training:", sum(y_train == 'science'))
print("Original dataframe shape:", original_df.shape)
print("Columns in original dataframe:", original_df.columns.tolist())