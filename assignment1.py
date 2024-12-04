import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
from scipy.stats import skew

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def calc_stats(group):
    return pd.DataFrame({
        'mean': group[features].mean(),
        'variance': group[features].var()
    })


df = pd.read_csv('project10.csv')

post_features = ['post_score', 'post_upvote_ratio', 'post_thumbs_ups', 'post_total_awards_received']
comment_features = ['score', 'controversiality', 'ups', 'downs']
user_features = ['user_awardee_karma', 'user_awarder_karma', 'user_link_karma', 'user_comment_karma', 'user_total_karma']

is_post_level = all(feature in df.columns for feature in post_features)
if is_post_level:
    features = post_features + user_features
else:
    features = comment_features + user_features

stats = df.groupby('subreddit').apply(calc_stats).reset_index()
stats = stats.pivot(index='level_1', columns='subreddit', values=['mean', 'variance'])

subreddits = stats.columns.get_level_values(1).unique()
new_order = [(stat, subreddit) for subreddit in subreddits for stat in ['mean', 'variance']]
stats = stats.reindex(columns=new_order)

stats = stats.round(2)
stats.index.name = 'Feature'
stats.columns.names = ['Statistic', 'Subreddit']

stats.to_csv('subreddit_statistics.csv')


def is_skewed(data):
    return abs(stats.skew(data)) > 1

# Function to determine skewness
def get_skewness(data):
    skewness = skew(data)
    if abs(skewness) < 0.5:
        return "approximately symmetrical"
    elif skewness < 0:
        return "left-skewed"
    else:
        return "right-skewed"

# Function to plot distributions and print comments
def plot_and_comment(feature):
    plt.figure(figsize=(12, 6))
    for subreddit in df['subreddit'].unique():
        data = df[df['subreddit'] == subreddit][feature]
        sns.histplot(data, kde=True, label=subreddit)
        
        # Calculate and print statistics
        mean = data.mean()
        median = data.median()
        skewness = get_skewness(data)
        print(f"\n{subreddit} - {feature}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Distribution: {skewness}")
        print(f"  Range: {data.min():.2f} to {data.max():.2f}")
        
    plt.title(f'Distribution of {feature} by Subreddit')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{feature}_distribution.png')
    plt.close()

# Plot distributions for each feature
for feature in features:
    plot_and_comment(feature)
    
    
 # Function to determine skewness
def is_skewed(data):
    return abs(data.skew()) > 1

# Function to perform statistical test
def perform_test(feature):
    subreddit1, subreddit2 = df['subreddit'].unique()
    data1 = df[df['subreddit'] == subreddit1][feature]
    data2 = df[df['subreddit'] == subreddit2][feature]
    
    # Check for skewness
    if is_skewed(data1) or is_skewed(data2):
        # If skewed, use Mann-Whitney U test
        statistic, p_value = mannwhitneyu(data1, data2)
        test_name = "Mann-Whitney U test"
    else:
        # If not skewed, use t-test
        statistic, p_value = ttest_ind(data1, data2)
        test_name = "T-test"
    
    return test_name, statistic, p_value

# Perform tests for each feature
results = []
for feature in features:
    test_name, statistic, p_value = perform_test(feature)
    significant = "Yes" if p_value < 0.05 else "No"
    results.append({
        'Feature': feature,
        'Test': test_name,
        'Statistic': statistic,
        'P-value': p_value,
        'Significant at 5% level': significant
    })

# Create and display results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Feature')
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('statistical_test_results.csv')
print("\nResults have been saved to 'statistical_test_results.csv'")
significant_features = results_df[results_df['Significant at 5% level'] == 'Yes'].index

for feature in significant_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='subreddit', kde=True, element="step")
    plt.title(f'Distribution of {feature} by Subreddit')
    plt.savefig(f'{feature}_distribution.png')
    plt.close()
    
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('project10.csv')

# Convert post_created_time to datetime
df['post_created_time'] = pd.to_datetime(df['post_created_time'])

# Feature Engineering
def compute_aggregated_time_features(df):
    features = {}
    
    # 1. Hour of day distribution
    hour_dist = df['post_created_time'].dt.hour.value_counts(normalize=True).sort_index()
    features['peak_hour'] = hour_dist.idxmax()
    features['hour_entropy'] = stats.entropy(hour_dist)
    
    # 2. Day of week distribution
    dow_dist = df['post_created_time'].dt.dayofweek.value_counts(normalize=True).sort_index()
    features['peak_day'] = dow_dist.idxmax()
    features['day_entropy'] = stats.entropy(dow_dist)
    
    # 3. Month distribution
    month_dist = df['post_created_time'].dt.month.value_counts(normalize=True).sort_index()
    features['peak_month'] = month_dist.idxmax()
    features['month_entropy'] = stats.entropy(month_dist)
    
    # 4. Posting regularity
    time_diffs = df['post_created_time'].sort_values().diff().dt.total_seconds() / 3600  # in hours
    features['mean_time_between_posts'] = time_diffs.mean()
    features['std_time_between_posts'] = time_diffs.std()
    
    # 5. Temporal density
    time_range = (df['post_created_time'].max() - df['post_created_time'].min()).total_seconds() / 3600 / 24  # in days
    features['posts_per_day'] = len(df) / time_range if time_range > 0 else 0
    
    # 6. Burstiness
    if len(time_diffs) > 1:
        features['burstiness'] = (time_diffs.std() - time_diffs.mean()) / (time_diffs.std() + time_diffs.mean())
    else:
        features['burstiness'] = 0
    
    return pd.Series(features)

# Compute features for each subreddit
aggregated_features = df.groupby('subreddit').apply(compute_aggregated_time_features).reset_index()

# Function to test statistical significance
def test_significance(feature):
    data1 = aggregated_features.loc[aggregated_features['subreddit'] == aggregated_features['subreddit'].iloc[0], feature]
    data2 = aggregated_features.loc[aggregated_features['subreddit'] == aggregated_features['subreddit'].iloc[1], feature]
    
    # Since we have only one value per subreddit, we can't perform statistical tests
    # Instead, we'll calculate the absolute difference and relative difference
    abs_diff = abs(data1.iloc[0] - data2.iloc[0])
    rel_diff = abs_diff / ((data1.iloc[0] + data2.iloc[0]) / 2) if (data1.iloc[0] + data2.iloc[0]) != 0 else 0
    
    return abs_diff, rel_diff

# Test significance for each new feature
results = []

for feature in aggregated_features.columns[2:]:  # Skip 'subreddit' and 'level_1' columns
    abs_diff, rel_diff = test_significance(feature)
    results.append({
        'Feature': feature,
        'Absolute Difference': abs_diff,
        'Relative Difference': rel_diff
    })

# Create and display results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Feature')
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('time_feature_differences.csv')
print("\nResults have been saved to 'time_feature_differences.csv'")

# Visualize features
plt.figure(figsize=(12, 8))
sns.heatmap(aggregated_features.set_index('subreddit').iloc[:, 1:], annot=True, cmap='coolwarm', center=0)
plt.title('Aggregated Time Features by Subreddit')
plt.tight_layout()
plt.savefig('time_features_heatmap.png')
plt.close()

print("\nHeatmap of aggregated time features has been saved as 'time_features_heatmap.png'.")