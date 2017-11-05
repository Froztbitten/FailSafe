import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
#from pandas_ml import ConfusionMatrix

studentDF = pd.read_csv("starfish_data.csv")

# Remove date values
studentDF.drop('start_dt', axis=1, inplace=True)
studentDF.drop('end_dt', axis=1, inplace=True)
studentDF.drop('next_start_dt', axis=1, inplace=True)
studentDF.drop('program_start_date', axis=1, inplace=True)
studentDF.drop('program_end_date', axis=1, inplace=True)

# Remove binary values
#studentDF.drop('final_term', axis=1, inplace=True)
studentDF.drop('on_campus', axis=1, inplace=True)

# Remove other categorical data
studentDF.drop('credential', axis=1, inplace=True)
studentDF.drop('full_part_time', axis=1, inplace=True)
studentDF.drop('term_start_month', axis=1, inplace=True)
studentDF.drop('prior_term_start_month', axis=1, inplace=True)
studentDF.drop('prior_full_part_time', axis=1, inplace=True)
studentDF.drop('primary_major', axis=1, inplace=True)
studentDF.drop('admitted_zip_code', axis=1, inplace=True)
studentDF.drop('unmet_need', axis=1, inplace=True)

# Remove useless data

studentDF['prior_term_earned_credits'].fillna(0, inplace=True)
studentDF['prior_total_credit_hours'].fillna(0, inplace=True)
studentDF['prior_term_fraction_completed_credits'].fillna(0, inplace=True)
studentDF['prior_term_gpa'].fillna(0, inplace=True)
studentDF['prior_cumulative_gpa'].fillna(0, inplace=True)

studentDF['gpa_change'] = studentDF['prior_term_gpa'] - studentDF['high_school_gpa']
studentDF.drop('prior_term_gpa', axis=1, inplace=True)


# Drop values that are mostly NaN
studentDF = pd.get_dummies(studentDF)
studentDF.dropna(thresh=0.90*len(studentDF), axis=1, inplace=True)

# For students with multiple entries, consolidate their entries and take a mean.
# studentDF = studentDF.groupby('person_party_id').mean().reset_index()

# Drop the id fields
studentDF.drop('person_party_id', axis=1, inplace=True)
studentDF.drop('cms_user_ext_id', axis=1, inplace=True)

# Normalize with 0-1.
sdf_norm = studentDF
sdf_copy = studentDF
sdf_norm = (studentDF - studentDF.min()) / (studentDF.max() - studentDF.min())

# Drop the na values
sdf_norm = sdf_norm.dropna(axis=0, how='any')
sdf_copy = sdf_norm.dropna(axis=0, how='any')

# Drop outcome for the entries
sdf_norm = sdf_norm.drop('outcome', axis=1)


# Create a list of number of clusters and their intertias
clusterNumList = list()
inertiaList = list()
i = 0

while(i < 10):
    i = i + 1
    kmeans = MiniBatchKMeans(init='k-means++',n_clusters=i,reassignment_ratio=0,batch_size=7954)
    kmeans.fit(sdf_norm)
    clusterNumList.append(i)
    inertiaList.append(kmeans.inertia_)

# Print the chart to find the optimal cluster #.
plt.plot()
plt.grid(True)
plt.xlabel("# of Clusters")
plt.ylabel("Inertia")
plt.plot(clusterNumList, inertiaList)


# USE VALUE OF 2 FOR K

kmeans = MiniBatchKMeans(init='k-means++',n_clusters=2,reassignment_ratio=0,batch_size=7954)

# Fit the data to the clusters
test=kmeans.fit(sdf_norm)
test1 = kmeans.predict(sdf_norm)

# Print the highest means for the values in each cluster

print(pd.DataFrame({'columns':sdf_norm.columns,'means':
    kmeans.cluster_centers_[0]}).sort_values('means', ascending = False))
print()
print(pd.DataFrame({'columns':sdf_norm.columns,'means':
    kmeans.cluster_centers_[1]}).sort_values('means', ascending = False))
print()

# Create the crosstab to see how accurate the preductions are
# second value is the x axis
target = (pd.crosstab(test1, sdf_copy['outcome']))
#cfm = ConfusionMatrix(test1, sdf_copy['outcome'])

# Get the numbers of each entry in the crosstab
falsePositives = target[0.0][1]
falseNegatives = target[1.0][0]
truePositives = target[1.0][1]
trueNegatives = target[0.0][0]

d = {'False Positives': falsePositives, 'False Negatives': falseNegatives, 'True Positives': truePositives,  'True Negatives': trueNegatives}
df = pd.DataFrame(data=d, index=[0])

# Make a bar chart to show the values
df.plot(kind='bar')

# Values for a manual test entry
# final_term, term_length, term_attempted_credits, prior_term_earned_credits, prior_total_credit_hours, high_school_gpa, prior_term_fraction_completed_credits, transfer, prior_cumulative_gpa, time_at_institution, prediction, gpa_change
'''
dtest = {'final_term': 0, 'term_length': 120, 'term_attempted_credits': 15, 'prior_term_earned_credits': 16, 'prior_total_credit_hours':58, 'high_school_gpa':3.8, 'prior_term_fraction_completed_credits':3.7, 'transfer':0, 'prior_cumulative_gpa':3.6, 'time_at_institution': 2, 'prediction':.75, 'gpa_change':0.2}
dftest = pd.DataFrame(data=dtest, index=[0])
dftest_norm = (dftest - studentDF.min()) / (studentDF.max() - studentDF.min())
dftest_norm.drop('outcome', axis=1, inplace=True)
kmeans.predict(dftest_norm)
'''
