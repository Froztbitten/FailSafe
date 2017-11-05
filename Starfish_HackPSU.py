import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pyl
import itertools
from operator import itemgetter

pd.set_option('display.width', 500)

plt.style.use('ggplot')

df = pd.read_csv('Starfish_HackPSU_Data.csv')

dfNumeric = df.describe().transpose()
dfNumeric['Cardinality'] = df.apply(pd.Series.nunique)
dfNumeric['% Missing'] = df.isnull().sum() / df.count()
print('Numerical DataFrame')
print(dfNumeric)
print()

#corr = df.corr()
#sns.heatmap(corr)

dfCategoric = pd.DataFrame()
dfCategoric['final_term'] = df['final_term']
dfCategoric['credential'] = df['credential']
dfCategoric['full_part_time'] = df['full_part_time']
dfCategoric['on_campus'] = df['on_campus']
dfCategoric['prior_full_part_time'] = df['prior_full_part_time']
dfCategoric['transfer'] = df['transfer']
dfCategoricT = dfCategoric.describe().transpose()
dfCategoricT['% Missing'] = dfCategoric.isnull().sum() / dfCategoric.count()
print('Categorical DataFrame')
print(dfCategoricT)

dfItems = df
dfItems = dfItems.drop('prior_cumulative_gpa',axis=1)
dfItems['prior_cumulative_gpa'] = pd.cut(df['prior_cumulative_gpa'], bins=10)
dfItems = dfItems.drop('term_length', axis=1)
dfItems['term_length'] = pd.cut(df['term_length'], bins=[0,20,42,120])
dfItems = dfItems.drop('prior_cum_credits_not_completed', axis=1)
dfItems = dfItems.drop('term_attempted_credits', axis=1)
dfItems['term_attempted_credits'] = pd.cut(df['term_attempted_credits'], bins=10)
dfItems = dfItems.drop('delta_unmet_need', axis=1)
dfItems['delta_unmet_need'] = pd.cut(df['delta_unmet_need'], bins = 16)
dfItems = dfItems.drop('unmet_need', axis=1)
dfItems['unmet_need'] = pd.cut(df['unmet_need'], bins = 8)
dfItems = dfItems.drop('prior_term_earned_credits', axis=1)
dfItems['prior_term_earned_credits'] = pd.cut(df['prior_term_earned_credits'], bins = 11)
dfItems = dfItems.drop('prior_total_credit_hours', axis=1)
dfItems['prior_total_credit_hours'] = pd.cut(df['prior_total_credit_hours'], bins = 10)
dfItems = dfItems.drop('high_school_gpa', axis=1)
dfItems['high_school_gpa'] = pd.cut(df['high_school_gpa'], bins = 13)
dfItems = dfItems.drop('prior_term_gpa', axis=1)
dfItems['prior_term_gpa'] = pd.cut(df['prior_term_gpa'], bins = 10)
dfItems = dfItems.drop('full_part_time', axis=1)
dfItems = dfItems.drop('outcome', axis=1)
dfItems['outcome'] = pd.cut(df['outcome'], bins=2, labels=["fail","pass"])

basketList = []
l1TempDict = {}
l1Dict = {}
l2TempDict = {}
l2Dict = {}
l3Checked = {}
l3TempDict = {}
l3Dict = {}
l3Poss = []
count = 0

for i, r in dfItems.iterrows():
    basket = set()
    countIndex = 0
    for ele in r:
        basket.add(r.index[countIndex] + ":" + str(ele))
        countIndex += 1
    basketList.append(basket)
    count += 1
    
ratio = .5 * len(basketList)
# Count up all item frequencies
for customer in basketList:
    for item in customer:
        if item in l1TempDict:
            l1TempDict[item] += 1
        else:
            l1TempDict[item] = 1
# Trim all items not >= ratio to be considered frequent
for item in l1TempDict:
    if 'nan' in item:
        continue
    elif l1TempDict[item] >= ratio:
        l1Dict[item] = l1TempDict[item]
# Trim any customers that do not have any of these items
for customer in basketList:
    if any(items in l1Dict for items in customer) == False:
        basketList.remove(customer)
      
print()
print('Frequent Itemsets of Size 1')
l1Set = set()
for k, v in l1Dict.items():
    l1Set.add((v, k))
l1Sorted = sorted(l1Set, key=itemgetter(0), reverse=True)

for x in l1Sorted:
    print(x[0],x[1])
        

count = 0
# Iterate through all pairs of frequent items
for pair in itertools.combinations(l1Dict, 2):
    # count += 1
    # if(count % 10000 == 0):
    #     print( float(count) / 92235)
    # Apriori's Algorithm
    if(pair[0] in l1Dict and pair[1] in l1Dict):
        # Search customers using pairs
        for customer in basketList:
            if pair[0] in customer and pair[1] in customer:
                if pair in l2TempDict:
                    l2TempDict[pair] += 1
                else:
                    l2TempDict[pair] = 1
# Trim all items >= ratio to be considered frequent
for key in l2TempDict:
    if l2TempDict[key] >= ratio:
        l2Dict[key] = l2TempDict[key]
  
print()
print('Frequent Itemsets of Size 2')
l2Set = set()
for k, v in l2Dict.items():
    l2Set.add((v, k[0], k[1]))
l2Sorted = sorted(l2Set, key=itemgetter(0), reverse=True)

for x in l2Sorted:
    print(x[0],x[1].rjust(41),x[2].rjust(50))
        


count = 0
l3Checked = {}
# Iterate through all pairs of frequent items
for pair in itertools.combinations(l2Dict, 2):
    # count += 1
    # if(count % 100000 == 0):
    #     print(float(count) / 96278626)
    t = tuple(sorted(set(pair[0] + pair[1])))
    # Apriori's Algorithm
    if(t[0],t[1] in l2Dict and t[1],t[2] in l2Dict and t[0],t[2] in l2Dict):
        # Make sure combination is size 3 and not already in the hashmap
        if(len(t) == 3 and t not in l3Checked):
            l3Checked[t] = 1
            # Search customers using pairs
            for customer in basketList:
                if t[0] in customer and t[1] in customer and t[2] in customer:
                    if t in l3TempDict:
                        l3TempDict[t] += 1
                    else:
                        l3TempDict[t] = 1
# Trim all items >= ratio to be considered frequent
for key in l3TempDict:
    if l3TempDict[key] >= ratio:
        l3Dict[key] = l3TempDict[key]
        
print()
print('Frequent Itemsets of Size 3')
l3Set = set()
for k, v in l3Dict.items():
    l3Set.add((v, k[0], k[1], k[2]))
l3Sorted = sorted(l3Set, key=itemgetter(0), reverse=True)

for x in l3Sorted:
    print(x[0],x[1].rjust(41),x[2].rjust(50),x[3].rjust(50))



confDict = {}
mostInt = (0, 0, 0)
leastInt = (0, 0, 1)

# Iterate through all antecedent, consequent pairs
for i in l1Dict:
    for j in l1Dict:
        if (i,j) in l2Dict:
            confidence = float(l2Dict[i,j]) / float(l1Dict[i])
            # If confidence is >= 85% store in confDict
            if (confidence >= .85):
                confDict[(i, j)] = confidence

            confidence = float(l2Dict[i, j]) / float(l1Dict[j])
            # If confidence is >= 85% store in confDict
            if (confidence >= .85):
                confDict[(j, i)] = confidence

intDict = []
# For all confident rules, print with interest value
for pair in confDict:
    interest = confDict[(pair[0],pair[1])] - (float(l1Dict[pair[1]]) / float(7513))
    intDict.append([pair[0],pair[1],interest])

answer = sorted(intDict, key=itemgetter(2))
#for x in answer:
#    print(x[2])
#    if(abs(x[2]) <= 0.05):
#        answer.remove(x if )
#    elif "outcome" not in x[1]:
#        answer.remove(x)
answer = [x for x in answer if abs(x[2]) >= 0.05]
print()
print('Interesting Rules')
for x in answer:
    print('{0:.2f}'.format(x[2]).rjust(6) + '%',('If ' + str(x[0])).rjust(45),'then ' + x[1])


#for attr in dfCategoric:
#    plt.figure()
#    dfCategoric[attr].value_counts().plot(kind='bar')
#    plt.title(attr)
#    plt.savefig(attr)
#    plt.show()
#    plt.close()
#
#for attr in df.select_dtypes(include=['int64', 'float64']):
#    plt.figure()
#    df[attr].hist(bins=20)
#    plt.title(attr)
#    plt.savefig(attr)
#    plt.show()
#    plt.close()
#
#    if df[attr].quantile(q=0.75) - df[attr].quantile(q=0.25) != 0:
#        plt.figure()
#        df.boxplot(column=attr)
#        plt.savefig(attr)
#        plt.show()
#        plt.close()