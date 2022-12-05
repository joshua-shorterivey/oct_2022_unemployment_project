import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def spotlight_stats(df, feature, title=None, phase=None, orientation=None):

    ''' 
    Purpose:
        To create visuals and print statistics for the feature of the data set
    ---
    Parameters:
        df: dataframe containing features
        feature: the feature (column) to be used for testing and visualization
        phase: the phase of the pipeline for which the output is needed
    ---
    Output:
        prop_df: dataframe that contains population proportions and unemployment rate
    ---
    '''

    multi_col = pd.MultiIndex.from_tuples([('population_proportions', 'employed'), 
                                    ('population_proportions', 'unemployed'),
                                    ('population_proportions', 'change')])
    
    # dataframe, 3 columns, 
    prop_df = pd.DataFrame(columns=multi_col)
    prop_df['unemployment_rate'] = round(1 - df.groupby(by=feature).employed.mean().sort_values(ascending=True), 2)

    # i want to show the proportion of the population that each categorical option is
    employed_pop_proportion = df[df.employed == 1][feature].value_counts(normalize=True) 

    # i want to show the proportion of the population that each categorical option is
    unemployed_pop_proportion = df[df.employed == 0][feature].value_counts(normalize=True) 
    
    #assign proper values to dframe
    prop_df[('population_proportions', 'employed')] = employed_pop_proportion
    prop_df[('population_proportions', 'unemployed')] = unemployed_pop_proportion
    prop_df[('population_proportions', 'change')] = employed_pop_proportion - unemployed_pop_proportion

    #chi2 test
    alpha = .05
    crosstab = pd.crosstab(df[feature], df["employed"])

    chi2, p, dof, expected = chi2_contingency(crosstab)

    #prints crosstab only if phase of project is explore. during model phase just plots graph
    if phase == 'explore':
        print('Crosstab\n')
        print(crosstab.values)
        print('---\nExpected\n')
        print(f'{expected.astype(int)}')
        print('---\n')

    print(f'chi^2: {chi2:.4f}')
    print(f'p: {p:.4f}')
    print(f'degrees of freedom: {dof}')

    if p < alpha :
        print('Reject null hypothesis')
    else: 
        print('Fail to reject null hypothesis')

    #plots the distributions of the feature in separate columns for employed vs unemployed
    plt.figure(figsize=(20,6))
    if orientation =='h':
        g = sns.catplot(data=df, y=feature, col='employed', kind='count', sharex=False)
    else:
        g = sns.catplot(data=df, x=feature, col='employed', kind='count', sharey=False)
    plt.suptitle(f'Spotlight: {title}', y=1.02)
    plt.show()

    return round(prop_df, 3)