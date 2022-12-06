import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# exported to model.py

def spotlight_stats(df, feature, title=None, phase=None):

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
    prop_df['unemployment_rate'] = round(1 - df.groupby(by=feature).employed.mean().sort_values(ascending=True), 4)

    # i want to show the proportion of the population that each categorical option is
    employed_pop_proportion = df[df.employed == 1][feature].value_counts(normalize=True).sort_index() 

    # i want to show the proportion of the population that each categorical option is
    unemployed_pop_proportion = df[df.employed == 0][feature].value_counts(normalize=True).sort_index()
    
    #assign proper values to dframe
    prop_df[('population_proportions', 'employed')] = employed_pop_proportion
    prop_df[('population_proportions', 'unemployed')] = unemployed_pop_proportion
    prop_df[('population_proportions', 'change')] = unemployed_pop_proportion - employed_pop_proportion

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

    prop_df['unemployment_rate'].plot(kind='barh', title='Unemployment Rate')
    #pie chart creation
    fig, axs = plt.subplots(1,2, figsize=(12,8))
    labels1 = employed_pop_proportion.index
    labels2 = unemployed_pop_proportion.index
    colors = dict(zip(labels1, plt.cm.tab20.colors[:len(labels1)]))

    axs[0].pie(employed_pop_proportion, autopct='%1.1f%%', labels=labels1, colors=[colors[key] for key in labels1])
    axs[1].pie(unemployed_pop_proportion, autopct='%1.1f%%', labels=labels2, colors=[colors[key] for key in labels1])
    plt.tight_layout()
    axs[0].set_title('Employed Proportions')
    axs[1].set_title('Unemployed Proportions')
    plt.suptitle(f'Spotlight: {title}', y=.9)
    plt.show()

    return round(prop_df, 3)