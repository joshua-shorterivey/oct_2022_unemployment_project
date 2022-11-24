import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def acquire_oct():
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''

    df = pd.read_csv('oct22pub.csv')

    df.columns = df.columns.str.lower()
    
    return df

# create function to flatten race in less categories
def flatten_housing_type(val):
    if val == 1:
        val = 'perm_house_apt'
    elif 5 <= val <= 6:
        val = 'mobile_home'
    else:
        val = 'other'
    return val

def flatten_household_type(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 0:
        val = 'unknown'
    elif 1<= val <= 2:
        val = 'married'
    elif 3 <= val <= 8:
        val = 'unmarried'
    else:
        val = 'group'
    return val


def flatten_marital(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val in [1,2,5]:
        val = 'married'
    else:
        val = 'single'
    return val

def flatten_education(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val <= 38:
        val = 'no_high_school'
    elif val <= 40:
        val = 'high_school_ged'
    elif val < 43:
        val = 'associates'
    elif val == 43:
        val = 'bachelor'
    elif val > 43:
        val = 'post_grad'
    return val

# create function to flatten race in less categories
def flatten_race(val):
    if val == 1:
        val = 'white'
    elif val == 2:
        val = 'black'
    elif val == 3:
        val = 'AI/NA'
    elif val == 4:
        val = 'asian'
    elif val == 5:
        val = 'HI/PI'
    elif val in [6,7,8,9,16,17,18,19,20,21,23,24]:
        val = 'mixed_white'
    else :
        val = 'mixed_other'
    return val


def flatten_citizenship(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val <= 3:
        val = 'native'
    elif val == 4 :
        val = 'naturalized'
    else :
        val = 'foreign'

    return val


def flatten_immigration(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 57:
        val = 'us_50'
    elif val < 100:
        val = 'us_territories'
    else:
        val = 'foreign_country'
    return val

def flatten_immigration_era(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val <= 0:
        val = 'native'
    elif val < 17:
        val = 'pre_gwot'
    else:
        val = 'post_pwot'
    return val

def flatten_service_era(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 1:
        val = 'post_gwot'
    elif val > 1:
        val = 'pre_gwot'
    else:
        val = 'n/a'
    return val

def prep_columns(df): 
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''

    columns_to_keep = [ 'hehousut',
                        'hrnumhou', 
                        'hefaminc',
                        'hrhtype',
                        'hubus',
                        'gediv',
                        'gestfips',
                        'gtmetsta',
                        'gtcbsasz',
                        'prtage',
                        'pemaritl',
                        'pesex',
                        'peafever',
                        'peeduca',
                        'ptdtrace', 
                        'pehspnon',
                        'penatvty',
                        'pemntvty',
                        'pefntvty', 
                        'prcitshp', 
                        'prinuyer',
                        'prempnot',
                        'prftlf',
                        'prcow1',
                        'prmjind1',
                        'prmjocc1', 
                        'peschenr', 
                        'prnmchld',
                        'peafwhn1',
                        'pecert1',\
                        
            ]

    df = df[columns_to_keep]

    df = df.rename(columns={'hehousut': 'housing_type',
        'hrnumhou': 'household_num',
        'hefaminc': 'family_income',
        'hrhtype': 'household_type',
        'hubus': 'own_bus_or_farm',
        'gediv': 'country_region',
        'gestfips': 'state',
        'gtmetsta': 'metropolitan',
        'gtcbsasz': 'metro_area_size',
        'prtage': 'age',
        'pemaritl': 'marital_status',
        'pesex': 'is_male',
        'peafever': 'veteran',
        'peeduca': 'education',
        'ptdtrace': 'race',
        'pehspnon': 'hispanic_or_non',
        'penatvty': 'birth_country',
        'pemntvty': 'mother_birth_country',
        'pefntvty': 'father_birth_country',
        'prcitshp': 'citizenship',        
        'prinuyer': 'immigration_era',
        'prempnot': 'employed',
        'prftlf': 'ft_or_pt',
        'prcow1': 'worker_class',
        'prmjind1': 'industry',
        'prmjocc1': 'occupation',
        'peschenr': 'enrolled_in_school',
        'prnmchld': 'num_children',
        'peafwhn1': 'service_era',
        'pecert1': 'professional_certification'})                               

    return df
def generate_mappings():
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    region_map = {1: 'NEW ENGLAND', 2: 'MIDDLE ATLANTIC', 3: 'EAST NORTH CENTRAL', 4: 'WEST NORTH CENTRAL',
            5: 'SOUTH ATLANTIC',6: 'EAST SOUTH CENTRAL',7: 'WEST SOUTH CENTRAL', 8: 'MOUNTAIN',  9: 'PACIFIC' }

    state_map = {1:'AL',2:'AK',4:'AZ',5:'AR', 6:'CA',8:'CO', 9:'CT', 10:'DE',
                11:'DC', 12:'FL',13:'GA',15:'HI',16:'ID',17:'IL',18:'IN',
                19:'IA',20:'KS',21:'KY',22:'LA',23:'ME',24:'MD',25:'MA',26:'MI',
                27:'MN',28:'MS',29:'MO',30:'MT',31:'NE',32:'NV',33:'NH',34:'NJ',
                35:'NM',36:'NY',37:'NC',38:'ND',39:'OH',40:'OK',41:'OR',42:'PA',
                44:'RI',45:'SC',46:'SD',47:'TN',48:'TX',49:'UT',50:'VT',51:'VA',
                53:'WA',54:'WV',55 :'WI',56:'WY'}

    metropolitan_map = {1:'metro', 2:'non_metro',3:'not_identified'}

    ft_pt_map = {1: 'full_time', 2: 'part_time'}

    worker_class_map = {1:'FEDERAL GOVT', 2:'STATE GOVT', 3:'LOCAL GOVT',
                        4:'PRIVATE',5: 'SELF-EMPLOYED, UNINCORP.', 6: 'WITHOUT PAY'}

    industry_map = {1:'Agriculture, forestry, fishing, and hunting',2:	'Mining',3:	'Construction',										 
                    4:	'Manufacturing', 5:	'Wholesale and retail trade', 6:'Transportation and utilities',					      
                    7:	'Information', 8:'Financial activities', 9:'Professional and business services',					
                    10:	'Educational and health services', 11:'Leisure and hospitality',12:'Other services',									
                    13:'Public administration', 14:'Armed Forces'}

    occupation_map = {1:'Management, business, and financial occupations',
                    2:	'Professional and related occupations',					
                    3:	'Service occupations',				
                    4:	'Sales and related occupations',					
                    5:	'Office and administrative support occupations',
                    6:	'Farming, fishing, and forestry occupations',			
                    7:	'Construction and extraction occupations',				
                    8:	'Installation, maintenance, and repair occupations',	
                    9:	'Production occupations',						
                    10:	'Transportation and material moving occupations',	
                    11:	'Armed Forces'}

    return region_map, state_map, metropolitan_map, ft_pt_map, worker_class_map, industry_map, occupation_map

def prep_values(df):
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''

    #filters employed/unemployed from those not in labor force or without response
    df = df[(df.employed == 1) | (df.employed == 2)].reset_index(drop=True)

    ### imputing values ###
    df['own_bus_or_farm'] = df['own_bus_or_farm'].apply(lambda x: x == 1, 1,0)
    df['veteran'] = df['veteran'].apply(lambda x: x == 1, 1,0)
    df['service_era'] = df.service_era.apply(lambda x: 0 if x == -1 else x)
    df.enrolled_in_school = df.enrolled_in_school.apply(lambda x: np.nan if x == -1 else x)
    df[['worker_class', 'industry','occupation']] = df[['worker_class', 'industry','occupation']].applymap(lambda x: np.nan if x == -1 else x)

    imputer = KNNImputer(n_neighbors=1)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    ### flatten values ###
    df.housing_type = df.housing_type.apply(flatten_housing_type)
    df.household_type = df.household_type.apply(flatten_household_type)
    df.marital_status = df.marital_status.apply(flatten_marital)
    df['race'] = df['race'].apply(flatten_race)
    df.education = df.education.apply(flatten_education)
    df.citizenship = df.citizenship.apply(flatten_citizenship)
    df.birth_country = df.birth_country.apply(flatten_immigration)
    df.mother_birth_country = df.mother_birth_country.apply(flatten_immigration)
    df.father_birth_country = df.father_birth_country.apply(flatten_immigration)
    df.immigration_era = df.immigration_era.apply(flatten_immigration_era)
    df.service_era = df.service_era.apply(flatten_service_era)

    ### generate and use value mappings ###
    (region_map, state_map, metropolitan_map, ft_pt_map,
     worker_class_map, industry_map, occupation_map) = generate_mappings()
    df.country_region = df.country_region.map(region_map)
    df.state = df.state.map(state_map)
    df.metropolitan = df.metropolitan.map(metropolitan_map)
    df.ft_or_pt = df.ft_or_pt.map(ft_pt_map)
    df.worker_class = df.worker_class.map(worker_class_map)
    df.industry = df.industry.map(industry_map)
    df.occupation = df.occupation.map(occupation_map)

    #fixing binary columns freom -1, 1, 2 --> 0, 1
    binary_cols = ['own_bus_or_farm', 'is_male', 'veteran','hispanic_or_non', 'employed',
                'enrolled_in_school','professional_certification']

    # changed all non-affirmative answers to negative. school, most affected
    for col in binary_cols:
        df[col] = np.where(df[col] == 1, 1, 0)

    return df

def wrangle_oct():
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''

    df = acquire_oct()

    df = prep_columns(df)

    df = prep_values(df)

    return df

def split_scale(df, dummy='n'):
    ''' 
    Purpose:
        To split and scale the input dataframe
    ---
    Parameters:
        df: a tidy dataframe
        dummy: ignored
    ---
    Output:
        train: unscaled subset of dataframe for exploration and model training
        validate: unscaled and unseen data for model testing
        test: unscaled and unseen data for final model test
        train_scaled: scaled subset of dataframe for exploration and model training
        validate_scaled: scaled and unseen data for model testing
        test_scaled: scaled and unseen data for model testing
    ---
    '''
    #train_test_split
    train_validate, test = train_test_split(df, test_size=.2, random_state=514, stratify=df['employed'])
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514, stratify=train_validate['employed'])
    
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] = scaler.transform(test[num_cols.columns])

    train_scaled = pd.get_dummies(train_scaled)
    validate_scaled = pd.get_dummies(validate_scaled)
    test_scaled = pd.get_dummies(test_scaled)

    return train, validate, test, train_scaled, validate_scaled, test_scaled