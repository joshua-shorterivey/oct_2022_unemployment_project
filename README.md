# October 2022 Unemployment Project
* The Current Population Survey (CPS), sponsored jointly by the U.S. Census Bureau and the U.S. Bureau of Labor Statistics (BLS), is the primary source of labor force statistics for the population of the United States. This project uses the October 2022 edition of the Basic Monthly CPS. 

## Project Objectives  
* Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 
* Create modules that faciliate project repeatability, as well as final report readability

* Explore features more in depth compared to August iteration of project. 
    * Look for more crossover beetween features
* Construct model to predict `employed` status 
    * Experiment with different model types in order to obtain better predicitive results compared to August
* Refine work into `final_report` in form of jupyter notebook. 
    * Detail work done, underlying rationale for decisions, methodologies chosen, findings, takeaways, and conclusions.

## Project  Goals
* Construct ML Classification model that accurately predicts `employed` status of survey respondents using clustering techniques to guide feature selection for modeling</br>

* Focus on having a high True Negative Rate 

* Deliver report that the can read through while understanding what steps were taken, why and what the outcome was.

* Provide instructions to replicate project.

* Make recommendations on what works or doesn't work in predicting `employed` status.

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
</br>

## Data Dictionary
|       Target           | Non_Null Count/Datatype  |     Definition     |
|:-----------------------|:------------------------:|-------------------:|  
employed                 | 50556 non-null  object  | employed

|       Feature          |           Datatype      |     Definition     |
|:----------------------|:------------------------:|-------------------:|  
housing_type		    | 50556 non-null  object  |    TYPE OF HOUSING UNIT
family_income		    | 50556 non-null  object  |	FAMILY INCOME
household_num		    | 50556 non-null  object  |	TOTAL NUMBER OF PERSONS LIVING 
household_type		    | 50556 non-null  object  |	HOUSEHOLD TYPE					
own_bus_or_farm		    | 50556 non-null  object  |	DOES ANYONE IN THIS HOUSEHOLD
country_region		    | 50556 non-null  object  |	DIVISION
state		            | 50556 non-null  object  |	FEDERAL INFORMATION
metropolitan		    | 50556 non-null  object  |	METROPOLITAN STATUS						
metro_area_size		    | 50556 non-null  object  |	Metropolitan Area (CBSA) SIZE
age                     | 50556 non-null  object  |	PERSONS AGE
marital_status		    | 50556 non-null  object  |	MARITAL STATUS 										
veteran		            | 50556 non-null  object  |	DID YOU EVER SERVE ON ACTIVE DUTY								
education			    | 50556 non-null  object  |	HIGHEST LEVEL OF SCHOOL 								
race		            | 50556 non-null  object  |	RACE											
hispanic_or_non		    | 50556 non-null  object  |	HISPANIC OR NON-HISPANIC								
birth_country		    | 50556 non-null  object  |	COUNTRY OF BIRTH
mother_birth_country    | 50556 non-null  object  |	COUNTRY OF MOTHER'S BIRTH
father_birth_country    | 50556 non-null  object  |	COUNTRY OF FATHER'S BIRTH									
citizenship	            | 50556 non-null  object  |	CITIZENSHIP STATUS									
upaid_work_last_week	| 50556 non-null  object  |	LAST WEEK, DID YOU DO ANY													
usual_hours_worked		| 50556 non-null  object  |	DO YOU USUALLY WORK 35 HOURS OR	MORE PER WEEK							
children_in_household	| 50556 non-null  object  |	PRESENCE OF OWN CHILDREN <18 YEARS	
professional_certification |51736 non-null float64 |    DOES � HAVE A CURRENTLY

### Spotlight - Professional Certification
* **Question:** What is the effect of having a professional certification? 
 
* **Answer:** Most indivduals do not have certification, but those that do have a 2% unemployment rate vs 4% for those without.

>#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between having a `professional_certification` and `employment`  
>* ${H_a}$: There is a relationship between having a `professional_certification` and `employment`    
>* ${\alpha}$: .05
>* Result: There is enough evidence to reject our null hypothesis. 

### Spotlight - Race
* **Question:** Which industry shows the largest population proportion change between employed and unemployed?

* **Answer:** Indivduals identifying as White show the largest population proportion change with a drop of nearly 10% when comparing employed vs unemployed. Those identifying as mixed race other than with white, and Indigenous have the highest unemployed rates at 12% and 7% respectively. 

>#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between `race` and `employment` status   
>* ${H_a}$: There is a relationship between `race` and `employment` status   
>* ${\alpha}$: .05  
>* Result: There is enough evidence to reject our null hypothesis. 

### Spotlight - Industry 
* **Question:** Which industry shows the largest population proportion change between employed and unemployed?  
* **Answer:** Leisure and Hospitality. This industry also has the highest unemployment rate at 6%

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between industry of typical employment and employment status   
>* ${H_a}$: There is a relationship between industry of typical employment and employment status  
>* ${\alpha}$: .05  
>* Result: There is enough evidence to reject our null hypothesis. 
 
### Spotlight - Professional Certification
* **Question:** What is the effect of having a professional certification? 
 
* **Answer:** Most indivduals do not have certification, but those that do have a 2% unemployment rate vs 4% for those without.

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between having a `professional_certification` and `employment`  
>* ${H_a}$: There is a relationship between having a `professional_certification` and `employment`    
>* ${\alpha}$: .05
>* Result: There is enough evidence to reject our null hypothesis.

## Summary of Key Findings
>
-----
</br></br></br>

# Pipeline Walkthrough
## Plan
* Create and build out project README  
* Create skeletons of required, as well as supporting, project modules and notebooks
* `env.py`, `wrangle_zillow.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
    * Decide which colums to import
    * Made further decisions to eliminate `state` and immigration features after explore phase
* Examine Data Dictionary from Census
* Shrink to 30 Columns
* Add to data dictionary during process
* Delete columns identified as filler in census data dictionary
* Reduce to items not dependent on other responses
    * Make decision on how to handle various subgroups in the survey reponses
    * Had to flatten certain features to reduce dimensionality for modeling 
* Decide target disposition, and NLF inclusion
    * Eliminated all NLF (Not in Labor Force) responses
    * Utilized only those with clear employed vs unemployed status
* Deal with null values
    * Few null values after controlling for employement status
    * Treated nulls where applicable as other than affirmative responses
* Think through different imputation strategies
    * Used KNN imputer for school, occupation, and industry features
* Investigate columns that can be dropped or have redundant information  
* Decide how to deal with outliers 
    * Kept outliers as is for now. Next iteration will more closely look at flattening them into max+ type columns
* Work through questions involving variables typically present on resume
* Develop 5 deeper questions to ask of each spotlight category
    * Did further examination, but not specifically 5 deeper questions
    * Will address for next iteration
* Create additional spotlight category
    * Spotlighted Immigration categories, but they had minimal effect on employment disposition. Elimated from modeling consideration
* Primary focus for first pass through explore phase
* Craft general explore section outline
* Include areas for feature engineering via cluster, rfe, and selectKbest
* Work on modeling section
* Craft functions for model testing. **NO BIG MASS TESTING FUNCTION**
* Prep MVP
* Decide which questions to highlight within Final Report
* Go back through data dictionary
* Craft outline/skeleton for final report
    * Takeaways highlighted    
* Look into creating feature that blends `household_num` and `family_income` to measure at or above median income for family of that size
    * Unable to craft for this iteration of Project
----

# Wrangle (Acquire and Prep)
* Largest and most time intensive part of wrangle was deciding which columns to drop. 
* Deviated heavily from initial plan to **only** feature resume-like categories. Sub-optimal decision.

> ### Nulls/Missing Values
* Dropped any records that have 'NaN' for target variable, indicates incomplete survey/data. - Same rationale used to justify dropping most observations that had a '-1' value as that indicated the repsondent did not an apporiate repsonse for the area of the survey
* For some others such as `usual_hours_worked` information from multiple columns was used to infer proper disposition
* Example: Individuals that report variable work hours were assigned population mean hours work for above/or below 35 hours. 
* Further work on the project will require more research of how the survey is conducted and the data entered. 
---
>### Feature Engineering 
* Decided against engineering features due to poor model performance with accurately predicting employment disposition. No point increasing complexity
---   
> ### Flattening
* Had to make decisions in order to remove optionality from certain categorical columns when preparing the data
- Example: `race` orginally had over 20 different categories and was ~flattened~ down to 7 
* Decisons here driven mostly by desire to create larger cohorts within  features because the unemployment is already such as small amount

>## Exploration Summary
* Overall the conventional wisdom surrounding job prospects held true.

* It benefits an indvidual to acquire advanced dregrees and certifications
* Having a job or career in an industry that leans more towards being a profession helps
* With more time I want to dive into cross examinations of factors to see how they interact, but I'm doubtful that would help more than simply satisfying my curiousity. 
---
# Modeling
* * Decided to focus on only three types of models for this iteration of project.
* DTC modeling showed promise, but was prone to overfitting when it came time for use on the validation and test subsets
* Linear SVC and XGBOOST performed worse during the model phase, and they were not moved forward.
* Further iterations of project will focus much more on exploration and feature reduction in order to reduce noise and dimensionality. 


## Deliver **
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
----
</br>

## Project Reproduction Requirements
> Requires oct22pub.csv featuring Octbober 2022 CPS Data from U.S. Census Bureau
> <a href="https://www2.census.gov/programs-surveys/cps/datasets/2022/basic/oct22pub.csv">Available here</a>
> Steps:
* Fully examine this `README.md`
* Download oct22pub.csv to working directory
* Run `Final Report.ipynb`