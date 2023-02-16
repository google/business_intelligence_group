# CausalImpact with experimental design

##### This is not an official Google product.

CausalImpact is a R package for causal inference using Bayesian structural
time-series models. In using CausalImpact, the parallel trend assumption is
needed for counterfactual modeling, so this code performs classification of time
series data based on DTW distances.

## Overview

### What you can do with CausalImpact with experiment design

-   Experimental Design
    -   load time series data from google spreadsheet or csv file
    -   classify time series data so that parallel trend assumptions can be made
    -   simulate the conditions required for verification
-   CausalImpact Analysis
    -   load time series data from google spreadsheet or csv file
    -   CausalImpact Analysis

### Motivation to develop and open the source code

Some marketing practitioners pay attention to
[Causal inference in statistics](https://en.wikipedia.org/wiki/Causal_inference).　However,
using time series data without parallel trend assumptions does not allow for
appropriate analysis. Therefore, the purpose is to enable the implementation and
analysis of interventions after classifying time-series data for which parallel
trend assumptions can be made.

### Typical procedure for use

1. Assume a hypothetical solution to the issue and its factors.
2. Assume room for KPIs and the mechanisms that drive KPIs depending on the solution.
3. In advance, decide next-actions to be taken for each result of hypothesis testing (with/without significant difference).
    - Recommend supporting the mechanism with relevant data other than KPIs
4. Prepare time-series KPI data for at least 100 time points.
    - Regional segmentation is recommended.
    - Previous data, such as the previous year, may make a difference in the market environment.
    - Relevant data must be independent and unaffected by interventions
5. **(Experimental Design)** This tool is used to conduct the experimental design. 
    - Split into groups that are closest to each other where the parallel trend assumption can be placed.
    - Simulation of required timeframe and budget
    - :warning: If the parallel trend assumption cannot be placed, we recommend considering another approach
6. Implemented interventions.
7. Prepare time-series KPI data, including intervention period and assumed residual period, in addition to previous data.
8. **(CausalImpact Analysis)** Conduct CausalImpact analysis.
9. Implement next actions based on results of hypothesis testing

## Note

-   Do not do [HARKing](https://en.wikipedia.org/wiki/HARKing)(hypothesizing after the results are known)
-   Do not do [p-hacking](https://en.wikipedia.org/wiki/Data_dredging)

## Getting started

1.  Prepare the time series data on spreadsheet or csv file
2.  Open ipynb file with **[Open in Colab](https://colab.research.google.com/github/google/business_intelligence_group/blob/main/solutions/causal-impact/CausalImpact_with_Experimental_Design.ipynb)** Button.
3.  Run cells in sequence

## Tutorial

#### Experimental Design Section

1. Press the **Connect** button to connect to the runtime

2. Scroll to Experimental Design Section and run **Step 1** and **Step 2** cells

3. When the cell in Step 2 is executed, input fields will appear in the cell. Choose **google_spreadsheet** or **csv_file** as the data source and enter the required information in the selected source.

    ![ed_step2](https://user-images.githubusercontent.com/61218928/213386208-b43fba73-953a-4087-b65d-a1a3e008bad1.png)

4. Once the field is filled in, run the **Step 3** cell. (:warning: If you have selected **google_spreadsheet**, a pop-up will appear regarding granting permission, so please grant it to Colab.) After Step 3 is executed, the data will be read and the first 10 rows will be displayed. At the same time, input fields will appear, so enter **date column name** and select [**narrow_format** or **wide_format**](https://en.wikipedia.org/wiki/Wide_and_narrow_data) for the data format. In the case of **narrow_format**, also enter **pivot column name** and **kpi column name**.

    ![ed_step3](https://user-images.githubusercontent.com/61218928/213605948-aa150663-cbef-4939-9e9b-9b87c46c1b7f.png)

5. Once the fields are filled in, run the **Step 4** cell. After Step 4 is executed, the daily trend and 7-day moving average will be displayed. The "each" tab also allows you to see the daily trends for individual columns.

    ![ed_step4](https://user-images.githubusercontent.com/61218928/213606481-2704c85e-b1fa-455c-b324-797bece7b6be.png)
    
6. Next, run **Step 5** and **Step 6** cells; Step 6 cells take a little longer because they install the [tfcausalImpact library](https://github.com/WillianFuks/tfcausalimpact). After installation, input field will appear in the cell. 

    Are there columns that are not needed in the experimental design? If so, please enter the column names separated by commas. 
    
    Please also select the type of experimental design.
    * A: divide_equally divides the time series data into n groups with similar movements.
    * B: similarity_selection extracts n groups that move similarly to a particular column.

    ![ed_step6](https://user-images.githubusercontent.com/61218928/213608197-c4fcfae9-8704-4d5d-8333-553ca62aa362.png)

7. Once the fields are filled in, run the **Step 7** cell. After Step 7 is executed, you will see a list of the columns to be used this time and the input fields required for the design type you have selected. See the table in the cell for a description of the fields required for each design type.

    ![ed_step7](https://user-images.githubusercontent.com/61218928/213626558-16235d53-e964-4873-bdca-ef2dfa09aebd.png)
    
8. Once the fields are filled in, run the **Step 8** cell. After Step 8 is executed, You will see five choices in the given conditions. Choose one of those options and also the column that you want to test.

    ![ed_step8](https://user-images.githubusercontent.com/61218928/213634044-4a513433-d729-4cb3-8644-d70ddd5db3ef.png)

9. Once the fields are filled in, enter the estimated incremental CPA and run the **Step 9** cell. The simulation will then be run with the intervention period and provisional increments given. Depending on the data, this may take more than 10 minutes. 

    After Step 9 is run, the results are displayed in a table. Check the MAPE, budget and p-value, and consider the intervention period and the assumed increments to experimental design.
    
    ![ed_step9_1](https://user-images.githubusercontent.com/61218928/213636393-c3ad5fe3-a373-4f0e-b3e3-602013a433d6.png)
    
    ![ed_step9_2](https://user-images.githubusercontent.com/61218928/213636438-13e18342-8162-4985-be29-df9bb9f6cfbc.png)


#### CausalImpact Analysis Section

1. Press the **Connect** button to connect to the runtime

2. Run **Step 1** cell. If you do so, you will see some selections in the Step 1 cell.

3. In question 1, choose **CausalImpact Analysis** and update period before the intervention(**Pre Start & Pre End**) and the period during the intervention(**Post Start and Post End**).<br>
    ![ci_step1_1](https://user-images.githubusercontent.com/61218928/219256195-ba8d5e5d-df1e-4eb6-8df3-4021056122f6.png)

4. In question 2, please select the data source from **google_spreadsheet**, **CSV_file**, or **Big_Query**.<br>
    Then enter the required items.<br>
    ![ci_step1_2](https://user-images.githubusercontent.com/61218928/219256224-47af732f-f3d6-4a46-8eea-fb8eb13f82b8.png)

5. After entering the required items, the data format will be selected. For CausalImpact analysis, please prepare the data in **wide format** in advance.<br>
    After selecting wide format, please enter the **date column name**.<br>
    ![ci_step1_3](https://user-images.githubusercontent.com/61218928/219256241-52ab2ad7-a3e7-413c-b27b-b397867ba89c.png)

6. Once the items are filled in, run the **Step 2** cell. <br>
    (:warning: If you have selected **google_spreadsheet** or **big_query**, a pop-up will appear regarding granting permission, so please grant it to Colab.) 
    
7. After Step 2 is executed, you will see **the results of CausalImpact Analysis**.
    ![ci_step4](https://user-images.githubusercontent.com/61218928/213954148-2c811170-d025-4663-a91c-d7941ce48ae3.png)
