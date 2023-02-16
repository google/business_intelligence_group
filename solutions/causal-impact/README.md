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
#### CausalImpact Analysis Section

1. Press the **Connect** button to connect to the runtime

2. Run **Step 1** cell. Step 1 cells take a little longer because they install the [tfcausalImpact library](https://github.com/WillianFuks/tfcausalimpact).<br>If you do so, you will see some selections in the Step 1 cell.

3. In question 1, choose **CausalImpact Analysis** and update period before the intervention(**Pre Start & Pre End**) and the period during the intervention(**Post Start and Post End**).<br>
    ![ci_step1_1](https://user-images.githubusercontent.com/61218928/219256195-ba8d5e5d-df1e-4eb6-8df3-4021056122f6.png)

4. In question 2, please select the data source from **google_spreadsheet**, **CSV_file**, or **Big_Query**.<br>
    Then enter the required items.<br>
    ![ci_step1_2](https://user-images.githubusercontent.com/61218928/219256224-47af732f-f3d6-4a46-8eea-fb8eb13f82b8.png)

5. After entering the required items, the data format will be selected. For CausalImpact analysis, please prepare the data in **wide format** in advance.<br>
    After selecting wide format, enter the **date column name**.<br>
    ![ci_step1_3](https://user-images.githubusercontent.com/61218928/219256241-52ab2ad7-a3e7-413c-b27b-b397867ba89c.png)

6. Once the items are filled in, run the **Step 2** cell. <br>
    (:warning: If you have selected **google_spreadsheet** or **big_query**, a pop-up will appear regarding granting permission, so please grant it to Colab.) 
    
7. After Step 2 is executed, you will see **the results of CausalImpact Analysis**.
    ![ci_step4](https://user-images.githubusercontent.com/61218928/213954148-2c811170-d025-4663-a91c-d7941ce48ae3.png)

#### Experimental Design Section

1. Press the **Connect** button to connect to the runtime

2. Run **Step 1** cell. Step 1 cells take a little longer because they install the [tfcausalImpact library](https://github.com/WillianFuks/tfcausalimpact).<br>If you do so, you will see some selections in the Step 1 cell.

3. In question 1, choose **Experimental Design** and update the term(**Start Date & End Date**) to be used in the Experimental Design.
    ![ed_step1_1](https://user-images.githubusercontent.com/61218928/219262327-5dfd104f-c413-4ba4-b793-cf16badb6b84.png)

4. After updating the term, select the **type of Experimental Design** and update the required items.<br>
    * A: divide_equally divides the time series data into n groups with similar movements.
    * B: similarity_selection extracts n groups that move similarly to a particular column.

    ![ed_step1_2](https://user-images.githubusercontent.com/61218928/219262346-ccd9cb99-f45e-477d-81f2-f32263d842f7.png)
    
5. After updating required items, enter the estimated incremental CPA.
    ![ed_step1_3](https://user-images.githubusercontent.com/61218928/219262361-96d91a95-dc45-49e8-9fbf-dc00a2ddbca8.png)

6. In question 2, please select the data source from **google_spreadsheet**, **CSV_file**, or **Big_Query**.<br>
    Then enter the required items.<br>
    ![ed_step1_4](https://user-images.githubusercontent.com/61218928/219256224-47af732f-f3d6-4a46-8eea-fb8eb13f82b8.png)

6. After entering the required items, select data format [**narrow_format** or **wide_format**](https://en.wikipedia.org/wiki/Wide_and_narrow_data) and enter the required fields.
    ![ed_step1_5](https://user-images.githubusercontent.com/61218928/219262376-86e807b1-15f5-4551-a598-4dcda4ba410d.png)

7. Once the items are filled in, run the **Step 2** cell. <br>
    (:warning: If you have selected **google_spreadsheet** or **big_query**, a pop-up will appear regarding granting permission, so please grant it to Colab.) 

8. The output results will vary depending on the type of experimental design, but select the data on which you want to run the simulation.

9. Once the items are filled in, run the **Step 3** cell. Depending on the data, this may take more than 10 minutes. 
    After Step 3 is run, the results are displayed in a table. Check the MAPE, budget and p-value, and consider the intervention period and the assumed increments to experimental design.
    ![ed_step3](https://user-images.githubusercontent.com/61218928/213636393-c3ad5fe3-a373-4f0e-b3e3-602013a433d6.png)
    
10. run the **Step 4** cell. <br>
    ![ed_step4](https://user-images.githubusercontent.com/61218928/213636438-13e18342-8162-4985-be29-df9bb9f6cfbc.png)
