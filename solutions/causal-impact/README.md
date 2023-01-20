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

## Note

-   Do not do [HARKing](https://en.wikipedia.org/wiki/HARKing)(hypothesizing after the results are known)
-   Do not do [p-hacking](https://en.wikipedia.org/wiki/Data_dredging)

## Getting started

1.  Prepare the time series data on spreadsheet or csv file
2.  Open ipynb file with **[Open in Colab](https://colab.research.google.com/github/google/business_intelligence_group/blob/main/solutions/causal-impact/CausalImpact_with_Experimental_Design.ipynb)** Button.
3.  Run cells in sequence

## Tutorial

1. Press the **Connect** button to connect to the runtime

2. Scroll to Experimental Design Section and run **Step 1** and **Step 2** cells

3. When the cell in Step 2 is executed, input fields will appear in the cell. Choose **google_spreadsheet** or **csv_file** as the data source and enter the required information in the selected source.

    ![ed_step2](https://user-images.githubusercontent.com/61218928/213386208-b43fba73-953a-4087-b65d-a1a3e008bad1.png)

4. Once the field is filled in, run the **Step 3** cell. (:warning: If you have selected **google_spreadsheet**, a pop-up will appear regarding granting permission, so please grant it to Colab.) After Step 3 is executed, the data will be read and the first 10 rows will be displayed. At the same time, input fields will appear, so enter **date column name** and select [**narrow_format** or **wide_format**](https://en.wikipedia.org/wiki/Wide_and_narrow_data) for the data format. In the case of **narrow_format**, also enter **pivot column name** and **kpi column name**.

    ![ed_step3](https://user-images.githubusercontent.com/61218928/213605948-aa150663-cbef-4939-9e9b-9b87c46c1b7f.png)

5. Once the fields are filled in, run the **Step 4** cell. After Step 4 is executed, the daily trend and 7-day moving average will be displayed. The "each" tab also allows you to see the daily trends for individual columns.

    ![ed_step4](https://user-images.githubusercontent.com/61218928/213606481-2704c85e-b1fa-455c-b324-797bece7b6be.png)
    
6. Next, run **Step 5** and **Step 6** cells; Step 6 cells take a little longer because they install the [tfcausalImpact library](https://github.com/WillianFuks/tfcausalimpact).
After installation, an input field will appear in the cell.

    ![ed_step6](https://user-images.githubusercontent.com/61218928/213608197-c4fcfae9-8704-4d5d-8333-553ca62aa362.png)
