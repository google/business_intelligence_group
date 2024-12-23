# Business Intelligence Group - Marketing Analysis & Data Science Solution Packages

##### This is not an official Google product.

## Overview
This repository provides a collection of Jupyter Notebooks designed to help marketing analysts and data scientists measure and analyze the effectiveness of marketing campaigns. These notebooks facilitate data preprocessing, visualization, statistical analysis, and machine learning model building, enabling a deep dive into campaign performance and the extraction of actionable insights. By customizing and utilizing these notebooks, marketing analysts and data scientists can develop and execute more effective campaign strategies, ultimately driving data-informed decisions and optimizing marketing ROI.

### Motivation to develop and open the source code
-   CausalImpact with experimental design
    -   Some marketing practitioners pay attention to
[Causal inference in statistics](https://en.wikipedia.org/wiki/Causal_inference). However,
using time series data without parallel trend assumptions does not allow for
appropriate analysis. Therefore, the purpose is to enable the implementation and
analysis of interventions after classifying time-series data for which parallel
trend assumptions can be made.

For contributions, see [CONTRIBUTING.md](https://github.com/google/business_intelligence_group/blob/main/CONTRIBUTING.md).

### Available solution packages:
-   [CausalImpact with experimental design](https://github.com/google/business_intelligence_group/tree/main/solutions/causal-impact)


## Note
-   **<ins>Analysis should not be the goal</ins>**
    -   Solving business problems is the goal.
    -   Be clear about the decision you want to make to solve business problems.
    -   Make clear the path to what you need to know to make a decision.
    -   Analysis is one way to find out what you need to know.
-   **<ins>[Test your hypotheses instead of testing the effectiveness]</ins>**
    -   Formulate hypotheses about why there are issues in the current situation and how to solve them.
    -   Business situations are constantly changing, so analysis without a hypothesis will not be reproducible.
-   **<ins>[Be honest with the data]</ins>**
    -   However, playing with data to prove a hypothesis is strictly prohibited.
    -   Acquire the necessary knowledge to be able to conduct appropriate verification.
    -   Do not do [HARKing](https://en.wikipedia.org/wiki/HARKing)(hypothesizing after the results are known)
    -   Do not do [p-hacking](https://en.wikipedia.org/wiki/Data_dredging)
