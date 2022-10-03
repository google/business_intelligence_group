# CausalImpact with experimental design

##### This is not an official Google product.

CausalImpact is a R package for causal inference using Bayesian structural
time-series models. In using CausalImpact, the parallel trend assumption is
needed for counterfactual modeling, so this code performs classification of time
series data based on DTW distances.

## Overview

### What you can do with CausalImpact with experiment design

-   Experimental Design
    -   load time series data from spreadsheets
    -   classification of time series data
    -   export time series data to spreadsheets
-   Analysis
    -   load time series data from spreadsheets
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

1.  Prepare the time series data on spreadsheets
2.  Open ipynb in Google Colab.
3.  Run cells in sequence
