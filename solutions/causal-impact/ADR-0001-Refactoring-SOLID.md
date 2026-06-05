# Architecture Decision Record: Refactoring CausalImpact Notebook for SOLID and Google Python Style Guide

## Status
Accepted

## Context
The CausalImpact analysis solution consists of a single Jupyter Notebook (`CausalImpact_with_Experimental_Design.ipynb`) that handles both experimental design and causal impact analysis. Over time, the internal logic structure has grown, leading to tighter coupling between the data loading, pre-processing, UI interactions, and core algorithm components. The codebase did not strictly adhere to SOLID principles or the Google Python Style Guide (such as consistent Type Hints and Google-style English DocStrings).

In refactoring the code, we had two distinct architectural paths:
1. Extract classes into external `.py` files to completely decouple the codebase.
2. Keep the classes within the Jupyter Notebook but redesign their internal architecture.

## Decision
We decided to **maintain the single Jupyter Notebook structure** and refactor the internal classes to adhere to SOLID principles and the Google Python Style Guide.

Key structural changes implemented:
1. **Dependency Inversion Principle (DIP) / Open-Closed Principle (OCP)**:
   - Refactored `DataLoader` to use the Strategy Pattern by introducing the `IDataLoader` protocol.
   - Specific data loaders (`GoogleSheetLoader`, `CSVLoader`, `BigQueryLoader`) now implement the protocol, allowing for straightforward addition of new data sources without modifying the main `DataLoader` orchestrator.
2. **Type Hints and Documentation**:
   - Added comprehensive type hinting (`from typing import Dict, Any, List, Tuple, Protocol`, etc.) across internal data-wrangling components (`DataPreprocessor`, `ExploratoryDataAnalyzer`).
   - Standardized all classes and methods with Google-style English Docstrings.
3. **Dependency update**:
   - Fixed Numba version incompatibility by explicitly setting `tslearn==0.7.0`.

## Consequences
- **Positive:** UI/UX for end-users remains seamless, as they only need to interact with a single Jupyter Notebook file (no need to clone a repo or manually upload `.py` files to Google Colab).
- **Positive:** Future additions of data loading or processing steps will have less ripple effect across the codebase, due to well-defined protocols.
- **Negative:** The single notebook file is long, and managing large chunks of raw Python code inside a JSON notebook is slightly harder for development and code-reviews, requiring specialized scripts or strict adherence to disciplined notebook editing.
