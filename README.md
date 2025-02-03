# Authors
Ramiro H. Rodriguez

# Directory Structure

    ├── README.md                    <- The top-level README for developers using this project.
    │
    ├── reports                      <- Generated analysis as PDF, LaTeX, etc.
    │   ├── Final_report              
    │   └── Project_presentation     
    |
    ├── data
    │   ├── external                 <- Data from third party sources.
    │   ├── interim                  <- Intermediate data that has been transformed.
    │   ├── processed                <- The final, canonical data sets for modeling.
    │   └── raw                      <- The original, immutable data dump.
    │
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    |
    ├── notebooks                    <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                   the creator's initials, and a short `-` delimited description, e.g.
    │                                   `1.0-jqp-initial-data-exploration`.
    │
    ├── references                   <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── streamlit                    <- Source code for streamlit application.
    │   ├── stremalit_app.py   
    │   ├── project_tools.py
    │   ├── FusionModel_withVGG.py
    │   ├── images/                  <- Images for the streamlit page
    │   └── trained_models           <- Copy of trained models from models [to fix]
    │   
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


# Project Reports
* ```Rakuten_Final_Report_Ramiro_Rodriguez.pdf``` is a detailed report of the data exploration, preprocessing and modeling phases. It contains information about the models benchmark, evaluation and fine tuning, as well as the final model performance.  

* ```Project_Presentation.pdf``` are the slides for a brief description of the project.


# Environment 
Activate conda environment with:
```
conda activate rakuten_env2
```


# Streamlit application
Navigate to the repository streamlit folder. Make sure to activate the conda environment
```
cd C:\PATH_TO_REPOSITORY\streamlit
```

Then launch the streamlit application on the browser
```
streamlit --version
streamlit run streamlit_app.py
```
Have Fun !


# Versions
Version 3.0: Refactoring working project.  
Version 2.2: Final Project


