# Authors
Ramiro H. Rodriguez

# Description of the project:
Cataloging products according to different data (texts and images) is important for e-commerce since it allows for various applications such as product recommendation and personalized research. It is then a question of predicting the type code of the products knowing textual data (designation and description of the products) as well as image data (image of the product).

# Directory Structure

    ├── README.md                    <- The top-level README for developers using this project.
    │
    ├── reports                      <- Generated analysis as PDF, LaTeX, etc.
    │   ├── Final_report              
    │   └── Project_presentation     
    |
    ├── data                         <- Data from third party sources.
    │   
    │
    ├── rakuten_env2.yml             <- The requirements file for reproducing the analysis environment with conda
    │                                   
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



# Data: 
This project is part of the Rakuten France Multimodal Product Data Classification challenge, the data and their description are available at: https://challengedata.ens.fr/challenges/35 
* Text data: ~60 mb
* Image data: ~2.2 gb
* 99k data with over 1000 classes.  

<br>

# Project Reports
* ```Rakuten_Final_Report_Ramiro_Rodriguez.pdf``` is a detailed report of the data exploration, preprocessing and modeling phases. It contains information about the models benchmark, evaluation and fine tuning, as well as the final model performance.  

* ```Project_Presentation.pdf``` are the slides for a brief description of the project.

<br>

# Environment 
Activate conda environment with:
```
conda activate rakuten_env2
```

<br>

# Streamlit application
Navigate to the repository streamlit folder on this project. Make sure to activate the conda environment
```
cd streamlit/
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


