# UCB_AI-ML_20.1_Capstone_Initial_Report_EDA
---
This repository contains:
- **book_clustering_EDA.ipynb**: Jupyter Notebook containing exploratory data analysis, feature engineering, model comparison, and tuning.
- **data/**: Folder containing the dataset used in this project. 
- **images/**: Folder with generated plots and visualizations.
---
## 🧠 Executive Summary
This project investigates how user behavior around book ratings can be used to uncover distinct groups of readers using unsupervised learning. The focus is on clustering users into interpretable segments to enable more personalized book recommendations and strategic content targeting.

## 🎯 Rationale
**Why this matters**: Understanding user behavior is critical for strategic decision-making in digital content platforms. By clustering users with interpretable, behavior-based features, this project can uncover actionable insights. These segments can guide product development, content strategy, and marketing priorities—enabling more focused investments and personalized user experiences. Without this analysis, platforms risk missing key engagement drivers and allocating resources toward generic strategies that may not resonate.

## ❓ Research Question
Can we cluster users based on interpretable features extracted from their book rating behavior to uncover distinct reader segments?

## 💾 Data Sources 
The dataset is sourced from Kaggle: [Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset?select=Books.csv). It contains:

● Users.csv - Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Age is provided if available.

● Books.csv - Identified by their respective ISBN. Book metadata, such as Book-Title, Book-Author, Year-Of-Publication, Publisher, were obtained from Amazon Web Services. 

● Ratings.csv - Contains the book rating information (User-ID, ISBN, Rating). 

Note from the datacard:

out of 278859 users:
- only 99053 rated at least 1 book
- only 43385 rated at least 2 books.
- only 12306 rated at least 10 books.
  
out of 271379 books:
- only 270171 are rated at least once.
- only 124513 have at least 2 ratings.
- only 17480 have at least 10 ratings.


Under



## 🧭 Expected Results/Results
The project aims to identify user clusters based on rating behaviors, such as preferences for certain genres or rating patterns (e.g., frequent raters vs. rare raters). The clusters will be visualized to evaluate separation and meaning, supporting potential personalized strategies in content platforms. 

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()

### Contact and Further Information


