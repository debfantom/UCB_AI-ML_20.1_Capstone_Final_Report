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
The dataset is sourced from Kaggle: [Book Crossing Dataset](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset?select=BX-Book-Ratings.csv). Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books. It contains:

● BX-Users- Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Age and location is provided if available.

● BX-Books - Identified by their respective ISBN. Book metadata, such as Book-Title, Book-Author, Year-Of-Publication, Publisher, were obtained from Amazon Web Services. 

● BX-Book-Ratings - Contains the book rating information (User-ID, ISBN, Rating). 

---

### 🧠 Hypotheses

1. **Behavioral features are more effective than static demographics for clustering and age group prediction.**  
   Patterns in reading volume, rating behavior, and title preferences will yield more meaningful user segments and support more accurate age group imputation, which can then guide personalization and messaging.

2. **Title themes (i.e., common words in book titles) can indicate future user interest.**  
   Users are more likely to engage with books that share title themes with their previously read or highly rated books. Identifying these patterns enables content-based personalization, even for users with limited prior ratings.

---

### ✅ How We’ll Evaluate These

- For Hypothesis 1:
  - Train a model to predict `age_group` using only behavior-based features
  - Evaluate performance using holdout test data from users with valid age
  - Assess how well behavioral clusters align with age group patterns

- For Hypothesis 2:
  - Split each user’s interaction history into train/test sets
  - Derive top title words from the training set
  - Measure how often test-set titles include those words
  - Evaluate lift over random or baseline match rates

---

Would you like this inserted into a `Hypotheses & Evaluation` section of your README or kept separate for now?

## 🧭 Expected Results/Results
The project aims to identify user clusters based on rating behaviors, such as preferences for certain genres or rating patterns (e.g., frequent raters vs. rare raters). The clusters will be visualized to evaluate separation and meaning, supporting potential personalized strategies in content platforms. 

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()

### Contact and Further Information


