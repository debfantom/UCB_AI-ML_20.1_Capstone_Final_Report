# UCB_AI-ML_20.1_Capstone_Initial_Report_EDA

**Author** Debra Fant
---
This repository contains:
- **book_clustering_EDA.ipynb**: Jupyter Notebook containing exploratory data analysis, feature engineering, model comparison, and tuning.
- **data/**: Folder containing the datasets used in this project. 
- **images/**: Folder with generated plots and visualizations.
---
## 🧠 Executive Summary
This project investigates how user behavior around book ratings can be used to uncover distinct groups of readers using unsupervised learning. The focus is on clustering users into interpretable segments to enable more personalized book recommendations and strategic content targeting.

## 🎯 Rationale
**Why this matters**: Understanding user behavior is critical for strategic decision-making in digital content platforms. By clustering users with interpretable, behavior-based features, this project can uncover actionable insights. These segments can guide product development, content strategy, and marketing priorities—enabling more focused investments and personalized user experiences. Without this analysis, platforms risk missing key engagement drivers and allocating resources toward generic strategies that may not resonate.

## ❓ Research Question
Can we cluster users based on interpretable features extracted from their book rating behavior to uncover distinct reader segments?

**Hypothesis**: It is possible to uncover meaningful and interpretable user segments by clustering users based on book rating behavior and available profile information. These segments can reveal distinct patterns in engagement, preferences, and interest, supporting more targeted content, product, and marketing strategies.


## 💾 Data Sources 
The dataset is sourced from Kaggle: [Book Crossing Dataset](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset?select=BX-Book-Ratings.csv). Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books. It contains:

● BX-Users- Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Age and location is provided if available.

● BX-Books - Identified by their respective ISBN. Book metadata, such as Book-Title, Book-Author, Year-Of-Publication, Publisher, were obtained from Amazon Web Services. 

● BX-Book-Ratings - Contains the book rating information (User-ID, ISBN, Rating).

Additionally, I am using data from [Simple Maps, US Cities Data](https://https://simplemaps.com/data/us-cities) to get lat and lng for US cities.  Manual google search for high volume cities not included in the Simple Maps data to create [loc_lat_lng.csv]("data/loc_lat_lng.csv")

---
### Methodology
The project follows a streamlined version of CRISP-DM for unsupervised learning:

**1. Business Understanding**  
Develop reader segments that reflect different behaviors and engagement patterns using clustering techniques. The outcome will support personalization and strategic content decisions, particularly in environments with limited demographic data.

**2. Data Understanding**  
Integrate the 3 data files and explore and assess the structure, completeness, and behavioral richness of the Book-Crossing dataset. This includes rating patterns, book metadata (titles, authors, publishers, year), and user demographics. 

**3. Data Preparation**  
- Clean and structure the raw dataset
- Parse and standardize book metadata (author, publisher, pub year)
- Parse location into city, region, country (when available)
- Apply TF-IDF to title words (Book_titles excluding stop words) with N-grams = 1,2
- Engineer user-level features (examples below):
  - Reading and rating counts
  - Publication era preferences
  - Title-word extraction:
    - `interest_title_words` (all books)
    - `fav_title_words` (books rated > 7)
  - Favorite author/publisher (mode or frequency)
- Optionally split the dataset (e.g., 80/20) to reserve a portion for future prediction experiments.


**4. Modeling**  
- Use clustering techniques (ie KMeans) to cluster users based on behavioral and demographic features
- Apply dimensionality reduction (PCA) for visualization only
- Evaluate interpretability and separation of clusters


**5. Evaluation**  
- Silhouette Score to guide K choice
- Distribution analysis of users across clusters
- Interpretability based on key features per segment


**6. Deployment / Insights**  
- Assign persona labels to users
- Investigate implications for:
  - Personalized book recommendations
  - Thematic or genre-based content promotions
  - Identifying underserved reader personas


**7. Tools**
- **Data Wrangling**: `pandas`, `numpy`, `os`, `zipfile`  
- **Visualization**: `matplotlib.pyplot`, `seaborn`  
- **Text Processing**: `re`, `collections.Counter`, `nltk` (`stopwords`, `WordNetLemmatizer`)  
- **Modeling & Feature Engineering (scikit-learn)**:  
  - Preprocessing: `StandardScaler`, `OneHotEncoder`, `TfidfVectorizer`  
  - Workflow: `Pipeline`, `ColumnTransformer`, `train_test_split`  
  - Clustering & Evaluation: `KMeans`, `PCA`, `silhouette_score`

---

## 🧭 Results


## ✅ Next Steps
- **Feature Engineering Refinement**  
  Review features for opportunties to iterate
  
- **Model Exploration**  
  Evaluate alternative clustering algorithms such as DBSCAN to compare with KMeans results.

- **Hyperparameter Tuning**  
  Optimize the number of clusters (`k`) using grid search, test against different feature subsets.

- **Cluster Interpretation & Labeling**  
  Add richer profiling of clusters (e.g., top authors, preferred eras, average rating patterns) to support clearer persona labels.

- **Downstream Applications**  
  Explore opportunities to experiment utilizing the holdout data set and new learnings.





