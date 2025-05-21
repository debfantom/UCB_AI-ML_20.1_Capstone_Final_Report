# UCB_AI-ML_24.1_Capstone_Final_Report

**Author** Debra Fant
---
This repository contains:
- **book_xclustering_Final_Report.ipynb**: Jupyter Notebook containing exploratory data analysis, feature engineering, model comparison, and tuning and final report.
- **data/**: Folder containing the datasets used in this project. 
- **images/**: Folder with generated plots and visualizations.
- **APPENDIX**:
   - **book_xclustering_Initial_Report_EDA.ipynb**: Jupyter Notebook containing exploratory data analysis, feature engineering.
---
## 🧠 Executive Summary
This project investigates how user behavior around book ratings can be used to uncover distinct groups of readers using unsupervised learning. The focus is on clustering users into interpretable segments to enable more personalized book recommendations and strategic content targeting.

## 🎯 Rationale
**Why this matters**: Understanding user behavior is critical for strategic decision-making in digital content platforms. By clustering users with interpretable, behavior-based features, this project can uncover actionable insights. These segments can guide product development, content strategy, and marketing priorities—enabling more focused investments and personalized user experiences. Without this analysis, platforms risk missing key engagement drivers and allocating resources toward generic strategies that may not resonate.

## ❓ Research Question
Can we cluster users based on interpretable features extracted from their book rating behavior to uncover distinct reader segments?

**Hypothesis**: By clustering users using features derived from their book ratings and basic profile attributes, we can identify distinct, interpretable user segments. These segments will reflect real differences in reading habits, preferences, and engagement — enabling more personalized content, product features, and marketing strategies.


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
- Apply TF-IDF to title words (lemmatized book titles excluding stop words) with N-grams = 1,2
- Engineer user-level features (examples below):
  - Reading and rating counts and derivative features
  - Publication era preferences
  - Title-word extraction:
    - `interest_title_words` (all books)
    - `fav_title_words` (books rated > 7)
  - Favorite author/publisher (mode or frequency)
  - After initial review
    * Removed "Wild Animus" records (promotional anomaly)
    * Removed unrated (0) interactions
    * Excluded some fields from clustering to reduce dimentionality but retained for profiling
    * **TF-IDF vectors retained** for profiling interest/favorite title words
    * Switched to **SBERT embeddings** for clustering on book titles to improve semantic richness
- Optionally split the dataset (e.g., 80/20) to reserve a portion for future prediction experiments.


**4. Modeling**  
- Use clustering techniques (ie KMeans) to cluster users based on behavioral and demographic features
 * Expanded to include:
    * `AgglomerativeClustering` with multiple linkages
    * `DBSCAN` (tested but not tuned; instability and dense parameter sensitivity)
 * Hyperparameter tuning performed for:
    * `KMeans` (`k` range with silhouette + custom utility score)
    * `AgglomerativeClustering` (`k` + linkage types)
- Apply dimensionality reduction (PCA) for visualization only
- Evaluate interpretability and separation of clusters


**5. Evaluation**  
- Goals:
  - Silhouette Score to guide K choice
  - Distribution analysis of users across clusters
  - Interpretability based on key features per segment
- To balance the above goals, clustering models were evaluated using a **custom utility score** to balance statistical quality and real-world usefulness:

$$
\text{Utility Score} = 0.4 \times \text{Silhouette Score} + 0.1 \times \text{Balance Score} + 0.5 \times \left(\frac{\text{Usable Clusters}}{k}\right)
$$


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
  - Workflow: `Pipeline`, `ColumnTransformer`, `SentenceTransformer`, `train_test_split`  
  - Clustering & Evaluation: `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `PCA`, `silhouette_score`, 'time'

---

## 🧭 Results/Learnings
- **📊 Exploratory Data Analysis (EDA)**
  - While the dataset included a wide age range of users, most users fell into a small band, ages 25-40.  In the baseline cluster analysis, the mean age of each cluster only varied about 2.5 years.
  - A significant portion of user interactions in the dataset lack explicit feedback — 63% of ratings are zero (647k/1031k), indicating implicit interactions or unrated activity. This means that fewer than half of all book interactions result in a true rating (1–10).
  - Since the vast majority of readers are from the US, I focused the analysis on U.S. users. This also allowed me to utilize latitude and longitude for US cities for meaningful location-based clustering.
  - Interestingly, the most read book was "Wild Animus" by Rich Shapero despite the fact that it had mediocre rating of 4.4/10.  This book has an interesting [backstory](https://litreactor.com/columns/what-the-hell-is-wild-animus).  I excluded these records in the final modeling.
  ![My Image](images/exploratory_histograms.png)
  ![My Image](images/Top50Distributions.png)

- **🎯 Cluster Analysis Baseline K-Means Model**
  - The sillouette score of the baseline model after final feature engineering  and adjustments is .481 with k=3.  This is a decent score but >91% of users are in a single cluster which is not very useful.
  - At k=4, the sillouette score drops to .136 indicating limited separation.  Two of the clusters are pretty balanced at 49% each which is useful, while the other two clusters have minimal users and are not useful.
  - The PCA projection (for k=4) confirms that most user clusters are relatively close together, with limited visual separation. This suggests that the clusters may capture subtle variations in user behavior rather than strongly distinct personas. 
 ![My Image](images/userclustersviaPCA.png)


### 🧠 Cluster Summary & Personas 
NOTE: Excludes clusters 3 and 4 (both <0.1% of users) 

| C | % Users | Read Count | Rated High Count | Avg Rating | Author Diversity | Favorite Era | Favorite Authors                         | Most Read Books                                               | Persona Name             |
|---:|--------|------------|------------------|------------|------------------|---------------|------------------------------------------|----------------------------------------------------------------|--------------------------|
| 0  | 7.3%   | 41.76      | 26.38            | 7.82       | 0.68             | 1990s         | Janet Evanovich, Dean R. Koontz, Stephen King | A Painted House, Beloved, A Bend in the Road                | 🔁 The Loyal Devourer     |
| 1  | 28.2%  | 2.88       | 1.60             | 7.59       | 0.99             | 1990s         | Alice Sebold, John Grisham, Barbara Kingsolver | A Painted House, The Lovely Bones, 1st to Die               | 💬 The Sentimental Curator |
| 2  | 31.9%  | 2.86       | 1.57             | 7.53       | 0.99             | 1990s         | Stephen King, John Grisham, James Patterson   | A Painted House, The Lovely Bones, A Child Called "It"      | 👁️ The Thrill Seeker      |
| 5  | 32.6%  | 2.97       | 1.69             | 7.69       | 0.99             | 1990s         | Barbara Kingsolver, Dean R. Koontz, Nora Roberts | A Bend in the Road, A Heartbreaking Work..., A Child Called "It" | 🌍 The Curious Explorer     |


---
### ✍️ Persona Descriptions (in order of prevelance) & Strategy Recommendations

### 🌍 **The Curious Explorer (Cluster 5)** 

> 🌱 “I explore new voices and ideas, but I don’t stay long — I sample and move on.”

* **Users:** 10,139 (32.6%)
* **Books Read (avg):** 2.97
* **Top Authors:** Barbara Kingsolver, Dean R. Koontz, Nora Roberts
* **Most Read Books:** *A Bend in the Road*, *A Heartbreaking Work of Staggering Genius*, *A Child Called 'It'*
* **Top Interest Title Words:** `life`, `mystery`, `love`, `classic`, `club`
* **Top Favorite Title Words:** `life`, `classic`, `world`, `woman`, `guide`

**Behavioral Traits:**
Light readers with globally curious, socially reflective interests. Sample widely, value perspective.

**💡 Strategy Recommendation:**

* Promote globally relevant or award-winning titles
* Offer “worldview expanding” collections or challenges
* Use discovery-based messaging (e.g., “Expand your bookshelf”)

---

### 👻 **The Thrill Seeker (Cluster 2)**

> 👁️ “I’m into big names and dark stories — thrillers, crime, and dramatic plots.”

* **Users:** 9,946 (31.9%)
* **Books Read (avg):** 2.9
* **Top Authors:** Stephen King, John Grisham, James Patterson
* **Most Read Books:** *A Painted House*, *The Lovely Bones*, *A Child Called 'It'*
* **Top Interest Title Words:** `mystery`, `romance`, `series`, `hardcover`, `love`, `death`
* **Top Favorite Title Words:** `mystery`, `series`, `romance`, `death`, `history`, `love`

**Behavioral Traits:**
Drawn to dark, high-drama fiction by popular authors. Engages lightly but consistently with high-profile content.

**💡 Strategy Recommendation:**

* Surface trending thrillers and crime fiction
* Feature suspense bundles or “like Stephen King?” tracks
* Use high-urgency copy in re-engagement messaging

--- 

### 📖 **The Sentimental Curator (Cluster 1)**

> 💬 “I want emotional, literary stories that stay with me — even if I only read a few.”

* **Users:** 8,777 (28.2%)
* **Books Read (avg):** 2.9
* **Top Authors:** Alice Sebold, John Grisham, Barbara Kingsolver
* **Most Read Books:** *A Painted House*, *The Lovely Bones*, *1st to Die*
* **Top Interest Title Words:** `life`, `mystery`, `woman`, `guide`, `love`
* **Top Favorite Title Words:** `life`, `classic`, `woman`, `american`, `time`

**Behavioral Traits:**
Emotionally selective and values literary quality. Explores occasionally but intentionally.

**💡 Strategy Recommendation:**

* Promote titles with strong emotional hooks
* Offer curated lists like “Stories That Stay With You”
* Create personalized mini-challenges (e.g., 3 books in 3 months)

---
### 🧑‍💼 **The Loyal Devourer (Cluster 0)**

> 🧭 “I read a lot and stick with what I know — mystery, life stories, and trusted authors.”

* **Users:** 2,282 (7.3%)
* **Books Read (avg):** 41.8
* **Top Authors:** Janet Evanovich, Dean R. Koontz, Stephen King
* **Most Read Books:** *A Painted House*, *Beloved*, *A Bend in the Road*
* **Top Interest Title Words:** `mystery`, `life`, `guide`, `love`, `series`, `woman`
* **Top Favorite Title Words:** `life`, `mystery`, `series`, `potter`, `love`, `classic`

**Behavioral Traits:**
Reads heavily and loyally — drawn to mystery and emotionally rich fiction, often from trusted series authors.

**💡 Strategy Recommendation:**

* Highlight next-in-series and favorite-author alerts
* Offer “reading streak” challenges or loyalty badges
* Personalize feeds with author collections and lifetime stats









