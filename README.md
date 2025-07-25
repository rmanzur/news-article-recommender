## 📰 News Article Recommender Systems

### Objective

To create high-performing content-based recommender systems using article datasets provided by Microsoft News (MIND).

### Datasets:

The datasets for this project can be downloaded at https://msnews.github.io/ and made available by
"Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. MIND: A Large-scale Dataset for News Recommendation. ACL 2020."

### Technologies:

Python (sci-kit learn, pandas, numpy, streamlit), Jupyter Notebook

### Baseline Content-based Recommender Model: 
In this project, I used TF-IDF in combination with cosine similarity to compute textual similarities for articles in the news dataset.
  * TF-IDF was used to convert the "tags" column consisting of keywords from the title, category, subcategories, and abstract into feature vectors. The transformation captures the relative importance of each word within a document to its importance across the entire corpus.
  * Cosine similarity was then applied to measure how similar two news articles are based on the angle between their TF-IDF vectors.
  * One of the key challenges was hyperparameter tuning -> selecting the optimal "max_features" value for the TF-IDF vectorizer, which controls the maximum number of terms to include based on term frequency across the corpus.
      - To select a good value, I evaluated the resulting cosine similarity matrix using:
        - Variance of the similarity scores: to ensure meaningful differentiation between articles.
        - Density of high-similarity values: to control how many articles were considered similar, avoiding over-sparsity or over-clustering.
  * Model evaluation was performed through manual inspection of results, checking whether the top-ranked articles (based on cosine similarity) were contextually and semantically relevant to each query article.
  * To further refine results, the model was rerun to filter out results that did not match the category of the query article.
![Image](https://github.com/user-attachments/assets/2141485c-c1ed-4a69-bbde-506e6345d209)
