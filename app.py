import streamlit as st
from joblib import load
import os

st.title('Articles Recommended For You')

st.markdown("""
    <style>
    /* Changing background color */
    .stApp {
        background-color: #fff9c4;
    }
    /* Increase font size of the label text above selectbox */
    .custom-label {
        font-size: 24px;
        font-weight: light;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
article_path = os.path.join(BASE_DIR,"news_df.joblib")
similarity_path = os.path.join(BASE_DIR,"similarity.joblib")

article_df = load(article_path)
article_df_titles = article_df['title'].values
similarity = load(similarity_path)

st.markdown('<div class="custom-label">Please select an article that you like from the list below:</div>', unsafe_allow_html=True)
selected_article = st.selectbox('-->', article_df_titles)

def recommend_articles(article): #baseline recommender model using tf-idf and cosine similarity
    article_index = article_df[article_df['title'] == article].index[0]
    similarity_scores = similarity[article_index]
    article_list = sorted(list(enumerate(similarity[article_index])),reverse = True, key=lambda x: x[1])[1:6]

    return [(article_df.iloc[i[0]].title) for i in article_list]

def recommend_articles_within_category(article): #updated version of baseline model
    article_index = article_df[article_df['title'] == article].index[0]
    category = article_df.loc[article_df['title'] == article, 'category'].values[0]
    similarity_scores = similarity[article_index]
    same_category_article_indices = article_df[article_df['category'] == category].index
    # filter out articles that are not in the same category as the input
    article_list = sorted(list(enumerate(similarity[article_index])), reverse=True,
                          key=lambda x: x[1])
    filtered_list = [i for i in article_list if i[0] in same_category_article_indices][1:6]

    return [article_df.iloc[i[0]].title for i in filtered_list]

if st.button('Recommend'):
    recommendations1 = recommend_articles(selected_article)
    recommendations2 = recommend_articles_within_category(selected_article)

    st.subheader("Recommended Articles using TF_IDF + cosine similarity")
    for title in recommendations1:
        st.write(title)
    st.subheader("Recommended Articles using TF_IDF + cosine similarity + filtered by Category")
    for title in recommendations2:
        st.write(title)
