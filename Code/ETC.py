from umap import UMAP
from nltk import WordNetLemmatizer
from sklearn.cluster import KMeans
from lightgbm import plot_importance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import nltk
import string
import pycountry
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))


def country_to_continent(country_name, world):
    if country_name in world['name'].values:
        continent = world[world['name'] == country_name].iloc[0]['continent']
        return continent
    else:
        return country_name


def vectorize_categories(document_df):
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = document_df.tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(embeddings.shape)

    # umap_model = UMAP(n_neighbors=10, min_dist=0.5, n_components=2, random_state=42, metric='cosine')
    # reduced_embeddings = umap_model.fit_transform(embeddings)
    return embeddings


def clustering_categories(document_df, embeddings, n_clusters):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=1000).fit(embeddings)
    cluster_label = kmeans.labels_
    document_df = pd.concat([document_df, pd.DataFrame(cluster_label, columns=['cluster'])], axis=1)
    return document_df, cluster_label


def print_reduced_cluster(document_df):
    for i, label in enumerate(list(set(document_df['cluster']))):
        print(label)
        print(set(document_df.iloc[:, 0][document_df['cluster'] == label]))
        num = document_df.iloc[:, 0][document_df['cluster'] == label].count() / len(document_df) * 100
        print(f'{num:.2f}%')
        print()


def plot_UMAP(title, reduced_embeddings, cluster_labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        plt.scatter(reduced_embeddings[cluster_labels == label, 0],
                    reduced_embeddings[cluster_labels == label, 1], color=colors[i], label=label, alpha=0.6)

    plt.title(f"{title}", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def LemTokens(tokens):
    lemmar = WordNetLemmatizer()
    return [lemmar.lemmatize(word=token, pos='n') for token in tokens]


def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.wordpunct_tokenize(text.lower().translate(remove_punct_dict)))


def verterize(df):
    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 3))
    ftr_vect = tfidf_vect.fit_transform(df)
    print(ftr_vect.shape)
    return ftr_vect


def plot_top_categories_conversion_rate(df, category_col, target_col, lower, upper):
    conversion_counts = df.groupby([category_col, target_col]).size().unstack(fill_value=0)
    conversion_rates = conversion_counts.div(conversion_counts.sum(axis=1), axis=0)
    top_categories = conversion_rates[True].sort_values(ascending=False).index
    top_categories = top_categories[lower:upper]
    top_conversion_rates = conversion_rates.loc[top_categories]

    top_conversion_rates.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'From{lower} to {upper}/{category_col} Conversion Rate')
    plt.xlabel(category_col)
    plt.ylabel('Conversion Rate')
    plt.xticks(rotation=45)
    plt.legend(title='Is Converted', labels=['False', 'True'])
    plt.tight_layout()
    plt.show()


def calculate_roc_curve(gt, prediction, plot):
    # ROC Curve
    thresholds = np.linspace(0, 1, 100)
    TPRs, FPRs = [], []
    distances = []

    for threshold in thresholds:
        TP = np.sum(np.logical_and(gt == 1, prediction >= threshold))
        FN = np.sum(np.logical_and(gt == 1, prediction < threshold))
        FP = np.sum(np.logical_and(gt == 0, prediction >= threshold))
        TN = np.sum(np.logical_and(gt == 0, prediction < threshold))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPRs.append(TPR)
        FPRs.append(FPR)

        # (0,1)까지의 유클리드 거리 계산
        distance = np.sqrt((FPR - 0) ** 2 + (TPR - 1) ** 2)
        distances.append(distance)

    # 최소 거리와 해당하는 임계값 찾기
    min_distance_index = np.argmin(distances)
    optimal_threshold = thresholds[min_distance_index]

    if plot:
        # ROC curve 그리기
        plt.plot(FPRs, TPRs, label='ROC curve')
        plt.plot([0, 1], [0, 1], '--', label='Random')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    print(f'Optimal Threshold: {optimal_threshold}')

    return optimal_threshold


def plot_importace_lightGBM(bst):
    plt.figure(figsize=(10, 8))
    plot_importance(bst, importance_type='split')
    plt.title('Feature Importance by Weight')
    plt.show()

    plt.figure(figsize=(10, 8))
    plot_importance(bst, importance_type='gain')
    plt.title('Feature Importance by Gain')
    plt.show()


def map_pycountry_name(text):
    for country in pycountry.countries:
        if country.name.lower() in text.lower():
            return country.name
    return 'none'


def map_gpd_name(text):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    for country in world['name']:
        if country.lower() in text.lower():
            return country
    return text
