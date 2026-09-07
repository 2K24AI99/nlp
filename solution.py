import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Task 1: Bag of Words Matrix ----------

corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(corpus)

vocabulary = vectorizer.get_feature_names_out()

bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vocabulary)
bow_df.index = [f"Document {i+1}" for i in range(len(corpus))]

print("Vocabulary:")
print(list(vocabulary))
print("\nBag of Words Matrix:")
print(bow_df)


from sklearn.metrics.pairwise import cosine_similarity

# ---------- Task 2: Document Search Engine ----------

documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

query = ["machine learning algorithms for data"]

doc_vectorizer = CountVectorizer(stop_words='english')
doc_vectors = doc_vectorizer.fit_transform(documents)
query_vector = doc_vectorizer.transform(query)

scores = cosine_similarity(query_vector, doc_vectors)[0]

results = pd.DataFrame({
    "Document": documents,
    "Similarity Score": scores
})

ranked_results = results.sort_values(by="Similarity Score", ascending=False)

print("\nQuery:", query[0])
print("\nRanked Documents (highest similarity first):")
print(ranked_results.to_string(index=False))