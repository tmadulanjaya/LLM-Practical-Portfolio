# Lab 01
## Objective
Practice basic NLP tasks by classifying sentences as positive or negative using text preprocessing, Bag of Words and traditional machine learning models.

## Dataset
[Sentiment Labelled Sentences (UCI)]( https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences): 3,000 sentences from Amazon, IMDb and Yelp, labelled 1 (positive) or 0 (negative). After cleaning and removing duplicates, 2,971 sentences were used.

## What I Did
1. Loaded and combined the three dataset files
2. Preprocessed the text (lowercase, removed punctuation, numbers and extra spaces) into a `clean_text` column
3. Explored the data: class balance, example sentences and most frequent words
4. Split the data 80% training and 20% testing
5. Created a Bag of Words representation with `CountVectorizer`
6. Trained Logistic Regression and Linear SVM models
7. Evaluated the models using accuracy, precision, recall and F1-score
8. Tested the model on new sentences

## Notebook
- Kaggle: [Run on Kaggle](https://www.kaggle.com/code/ashendulanjaya/lab-01)

## Environment
- Platform: Kaggle Notebook (Python 3)
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.824 | 0.829 | 0.815 | 0.822 |
| Linear SVM | 0.807 | 0.807 | 0.805 | 0.806 |

Logistic Regression performed best. Both models have balanced precision and recall.

**Strongest positive words:** great, nice, excellent, amazing, awesome
**Strongest negative words:** bad, worst, poor, terrible, awful

### Screenshots
![Frequent words](./screenshots/Frequent Words.png)
![Results table](./screenshots/Results Table.png)
![Confusion matrices](./screenshots/Confusion Matrices.png)

## Key Learnings
- Text must be cleaned and converted to numbers before a model can use it.
- Bag of Words is a simple but effective way to represent text.
- Fitting the vocabulary on training data only avoids data leakage.
- Stopwords like "not" were kept because they change the sentiment.

## Challenges & Limitations
- Bag of Words ignores word order and context, so sarcasm and negation can be misclassified.
- Neutral sentences like "It was okay, nothing special" were predicted as Negative because the dataset has no neutral class.
