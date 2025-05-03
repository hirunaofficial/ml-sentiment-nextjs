import pandas as pd
import numpy as np
import re
import string
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set up plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Download NLTK resources with robust error handling
print("Downloading NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Will use simplified text processing")

# Setup stopwords
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['rt', 'via', 'amp', 'gt', 'lt']) # Add Twitter-specific stopwords
except Exception as e:
    print(f"Error loading stopwords: {e}")
    # Fallback to a basic set of stopwords
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                 'when', 'where', 'how', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'to', 'for', 'with', 'about',
                 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
                 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
                 'just', 'should', 'now', 'rt', 'via', 'amp', 'gt', 'lt'}

# Try to load lemmatizer, with fallback
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    have_lemmatizer = True
except Exception as e:
    print(f"Error loading lemmatizer: {e}")
    have_lemmatizer = False

# Enhanced sentiment lexicon - words with strong sentiment signals
# This will help improve confidence in classifications
positive_lexicon = {
    'amazing': 3.0, 'awesome': 3.0, 'excellent': 3.0, 'fantastic': 3.0, 'outstanding': 3.0,
    'perfect': 3.0, 'wonderful': 3.0, 'brilliant': 3.0, 'great': 2.5, 'love': 2.5,
    'best': 2.5, 'superb': 2.5, 'incredible': 2.5, 'terrific': 2.5, 'exceptional': 2.5,
    'good': 2.0, 'nice': 2.0, 'well': 2.0, 'like': 2.0, 'happy': 2.0,
    'enjoy': 2.0, 'impressive': 2.0, 'recommend': 2.0, 'favorite': 2.0, 'pleased': 2.0
}

negative_lexicon = {
    'terrible': 3.0, 'awful': 3.0, 'horrible': 3.0, 'worst': 3.0, 'disgusting': 3.0,
    'pathetic': 3.0, 'dreadful': 3.0, 'abysmal': 3.0, 'bad': 2.5, 'hate': 2.5,
    'poor': 2.5, 'disappointed': 2.5, 'frustrating': 2.5, 'disappointing': 2.5, 'useless': 2.5,
    'annoying': 2.0, 'mediocre': 2.0, 'problem': 2.0, 'issues': 2.0, 'fail': 2.0,
    'fails': 2.0, 'sucks': 2.0, 'waste': 2.0, 'boring': 2.0, 'broken': 2.0
}

# Define a robust text preprocessing function with enhanced features
def preprocess_text(text, extract_features=False):
    """
    Preprocess text with robust error handling and feature extraction
    If extract_features=True, returns a tuple of (cleaned_text, features_dict)
    """
    if not isinstance(text, str):
        if extract_features:
            return "", {}
        return ""
    
    # Original text for feature extraction
    original_text = text.lower()
    
    # Convert to lowercase
    text = text.lower()
    
    # Extract features before cleaning
    features = {}
    if extract_features:
        # Count exclamation marks (enthusiasm/intensity)
        features['exclamation_count'] = text.count('!')
        
        # Count question marks (uncertainty)
        features['question_count'] = text.count('?')
        
        # Count capitalized words (emphasis)
        features['caps_count'] = sum(1 for word in text.split() if word.isupper())
        
        # Check for strong positive and negative words
        features['positive_score'] = sum(positive_lexicon.get(word.lower(), 0) 
                                     for word in text.split())
        features['negative_score'] = sum(negative_lexicon.get(word.lower(), 0) 
                                     for word in text.split())
        
        # Common negation words
        negations = {'not', 'no', 'never', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"}
        features['has_negation'] = any(neg in text.split() for neg in negations)
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize safely
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords and apply lemmatization if available
    if have_lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    else:
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    cleaned_text = ' '.join(tokens)
    
    if extract_features:
        return cleaned_text, features
    return cleaned_text

# Load dataset
print("Loading dataset...")
try:
    # Try with header first
    df = pd.read_csv('data/sentiment_data.csv')
    
    # Check if expected columns exist
    if 'text' in df.columns and 'sentiment' in df.columns:
        print("Found expected columns: text and sentiment")
        text_column = 'text'
        sentiment_column = 'sentiment'
    elif 'Tweet' in df.columns and 'Sentiment' in df.columns:
        print("Found columns: Tweet and Sentiment. Renaming for consistency.")
        df = df.rename(columns={'Tweet': 'text', 'Sentiment': 'sentiment'})
        text_column = 'text'
        sentiment_column = 'sentiment'
    else:
        # Try to detect text and sentiment columns
        text_col = None
        sentiment_col = None
        for col in df.columns:
            if df[col].dtype == object:
                # Check if column contains text content
                if text_col is None and df[col].astype(str).str.len().mean() > 20:
                    text_col = col
                # Check if column contains sentiment labels
                elif sentiment_col is None and df[col].astype(str).str.lower().isin(['positive', 'negative', 'neutral']).mean() > 0.5:
                    sentiment_col = col
        
        if text_col and sentiment_col:
            print(f"Auto-detected text column: {text_col}")
            print(f"Auto-detected sentiment column: {sentiment_col}")
            df = df.rename(columns={text_col: 'text', sentiment_col: 'sentiment'})
            text_column = 'text'
            sentiment_column = 'sentiment'
        else:
            print("Could not auto-detect columns, using default assumptions")
            # Default assumptions
            df = df.iloc[:, :4]
            if len(df.columns) >= 4:
                df.columns = ['textID', 'text', 'selected_text', 'sentiment']
                text_column = 'text'
                sentiment_column = 'sentiment'
            else:
                # If less than 4 columns, assume simpler structure
                df.columns = ['textID', 'text', 'sentiment'][:len(df.columns)]
                text_column = 'text'
                sentiment_column = 'sentiment'
        
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Creating sample data for demonstration")
    # Create a sample dataset
    df = pd.DataFrame({
        'text': [
            "I absolutely love this product! It's amazing!",
            "This is the worst purchase I've ever made.",
            "It's okay, not great but not terrible.",
            "Great customer service! Very helpful staff.",
            "Never buying this again. Completely disappointed."
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
    })
    text_column = 'text'
    sentiment_column = 'sentiment'

# Display dataset info
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Standardize sentiment labels to lowercase
df[sentiment_column] = df[sentiment_column].str.lower()

# Show sentiment distribution
print("\nSentiment distribution:")
sentiment_counts = df[sentiment_column].value_counts()
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution in Dataset', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add count labels on top of bars
for i, count in enumerate(sentiment_counts.values):
    ax.text(i, count + 50, f"{count}", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('plots/sentiment_distribution.png', dpi=300)
plt.close()

# Keep only positive and negative for binary classification
if set(df[sentiment_column].unique()) - {'positive', 'negative'}:
    print("\nKeeping only positive and negative for binary classification...")
    df_filtered = df[df[sentiment_column].isin(['positive', 'negative'])]
    print(f"Dataset after filtering: {df_filtered.shape}")
    
    # Plot before and after filtering
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=df[sentiment_column], palette='viridis')
    plt.title('Before Filtering', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=df_filtered[sentiment_column], palette='viridis')
    plt.title('After Filtering (Binary)', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/sentiment_filtering.png', dpi=300)
    plt.close()
    
    # Update df to filtered version
    df = df_filtered
else:
    print("Dataset already contains only positive and negative sentiments.")

# Drop any missing text values
df = df.dropna(subset=[text_column])
print(f"Dataset after dropping null values: {df.shape}")

# Preprocess the text
print("\nPreprocessing text with feature extraction...")
# Apply preprocessing and extract features
preprocessed_data = [preprocess_text(text, extract_features=True) for text in df[text_column]]
df['clean_text'] = [item[0] for item in preprocessed_data]

# Add extracted features as columns
feature_keys = preprocessed_data[0][1].keys()
for key in feature_keys:
    df[key] = [item[1].get(key, 0) for item in preprocessed_data]

# Display sample of cleaned text and features
print("\nSample of cleaned text with features:")
sample_cols = ['clean_text', 'positive_score', 'negative_score', 'has_negation']
print(df[sample_cols].head())

# Generate word clouds for positive and negative texts
print("\nGenerating word clouds...")

def generate_wordcloud(texts, title, filename):
    """Generate and save wordcloud for a set of texts"""
    text = ' '.join(texts)
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        max_words=200,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Generate word clouds by sentiment
positive_texts = df[df[sentiment_column] == 'positive']['clean_text']
negative_texts = df[df[sentiment_column] == 'negative']['clean_text']

generate_wordcloud(positive_texts, 'Positive Sentiment Word Cloud', 'plots/positive_wordcloud.png')
generate_wordcloud(negative_texts, 'Negative Sentiment Word Cloud', 'plots/negative_wordcloud.png')

# Plot feature distributions
print("\nPlotting feature distributions...")

# Create a figure for feature distributions
plt.figure(figsize=(16, 12))

# Plot text length distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='text_length', hue=sentiment_column, bins=30, kde=True, palette='viridis')
plt.title('Text Length Distribution by Sentiment', fontsize=14)
plt.xlabel('Text Length (characters)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Plot word count distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='word_count', hue=sentiment_column, bins=30, kde=True, palette='viridis')
plt.title('Word Count Distribution by Sentiment', fontsize=14)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Plot exclamation count
plt.subplot(2, 2, 3)
sns.countplot(data=df, x='exclamation_count', hue=sentiment_column, palette='viridis')
plt.title('Exclamation Mark Counts by Sentiment', fontsize=14)
plt.xlabel('Number of Exclamation Marks', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xlim(-0.5, 5.5)  # Limit x-axis to common values

# Plot sentiment scores
plt.subplot(2, 2, 4)
df_melted = pd.melt(df, 
                    id_vars=[sentiment_column], 
                    value_vars=['positive_score', 'negative_score'],
                    var_name='Score Type', value_name='Score Value')
sns.boxplot(data=df_melted, x=sentiment_column, y='Score Value', hue='Score Type', palette='viridis')
plt.title('Sentiment Scores by Sentiment Class', fontsize=14)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Score Value', fontsize=12)

plt.tight_layout()
plt.savefig('plots/feature_distributions.png', dpi=300)
plt.close()

# Convert sentiment to binary label
df['sentiment_label'] = df[sentiment_column].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df, df['sentiment_label'], test_size=0.2, random_state=42, stratify=df['sentiment_label']
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create features for model training
print("\nPreparing features for model training...")

# Function to create feature matrices
def create_feature_matrix(df, vectorizer=None, is_training=False):
    """Create TF-IDF features and combine with additional features"""
    # Text features
    if is_training:
        # During training, fit the vectorizer
        tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    else:
        # During testing, use the fitted vectorizer
        tfidf_matrix = vectorizer.transform(df['clean_text'])
    
    # Additional engineered features
    feature_columns = [
        'positive_score', 'negative_score', 'exclamation_count', 
        'question_count', 'has_negation', 'word_count'
    ]
    
    # Convert additional features to numpy array
    additional_features = df[feature_columns].values
    
    # Return the TFIDF matrix and additional features
    return tfidf_matrix, additional_features

# Create TF-IDF vectorizer with improved parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,  # Increase vocabulary size
    min_df=2,           # Minimum document frequency
    max_df=0.95,        # Maximum document frequency
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    sublinear_tf=True,   # Apply sublinear tf scaling (logarithmic)
    use_idf=True,        # Use inverse document frequency
    norm='l2'            # Normalize with L2 norm
)

# Get text features
X_train_tfidf, X_train_extra = create_feature_matrix(X_train, tfidf_vectorizer, is_training=True)
X_test_tfidf, X_test_extra = create_feature_matrix(X_test, tfidf_vectorizer, is_training=False)

# Train optimized Logistic Regression model
print("\nTraining optimized Logistic Regression model...")
# Using stronger regularization and balanced class weights
log_reg = LogisticRegression(
    C=5.0,                # Lower C means stronger regularization; tune this value
    max_iter=2000,        # More iterations for convergence
    solver='liblinear',   # Usually works better for smaller datasets
    class_weight='balanced',  # Handles class imbalance
    dual=False,           # Feature count > samples
    random_state=42
)

# Train model on TF-IDF features
log_reg.fit(X_train_tfidf, y_train)

# Evaluate the basic model
y_pred_base = log_reg.predict(X_test_tfidf)
base_accuracy = accuracy_score(y_test, y_pred_base)
print(f"\nBase Logistic Regression model accuracy: {base_accuracy:.4f}")
print("\nClassification Report (Base Model):")
base_report = classification_report(y_test, y_pred_base, output_dict=True)
print(classification_report(y_test, y_pred_base))

# Plot confusion matrix for base model
cm = confusion_matrix(y_test, y_pred_base)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix - Base Model', fontsize=16)
plt.tight_layout()
plt.savefig('plots/base_confusion_matrix.png', dpi=300)
plt.close()

# Apply probability calibration for better confidence scores
print("\nCalibrating model probabilities for improved confidence estimates...")
calibrated_model = CalibratedClassifierCV(log_reg, method='sigmoid', cv=5)
calibrated_model.fit(X_train_tfidf, y_train)

# Evaluate calibrated model
y_pred_calibrated = calibrated_model.predict(X_test_tfidf)
calibrated_accuracy = accuracy_score(y_test, y_pred_calibrated)

# Calculate confidence scores (probabilities)
confidence_scores = calibrated_model.predict_proba(X_test_tfidf)
avg_confidence = np.mean(np.max(confidence_scores, axis=1))

print(f"Calibrated model accuracy: {calibrated_accuracy:.4f}")
print(f"Average confidence score: {avg_confidence:.4f}")

# Plot confusion matrix for calibrated model
cm_calibrated = confusion_matrix(y_test, y_pred_calibrated)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_calibrated, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix - Calibrated Model', fontsize=16)
plt.tight_layout()
plt.savefig('plots/calibrated_confusion_matrix.png', dpi=300)
plt.close()

# Plot ROC curve
y_pred_proba = calibrated_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=300)
plt.close()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = np.mean(precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f'Avg precision = {avg_precision:.3f}')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/precision_recall_curve.png', dpi=300)
plt.close()

# Plot confidence distribution
plt.figure(figsize=(10, 6))
sns.histplot(np.max(confidence_scores, axis=1), bins=20, kde=True, color='purple')
plt.axvline(x=avg_confidence, color='red', linestyle='--', 
            label=f'Average Confidence: {avg_confidence:.3f}')
plt.xlabel('Confidence Score', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Confidence Scores', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/confidence_distribution.png', dpi=300)
plt.close()

# Extract feature importance
if hasattr(log_reg, 'coef_'):
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get coefficients
    coefficients = log_reg.coef_[0]
    
    # Create a DataFrame of features and their importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    
    # Sort by absolute importance
    feature_importance['Abs_Importance'] = abs(feature_importance['Importance'])
    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
    
    # Get top 20 features
    top_features = feature_importance.head(20)
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    colors = ['green' if imp > 0 else 'red' for imp in top_features['Importance']]
    sns.barplot(x='Importance', y='Feature', data=top_features, palette=colors)
    plt.title('Top 20 Features by Importance', fontsize=16)
    plt.xlabel('Coefficient Value (Green = Positive, Red = Negative)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    
    # Save top positive and negative features to file
    pos_features = feature_importance[feature_importance['Importance'] > 0].head(20)
    neg_features = feature_importance[feature_importance['Importance'] < 0].head(20)
    
    pos_features.to_csv('results/top_positive_features.csv', index=False)
    neg_features.to_csv('results/top_negative_features.csv', index=False)


# Save in models directory for organization
joblib.dump(calibrated_model, 'models/sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/vectorizer.pkl')

# Save feature extraction function for prediction (via lambda serialization)
joblib.dump(preprocess_text, 'models/preprocess_function.pkl')

print("\nFiles saved successfully:")
print("- sentiment_model.pkl")
print("- vectorizer.pkl")
print("- Various visualization plots in 'plots' directory")

# Create a complete sentiment prediction function with visualization
def predict_sentiment(text, model_file='sentiment_model.pkl', vectorizer_file='vectorizer.pkl', 
                      visualize=False, save_path=None):
    """
    Predict sentiment for new text with enhanced confidence
    If visualize=True, creates and returns a visualization of the prediction
    """
    # Load model and vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    
    # Preprocess text and extract features
    clean_text, features = preprocess_text(text, extract_features=True)
    
    # Vectorize text
    text_vector = vectorizer.transform([clean_text])
    
    # Predict with calibrated probabilities
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Get confidence (highest probability)
    confidence = max(probabilities)
    
    # Get top influential words
    influential_words = []
    try:
        if hasattr(model, 'base_estimator'):
            # For calibrated classifier, get the base model
            base_model = model.base_estimator
        else:
            base_model = model
            
        if hasattr(base_model, 'coef_'):
            coefficients = base_model.coef_[0]
            vocabulary = vectorizer.vocabulary_
            
            # Check which words from the text are in the vocabulary
            words = clean_text.split()
            word_influences = []
            
            for word in set(words):
                if word in vocabulary:
                    idx = vocabulary[word]
                    influence = coefficients[idx]
                    word_influences.append((word, influence))
            
            # Sort by absolute influence
            word_influences.sort(key=lambda x: abs(x[1]), reverse=True)
            influential_words = word_influences[:5]  # Top 5 words
    except Exception as e:
        print(f"Could not extract influential words: {e}")
    
    # Calculate confidence boost from lexicon
    lexicon_confidence = 0
    for word in text.lower().split():
        if word in positive_lexicon and prediction == 1:
            lexicon_confidence += positive_lexicon[word] / 10  # Scale down the boost
        elif word in negative_lexicon and prediction == 0:
            lexicon_confidence += negative_lexicon[word] / 10  # Scale down the boost
    
    # Cap the lexicon confidence boost
    lexicon_confidence = min(lexicon_confidence, 0.2)  # Maximum 20% boost
    
    # Adjust confidence (but don't exceed 100%)
    adjusted_confidence = min(confidence + lexicon_confidence, 0.99)
    
    # Create result dictionary
    result = {
        "text": text,
        "cleaned_text": clean_text,
        "sentiment": "Positive 😀" if prediction == 1 else "Negative 😠",
        "confidence": f"{adjusted_confidence:.2%}",
        "raw_confidence": f"{confidence:.2%}",
        "influential_words": influential_words,
        "features": features
    }
    
    # Create visualization if requested
    if visualize:
        plt.figure(figsize=(12, 8))
        
        # Plot sentiment prediction
        plt.subplot(2, 2, 1)
        sentiment_colors = ['#FF6B6B', '#4ECB71']  # Red for negative, green for positive
        plt.bar(['Negative', 'Positive'], probabilities, color=sentiment_colors)
        plt.title('Sentiment Prediction', fontsize=14)
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)
        
        # Highlight the predicted class
        plt.annotate(f"{adjusted_confidence:.1%}", 
                    xy=(prediction, adjusted_confidence), 
                    xytext=(prediction, adjusted_confidence + 0.05),
                    fontsize=14, fontweight='bold',
                    ha='center')
        
        # Plot influential words
        plt.subplot(2, 2, 2)
        if influential_words:
            words = [word for word, _ in influential_words]
            scores = [score for _, score in influential_words]
            colors = ['green' if s > 0 else 'red' for s in scores]
            
            y_pos = range(len(words))
            plt.barh(y_pos, scores, color=colors)
            plt.yticks(y_pos, words)
            plt.xlabel('Influence Score', fontsize=12)
            plt.title('Top Influential Words', fontsize=14)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No influential words found", 
                    ha='center', va='center', fontsize=12)
            plt.title('Top Influential Words', fontsize=14)
        
        # Plot lexicon scores
        plt.subplot(2, 2, 3)
        lex_scores = [features['negative_score'], features['positive_score']]
        plt.bar(['Negative Words', 'Positive Words'], lex_scores, 
                color=['#FF6B6B', '#4ECB71'])
        plt.title('Sentiment Lexicon Scores', fontsize=14)
        plt.ylabel('Score', fontsize=12)
        
        # Plot text statistics
        plt.subplot(2, 2, 4)
        stats = [
            features['word_count'], 
            features['exclamation_count'],
            features['question_count'],
            int(features['has_negation'])
        ]
        stat_names = ['Word Count', 'Exclamations', 'Questions', 'Has Negation']
        plt.bar(stat_names, stats, color='skyblue')
        plt.title('Text Statistics', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or return the figure
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
            result['visualization_path'] = save_path
        else:
            result['visualization'] = plt.gcf()
            
    return result