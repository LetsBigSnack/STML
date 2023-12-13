# Basic Data Handling and Visualization Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# NLP Libraries
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Machine Learning Libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, make_pipeline

# LIME and SHAP for Explainable AI
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import shap


## Text - Cleaning

def RemoveSpecialCharacters(sentence):
    return re.sub('[^a-zA-Z0-9!?]+',' ',sentence)

def ConvertAndRemove(sentence):
    sentence = str(sentence)
    sentence = RemoveSpecialCharacters(sentence)
    # convert to lower case
    return sentence

def CleanText(sentence):
    sentence = str(sentence)

    # Remove stopwords
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', 'im', 'dont', 'doin', 'ure']
    # Remove punctuation
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    sentence = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    sentence = ConvertAndRemove(sentence)
    return sentence

## Exploratory analysis

def datasetAnalysis(dataframe):
    # Displays the Structure of the Dataframe
    print("Dataframe Structure:")
    print(df.info())
    print("---------------------")
    # Displays the first 5 entries in the Dataset
    print("First 5 Entries:")
    print(df.head())
    print("---------------------")
    # Checks for missing Values and display the information
    print("Number of missing Values")
    print(df.isnull().sum())
    print("---------------------")
    plt.figure(figsize=(10, 10))  
    sns.countplot(x=dataframe.clickbait)
    plt.title('Clickbait Distribution') 
    plt.ylabel('Count')           
    plt.xlabel('Clickbait')   
    plt.savefig('plots/distribution.png') 
    plt.close()

## Model training
## Model evaluation
def getPredictions(Classifiers, X_train_clean, y_train, X_test_dtm):
    predictions = []
    for c in Classifiers:
        try:
            classifier = c['model']
            fit = classifier.fit(X_train_clean, y_train)
            pred = fit.predict(X_test_dtm)
            
        except Exception:
            fit = classifier.fit(X_train_clean, y_train)
            pred = fit.predict(X_test_dtm)

        predictions.append(pred)
    return predictions

def plotMetrics(Classifiers,y_test, predictions):
    score_Accuracy = []
    score_F1 = []
    score_Recall = []
    score_Precision = []
    Model = []

    for i,p in enumerate(predictions):
        accuracy = accuracy_score(y_test,p)
        print(f"The accuracy score of {Classifiers[i]['label']} is {accuracy}")
        prec_score = precision_score(y_test, p, average='weighted')
        print(f"The precision score of {Classifiers[i]['label']} is {prec_score}")
        recall = recall_score(y_test, p, average='weighted')
        print(f"The recall score of {Classifiers[i]['label']} is {recall}")
        f1 = f1_score(y_test, p, average='weighted')
        print(f"The F1 score of {Classifiers[i]['label']} is {f1}")
        print("-----------------------------------------------")

        score_Accuracy.append(accuracy)
        score_F1.append(f1)
        score_Recall.append(recall)
        score_Precision.append(prec_score)

    for c in Classifiers:
        Model.append(c['label'])

  
    Index = [1,2,3,4,5]
    plt.figure(figsize=(10, 10))  
    plt.bar(Index,score_Accuracy)
    plt.xticks(Index,Model,rotation=45)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Accuracies of Models')
    plt.savefig('plots/accuracy.png')
    plt.close()

    Index = [1,2,3,4,5]
    plt.figure(figsize=(10, 10))  
    plt.bar(Index,score_F1)
    plt.xticks(Index,Model,rotation=45)
    plt.ylabel('F1')
    plt.xlabel('Model')
    plt.title('F1 of Models')
    plt.savefig('plots/f1.png')
    plt.close()

    Index = [1,2,3,4,5]
    plt.figure(figsize=(10, 10))  
    plt.bar(Index,score_Precision)
    plt.xticks(Index,Model,rotation=45)
    plt.ylabel('Precision')
    plt.xlabel('Model')
    plt.title('Precisions of Models')
    plt.savefig('plots/percision.png')
    plt.close()

    Index = [1,2,3,4,5]
    plt.figure(figsize=(10, 10))  
    plt.bar(Index,score_Recall)
    plt.xticks(Index,Model,rotation=45)
    plt.ylabel('Recall')
    plt.xlabel('Model')
    plt.title('Recalls of Models')
    plt.savefig('plots/recall.png')
    plt.close()

def plotAUCROC(Classifiers,X_train_clean, y_train, X_test_dtm):
    plt.figure()
    # Below for loop iterates through your models list
    for m in Classifiers:
        model = m['model'] # select the model
        model.fit(X_train_clean, y_train) # train the model
        y_pred=model.predict(X_test_dtm) # predict the test data
    # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test_dtm)[:,1])
    # Calculate Area under the curve to display on the plot
        auc = metrics.roc_auc_score(y_test,model.predict(X_test_dtm))
    # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('plots/aucroc.png')
    plt.close()

## Explainability



# Classifieres
Classifiers = [
{
   'label': 'Logistic_Regression',
   'model': LogisticRegression(C=0.00000001,solver='liblinear',max_iter=200, multi_class='auto'),
},
{
    'label': 'SGD_Classifier',
    'model': SGDClassifier(loss='log', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False),
},
{
    'label': 'KNeighbours',
    'model': KNeighborsClassifier(n_neighbors=15),
},
{
    'label': 'Decision_Tree',
    'model': DecisionTreeClassifier(max_depth=10,random_state=101,max_features= None,min_samples_leaf=15),
},
{
   'label': 'Random_Forest',
   'model': RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,random_state=101,max_features= None,min_samples_leaf = 30),
}
]


# Loads the Dataset from a local .csv file
df=pd.read_csv('data/clickbait_data.csv')

#Removing Special Characters and transforming text to lower case in the headline column
df['Text_cleaning'] = df.headline.apply(CleanText)
df.head()

#Define the features (X) and label (Y)
X = df.Text_cleaning
y = df.clickbait

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_clean = tfidf_transformer.fit_transform(X_train_dtm)


datasetAnalysis(df)
predictions = getPredictions(Classifiers, X_train_clean, y_train, X_test_dtm)
plotMetrics(Classifiers, y_test, predictions)
plotAUCROC(Classifiers,X_train_clean,y_train,X_test_dtm)

# Model Testing

model = SGDClassifier(loss='log', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False)
model.fit(X_train_clean, y_train)

pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfid', TfidfTransformer()),
                 ('model', Classifiers[1]['model'])])
pipe.fit(X_train, y_train)

# CHATGPT Generated Clickbair / Non-Clickbait Titles
many_titles = [
    "You Won't Believe What This Celebrity Ate for Breakfast!",
    "Shocking Secrets Your Doctor Doesn't Want You to Know!",
    "10 Mind-Blowing Tricks to Get Rich Overnight!",
    "This Simple Trick Will Make You Lose 20 Pounds in a Week!",
    "Aliens Found? The Government Is Hiding the Truth!",
    "New Study Reveals a Miracle Cure for Aging!",
    "The One Thing You Should Never Eat—It's Killing You Slowly!",
    "Is Your Partner Cheating? Find Out Now with This App!",
    "Secret Government Plot Exposed: You're in Danger!",
    "Unbelievable! Man Survives a Shark Attack Using This Household Item!",
    "Understanding the Benefits of Regular Exercise for Your Health",
    "Tips for Effective Time Management in a Busy World",
    "Exploring the Latest Advances in Renewable Energy",
    "The Impact of Social Media on Mental Health",
    "Interview with Expert: Navigating Career Changes Successfully",
    "Recent Scientific Discoveries in Space Exploration",
    "Local Community Event Aims to Combat Hunger",
    "Guide to Healthy Eating Habits for Long-Term Wellness",
    "How to Improve Sleep Quality",
    "Exploring Different Meditation Techniques for Stress Relief",
    "You won't believe what happened next",
    "10 Shocking Secrets That Will Change Your Life",
    "The One Thing You're Doing Wrong and How to Fix It",
    "The Top 5 Most Surprising (Insert topic here) of All Time",
    "The One Thing That Can Make or Break (Insert topic here)",
]

prediction = pipe.predict(many_titles)

for i in range(len(prediction)):
  if prediction[i] == 0:
    result = 'Not Clickbait'
  else:
    result = 'Clickbait'
  print(f'{result} -> {many_titles[i]}')

# LIME - Explaining a Prediction
explainer = LimeTextExplainer(class_names=['Not Clickbait', 'Clickbait'])
idx = 1  # Index of the sample you want to explain

for i in range(100):
    exp = explainer.explain_instance(X_test.iloc[i], pipe.predict_proba)
    exp.show_in_notebook(text=True)
    exp.save_to_file('lime/lime_'+str(i)+'.html')

# Create a SHAP explainer using the model in the pipeline
explainer = shap.Explainer(pipe.named_steps['model'], pipe.named_steps['bow'].transform(X_train))
# Compute SHAP values for a sample of the test set
shap_values = explainer.shap_values(pipe.named_steps['bow'].transform(X_test))
# Visualize the SHAP values - Summary plot
plt.close()  
# Clear the current figure and axes
plt.clf()
plt.cla()

shap.summary_plot(shap_values, pipe.named_steps['bow'].transform(X_test), feature_names=vect.get_feature_names_out(),show=False, max_display=20)
plt.savefig("plots/shap_summary.png")


plt.close()  
# Clear the current figure and axes
plt.clf()
plt.cla()

# Create and train the SGDClassifier
sgd_classifier = SGDClassifier(loss='log', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False)
sgd_classifier.fit(X_train_clean, y_train)

for i in range(100):
    plt.close()  
    # Clear the current figure and axes
    plt.clf()
    plt.cla()
    # Compute SHAP values
    explainer = shap.Explainer(sgd_classifier, X_train_clean)
    shap_values = explainer(X_test_dtm[i])

    # Visualize the SHAP values - Summary plot
    shap.summary_plot(shap_values, X_test_dtm[i], feature_names=vect.get_feature_names_out(), show=False, max_display=20)
    plt.savefig("shap/shap_summary_["+X_test.iloc[i].replace(" ", "_")+"].png")