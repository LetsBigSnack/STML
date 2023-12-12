import pandas as pd
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import lime
import sklearn.ensemble
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
# LIME and SHAP Imports
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import shap

df=pd.read_csv('data/clickbait_data.csv')
df.info()

#checking for missing values
df.isnull().sum()

sns.countplot(x=df.clickbait)

nltk.download('stopwords')


def RemoveSpecialCharacters(sentence):
    return re.sub('[^a-zA-Z]+',' ',sentence)

def ConvertToLowerCase(sentence):
    return sentence.lower()

def ConvertAndRemove(sentence):
    sentence = str(sentence)
    sentence = RemoveSpecialCharacters(sentence)
    # convert to lower case
    sentence = ConvertToLowerCase(sentence)
    return sentence

def CleanText(sentence):
    sentence = str(sentence)

    # Remove stopwords
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Remove punctuation
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    sentence = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    sentence = ConvertAndRemove(sentence)
    return sentence


#Function testing
print(CleanText('I am going to the Ne\'therla\'nds and I\'m going to win an Olympic medal.'))


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

print(X_train_clean)


Classifiers = [
{
   'label': 'Logistic Regression',
   'model': LogisticRegression(C=0.00000001,solver='liblinear',max_iter=200, multi_class='auto'),
},
{
    'label': 'SGD Classifier',
    'model': SGDClassifier(loss='log', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False),
},
{
    'label': 'KNeighbours',
    'model': KNeighborsClassifier(n_neighbors=15),
},
{
    'label': 'Decision Tree',
    'model': DecisionTreeClassifier(max_depth=10,random_state=101,max_features= None,min_samples_leaf=15),
},
{
   'label': 'Random Forest',
   'model': RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,random_state=101,max_features= None,min_samples_leaf = 30),
}
]


Accuracy=[]
Model=[]
prediction = []
for c in Classifiers:
    try:
        classifier = c['model']
        fit = classifier.fit(X_train_clean, y_train)
        pred = fit.predict(X_test_dtm)
    except Exception:
        fit = classifier.fit(X_train_clean, y_train)
        pred = fit.predict(X_test_dtm)
    prediction.append(pred)
    accuracy = accuracy_score(pred,y_test)
    Accuracy.append(accuracy)
    Model.append(c['label'])
    print('Accuracy of '+c['label']+' is '+str(accuracy))


Index = [1,2,3,4,5]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')
plt.show()

for p in prediction:
    true_negative, false_positive, false_negative , true_positive = confusion_matrix(y_test,p).ravel()
    print(true_negative, false_positive, false_negative , true_positive)


for i,p in enumerate(prediction):
    cm = confusion_matrix(y_test, p, labels=fit.classes_)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format='g')
    disp.ax_.set_title(Classifiers[i]['label'])

plt.show()

for i,p in enumerate(prediction):
    prec_score = precision_score(y_test, p, average='weighted')
    print(f"The precision score of {Classifiers[i]['label']} is {prec_score}")

for i,p in enumerate(prediction):
    recall = recall_score(y_test, p, average='weighted')
    print(f"The recall score of {Classifiers[i]['label']} is {recall}")

for i,p in enumerate(prediction):
    f1 = f1_score(y_test, p, average='weighted')
    print(f"The F1 score of {Classifiers[i]['label']} is {f1}")

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
plt.show()   # Display


pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfid', TfidfTransformer()),
                 ('model', Classifiers[1]['model'])])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)


#sample clickbait from the dataset
test_sample1=['The New Star Wars: The Force Awakens Trailer Is Here To Give You Chills']

#sample not a clickbait from the dataset
test_sample2 = ['Scientology defector arrested after attempting to leave organization']

#made up sample, expected to be clickbait
test_sample3 =['Hurry grab our promo now!']

test_sample4=['The rights you might not realize you have - Shannon Odell']

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
exp = explainer.explain_instance(X_test.iloc[idx], pipe.predict_proba)
exp.save_to_file('lime.html')
#JONAS ist ein hella süssie