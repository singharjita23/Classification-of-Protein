import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import csv,os,re,sys,codecs
import joblib,  statistics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

from datetime import datetime

def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, '%d-%m-%Y')
        return True
    except ValueError:
        return False

##################################################################################################################################################################
#          PREPROCESSING FUNCTION 
##################################################################################################################################################################
def preprocess_csv(input_file_path, output_file_path):
    with open(input_file_path, 'r', newline='', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
            csv_reader = csv.reader(input_file)
            csv_writer = csv.writer(output_file)

            for row in csv_reader:
                if len(row) > 14:
                    corrected_row = row[:14]
                    csv_writer.writerow(corrected_row)
                else:
                    csv_writer.writerow(row)

    # print("CSV file processed and fixed successfully.")
    df=pd.read_csv(output_file_path)

    # CONVERTING THE FILE TO A DATAFRANE
    df=pd.DataFrame(df)

    ##################################################################################################################################################################
    # FIXING THE ALIGNMENT ISSUE
    ##################################################################################################################################################################
    df.insert(0, 'miscellaneous', None)
    column_name='structureId'
    for i in range(3):
        for index, row in df.iterrows():
             if len(df.loc[index, column_name])== 4 or is_numeric(df.loc[index, column_name])==True:
                 pass
             else:
                 temp=df.iloc[index, 1:].values
                 temp2=np.append(temp,np.nan)
                 df.iloc[index, 0:]= temp2
    column_name='experimentalTechnique'
    for index, row in df.iterrows():
         if len(df.loc[index, column_name])== 4:
             temp=df.iloc[index, 1:].values
             temp2=np.append(temp,np.nan)
             df.iloc[index, 0:]= temp2
         else:
             pass
    ##################################################################################################################################################################     
    # ABSURD VALUES IN THE PH COLUMN
    ##################################################################################################################################################################
    value_to_check = 'M Tris/HCl pH 7.0 0.05 M NaCl 0.03 M Lithiumsulfat'
    for index, row in df.iterrows():
        if row["phValue"] == value_to_check:
            temp=df.iloc[index, 12:].values
            temp=df.iloc[index, 13:].values
            temp2=np.append(temp,np.nan)
            df.iloc[index, 12:]= temp2
    value_to_check = '724'
    for index, row in df.iterrows():
        if row["phValue"] == value_to_check:
          df.at[index, "phValue"] = np.nan
    value_to_check = '100'
    for index, row in df.iterrows():
        if row["phValue"] == value_to_check:
          df.at[index, "phValue"] = np.nan

    ##################################################################################################################################################################
    # ONE HOT ENCODING FOR 'experimentalTechnique'
    ##################################################################################################################################################################
    df_exp = df['experimentalTechnique'].str.split(', ', expand=True)
    df_exp = pd.DataFrame(df_exp)

    df_exp_encoded = pd.get_dummies(df_exp, prefix='', prefix_sep='')

    df_exp_encoded.columns = df_exp_encoded.columns.map(''.join)

    duplicated_columns = df_exp_encoded.columns[df_exp_encoded.columns.duplicated()]

    df_combined = pd.DataFrame()
    for col in duplicated_columns:
        columns_with_same_name = [c for c in df_exp_encoded.columns if c.startswith(col)]
        df_combined[col] = df_exp_encoded[columns_with_same_name].max(axis=1)

    df_exp_encoded = df_exp_encoded.drop(duplicated_columns, axis=1)
    df_combined = pd.concat([df_exp_encoded, df_combined], axis=1)

    df = df.drop('experimentalTechnique', axis=1)
    df = pd.concat([df, df_combined], axis=1)

    ##################################################################################################################################################################
    # IMPUTING 'macromoleculeType' WITH MODE 'Protein'
    ##################################################################################################################################################################
    mode_value = df['macromoleculeType'].mode()[0]
    df['macromoleculeType'] = df['macromoleculeType'].fillna(mode_value)

    ##################################################################################################################################################################
    # ONE HOT ENCODING FOR 'macromoleculeType'
    ##################################################################################################################################################################
    df_macro = df['macromoleculeType'].str.split('#', expand=True)
    df_macro = pd.DataFrame(df_macro)

    df_macro_encoded = pd.get_dummies(df_macro, prefix='', prefix_sep='')
    df_macro_encoded.columns = df_macro_encoded.columns.map(''.join)

    duplicated_columns = df_macro_encoded.columns[df_macro_encoded.columns.duplicated()]
    df_combined_macro = pd.DataFrame()

    for col in duplicated_columns:
        columns_with_same_name = [c for c in df_macro_encoded.columns if c.endswith(col)]
        df_combined_macro[col] = df_macro_encoded[columns_with_same_name].max(axis=1)

    df_macro_encoded = df_macro_encoded.drop(duplicated_columns, axis=1)
    df_combined_macro = pd.concat([df_macro_encoded, df_combined_macro], axis=1)
    
    df = df.drop('macromoleculeType', axis=1)
    df = pd.concat([df, df_combined_macro], axis=1)

    ##################################################################################################################################################################
    # DROPPING COLUMNS
    ##################################################################################################################################################################
    # Dropping 'CRYO-ELECTRON MICROSCOPY','ELECTRON DIFFRACTION' as they do not appear in the test data
    columns_to_drop = ['miscellaneous', 'structureId', 'crystallizationMethod', 'pdbxDetails', 'Unnamed: 13', 'publicationYear', 'CRYO-ELECTRON MICROSCOPY','ELECTRON DIFFRACTION']
    df= df.drop(columns=columns_to_drop, errors='ignore')

    ##################################################################################################################################################################
    # CONVERTING TO FLOAT DATATYPE
    ##################################################################################################################################################################
    df['residueCount'] = df['residueCount'].astype(float)
    df['resolution'] = df['resolution'].astype(float)
    df['structureMolecularWeight'] = df['structureMolecularWeight'].astype(float)
    df['crystallizationTempK'] = df['crystallizationTempK'].astype(float)
    df['densityMatthews'] = df['densityMatthews'].astype(float)
    df['densityPercentSol'] = df['densityPercentSol'].astype(float)
    df['phValue'] = df['phValue'].astype(float)

    ##################################################################################################################################################################
    # IMPUTATION
    ##################################################################################################################################################################
    df['phValue']=df['phValue'].fillna(df['phValue'].mean())
    df['resolution']=df['resolution'].fillna(df['resolution'].mean())
    df['crystallizationTempK']=df['crystallizationTempK'].fillna(df['crystallizationTempK'].mean())
    df['densityMatthews']=df['densityMatthews'].fillna(df['densityMatthews'].mean())
    df['densityPercentSol']=df['densityPercentSol'].fillna(df['densityPercentSol'].mean())
    return df

##################################################################################################################################################################
# LOADING TRAINING DATA
##################################################################################################################################################################
input_file_path = r"C:\Users\siddh\OneDrive\Desktop\ML project\protein_trn_data.csv"
output_file_path = 'protein_trn_data_new.csv'
# PREPROCESSING TRAINING DATA
train_df = preprocess_csv(input_file_path, output_file_path)

##################################################################################################################################################################
# LOADING TEST DATA
##################################################################################################################################################################
input_file_path = r"C:\Users\siddh\OneDrive\Desktop\ML project\protein_tst_data.csv"
output_file_path = 'protein_tst_data_new.csv'
# PREPROCESSING TEST DATA
test_df = preprocess_csv(input_file_path, output_file_path)

##################################################################################################################################################################
# LOADING TRAINING LABELS AND FORMATTING THEM
##################################################################################################################################################################
labels=pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\ML project\protein_trn_class_labels.csv", header=None, names=['Datapoints', 'CLASS'])
labels=pd.DataFrame(labels)
# Updating the class labels
labels['CLASS'] = labels['CLASS'].str.upper()
labels['CLASS']= labels['CLASS'].str.replace('-', ' ')

train_df['CLASS'] = labels['CLASS']
X=train_df.drop(['CLASS'], axis=1)
y=train_df['CLASS']

warnings.filterwarnings("ignore", category=UserWarning)

##################################################################################################################################################################
# TRAIN TEST SPLIT
##################################################################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

columns_before_scaling = X_train.columns.tolist()
# Print column names before scaling for debugging
# print("Columns before scaling in training data:", columns_before_scaling)
# print("Columns in test data:", test_df.columns.tolist())
if X_train.shape[1] != test_df.shape[1] or not set(test_df.columns) == set(columns_before_scaling):
    raise ValueError("Test data columns do not match training data columns.")
if not set(test_df.columns) == set(columns_before_scaling):
    raise ValueError("Test data columns do not match training data columns.")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_df=test_df[columns_before_scaling]
# Print column names before scaling for debugging
# print("Columns before scaling in training data:", columns_before_scaling)
# print("Columns in test data:", test_df.columns.tolist())
# test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=columns_before_scaling)
test_df_scaled=scaler.transform(test_df)
X_test_scaled = scaler.transform(X_test)

# ##################################################################################################################################################################
# # Decision Tree classifier
# ##################################################################################################################################################################
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
test_pred = clf.predict(test_df_scaled)

predictions_df = pd.DataFrame(data=test_pred, columns=['Predicted_CLASS'])
predictions_df.to_csv('predictions.csv', index=False)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Decision Tree classifier')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print(40*'*')

# ##################################################################################################################################################################
# # Logistic regression model
# ##################################################################################################################################################################
# logreg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# logreg_model.fit(X_train_scaled, y_train)
# y_pred = logreg_model.predict(X_test_scaled)

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print('Logistic regression model')
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print(40*'*')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ##################################################################################################################################################################
# # SVM classifier
# ##################################################################################################################################################################
# svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
# svm_classifier.fit(X_train_scaled, y_train)
# y_pred = svm_classifier.predict(X_test_scaled)

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print("SVM")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print(40*'*')

# ##################################################################################################################################################################
# # KNN classifier
# ##################################################################################################################################################################
# knn_classifier = KNeighborsClassifier(n_neighbors=5)
# knn_classifier.fit(X_train_scaled, y_train)
# y_pred = knn_classifier.predict(X_test_scaled)

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print("KNN")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print(40*'*')

# ##################################################################################################################################################################
# # MULTINOMIAL NAIVE BAYES
# ##################################################################################################################################################################
# scaler =MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# naive_bayes_classifier = MultinomialNB()
# naive_bayes_classifier.fit(X_train_scaled, y_train)
# y_pred = naive_bayes_classifier.predict(X_test_scaled)

# # Evaluation
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print("Naive Bayes")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print(40*'*')

# ##################################################################################################################################################################
# # MIN MAX SCALING + DECISION TREE CLASSIFIER
# ##################################################################################################################################################################
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# scaler =MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # DecisionTreeClassifier + MinMaxScaler
# clf = DecisionTreeClassifier()
# clf.fit(X_train_scaled, y_train)
# y_pred = clf.predict(X_test_scaled)

# # Evaluate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print("Naive Bayes")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
# print(40*'*')

# ##################################################################################################################################################################
# # GRID SEARCH
# ##################################################################################################################################################################
# class classification():
#      def __init__(self, clf_opt='lr',no_of_selected_features=None):
#         self.clf_opt=clf_opt
#         self.no_of_selected_features=no_of_selected_features
#         if self.no_of_selected_features!=None:
#             self.no_of_selected_features=int(self.no_of_selected_features)

# # Selection of classifiers
#      def classification_pipeline(self):
#     # Decision Tree
#         if self.clf_opt=='dt':
#             print('\n\t### Training Decision Tree Classifier ### \n')
#             clf = DecisionTreeClassifier(random_state=40)
#             clf_parameters = {
#             'clf__criterion':('gini', 'entropy'),
#             'clf__max_features':('auto', 'sqrt', 'log2'),
#             'clf__max_depth':(10,40,45,60),
#             'clf__ccp_alpha':(0.009,0.01,0.1),
#             }
#     # Logistic Regression
#         elif self.clf_opt=='lr':
#             print('\n\t### Training Logistic Regression Classifier ### \n')
#             clf = LogisticRegression(solver='liblinear',class_weight='balanced')
#             clf_parameters = {
#             'clf__random_state':(0,10),
#             }
#     # Linear SVC
#         elif self.clf_opt=='ls':
#             print('\n\t### Training Linear SVC Classifier ### \n')
#             clf = svm.LinearSVC(class_weight='balanced')
#             clf_parameters = {
#             'clf__C':(0.1,1,100),
#             }
#     # Multinomial Naive Bayes
#         elif self.clf_opt=='nb':
#             print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
#             clf = MultinomialNB(fit_prior=True, class_prior=None)
#             clf_parameters = {
#             'clf__alpha':(0,1),
#             }
#     # Random Forest
#         elif self.clf_opt=='rf':
#             print('\n\t ### Training Random Forest Classifier ### \n')
#             clf = RandomForestClassifier(max_features=None,class_weight='balanced')
#             clf_parameters = {
#             'clf__criterion':('entropy','gini'),
#             'clf__n_estimators':(30,50,100),
#             'clf__max_depth':(10,30,50,100),
#             }
#     # Support Vector Machine
#         elif self.clf_opt=='svm':
#             print('\n\t### Training SVM Classifier ### \n')
#             clf = svm.SVC(class_weight='balanced',probability=True)
#             clf_parameters = {
#             'clf__C':(0.1,1,100),
#             'clf__kernel':('linear','rbf','poly','sigmoid'),
#             }
#         else:
#             print('Select a valid classifier \n')
#             sys.exit(0)
#         return clf,clf_parameters

# # Statistics of individual classes
#      def get_class_statistics(self,y):
#         class_statistics=Counter(y)
#         print('\n Class \t\t Number of Instances \n')
#         for item in list(class_statistics.keys()):
#             print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))

# # Load the data
#      def get_data(self):

#         self.get_class_statistics(y)
#         # Training and Test Split
#         training_data, validation_data, training_cat, validation_cat = train_test_split(X, y,
#                                                test_size=0.05, random_state=42,stratify=y)

#         return training_data, validation_data, training_cat, validation_cat

# # Classification using the Gold Statndard after creating it from the raw text
#      def classification(self):
#         training_data, validation_data, training_cat, validation_cat=self.get_data()

#         clf,clf_parameters=self.classification_pipeline()
#         pipeline = Pipeline([
#                     ('scaler', MinMaxScaler()),
#                     ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features)),
#                     ('clf', clf),])
#         grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_macro',cv=5)
#         grid.fit(training_data,training_cat)
#         clf= grid.best_estimator_
#         print('\n\n The best set of parameters of the pipiline are: ')
#         print(clf)
#         joblib.dump(clf, 'training_model.joblib')
#         predicted=clf.predict(validation_data)

#     # Evaluation
#         class_names=list(Counter(validation_cat).keys())
#         class_names = [str(x) for x in class_names]
#     # Evaluation
#         print('\n *************** Confusion Matrix ***************  \n')
#         print (confusion_matrix(validation_cat, predicted))

#         class_names=list(Counter(validation_cat).keys())
#         class_names = [str(x) for x in list(Counter(validation_cat).keys())]

#         # Classification report
#         print('\n ##### Classification Report ##### \n')
#         print(classification_report(validation_cat, predicted, target_names=class_names))

#         pr=precision_score(validation_cat, predicted, average='macro')
#         print ('\n Precision:\t'+str(pr))

#         rl=recall_score(validation_cat, predicted, average='macro')
#         print ('\n Recall:\t'+str(rl))

#         fm=f1_score(validation_cat, predicted, average='macro')
#         print ('\n F1-Score:\t'+str(fm))

# obj=classification(clf_opt='dt', no_of_selected_features=25)
# obj.classification()
# obj2=classification(clf_opt='svm', no_of_selected_features=25)
# obj2.classification()