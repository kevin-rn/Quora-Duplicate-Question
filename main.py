import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import re
import json

def pre_process(txt):
    def replace_all(t, dic):
        for i, j in dic.items():
            t = t.replace(i, j)
        return t

    # Set text to lowercase and removing leading and tracing whitespaces
    txt = str(txt).lower().strip()
    # Replace currencies,characters and contractions to be more uniform
    with open('data/replacechars.json', 'r') as JSON:
        json_dict = json.load(JSON)
    text = replace_all(txt, json_dict)
    text = re.sub(r'([0-9]+)000000000', r'\1b', text)
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)
    return text


def reformat(dataframe):
    df2 = dataframe.drop(['qid1', 'qid2'], axis=1)
    df2['question1'] = dataframe['question1'].apply(pre_process)
    df2['question2'] = dataframe['question2'].apply(pre_process)
    return df2

# Loads file into dataframe with optional preprocessing
def load_data(filename, preprocess):
    df = pd.read_csv(filename, index_col='id')
    if preprocess:
        df = reformat(df)
    return df

# Perform (partial) string matching by checking equality of the strings and is one is a substring of the other.
def string_matching():
    df = load_data('data/Test set.csv', False)
    df['is_duplicate'] = ((df['question1'] == df['question2']) | df['question1'].isin(df['question2']) | df['question2'].isin(df['question1'])).astype(int)
    df['is_duplicate'].to_csv('string_match.csv')

# Helper method for extracting features
def word_count(entry):
    q1_set = set(entry['question1'].split(" "))
    q2_set = set(entry['question2'].split(" "))
    common = len(q1_set & q2_set)
    total = (len(q1_set) + len(q2_set))
    shared = round(common/total, 2)
    return common, total, shared

# Random Forest Classifier
def rf_duplicate_questions():
    # training set
    df_train = load_data('data/Development set.csv', True)
    df_train[['common', 'total', 'shared']] = df_train.apply(word_count, axis=1, result_type='expand')
    df_train = df_train.drop(columns=['question1', 'question2'])
    x_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 0].values  # is_duplicated

    # test set
    df_test = load_data('data/Test set.csv', True)
    df_test[['common', 'total', 'shared']] = df_test.apply(word_count, axis=1, result_type='expand')
    df_test = df_test.drop(columns=['question1', 'question2', '?'])
    x_test = df_test.values

    # predict using Random forest
    cf = RandomForestClassifier()
    cf.fit(x_train, y_train)
    df_test['is_duplicate'] = cf.predict(x_test)
    df_test['is_duplicate'].to_csv('random_forrest.csv')


if __name__ == '__main__':
    # string_matching()
    rf_duplicate_questions()