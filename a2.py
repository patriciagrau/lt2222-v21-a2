import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import nltk
import string
import sklearn

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    """Lemmatizes and lowercases the third element of a file.

    Given an input file, it separates the elements of each of
    its lines and lowercases and lemmatizes the third element,
    a token in a sentence. Punctuation is removed.

    Args:
        inputfile: An input file with five columns: an index,
        the sentence the token is in, the token itself, a POS
        tag and a Named Entity classificator.

    Returns:
        A list of lists containing the same lines lowercased
        and lemmatized. In this version, no punctuation has 
        been kept.
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Preparing the tokens that it should ignore
    punct = set(string.punctuation)
    others = set(['""""', '``', '´´'])
    checking_against = punct.union(others)
    
    # Lemmatizing and lowercasing
    new = []
    for line in inputfile.readlines()[1:]: # e.g. '19\t1.0\ttroops\tNNS\tO\n'
        line = line.replace('\n', '')
        sep = line.split('\t') # e.g. ['19', '1.0', 'troops', 'NNS', 'O']
        sep[2] = sep[2].lower()
        if sep[2] not in checking_against:
            if sep[3].startswith('N'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'n')
            elif sep[3].startswith('A'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'a')
            elif sep[3].startswith('V'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'v')
            elif sep[3].startswith('R'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'r')
            else:
                sep[2] = sep[2].lower()
            new.append(sep)

    return new

# Alternative function for Part 1
def preprocess2(inputfile):
    """Lemmatizes and lowercases the third element of a file.

    Given an input file, it separates the elements of each of
    its lines and lowercases and lemmatizes the third element,
    a token in a sentence. Stopwords and punctuation are re-
    moved.

    Args:
        inputfile: An input file with five columns: an index,
        the sentence the token is in, the token itself, a POS
        tag and a Named Entity classificator.

    Returns:
        A list of lists containing the same lines lowercased
        and lemmatized. In this version, no stopwords or punc-
        tuation has been kept.
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    from nltk.corpus import stopwords
    
    # Preparing the tokens that it should ignore
    stops = set(stopwords.words('english'))
    punct = set(string.punctuation)
    others = set(['""""', '``', '´´'])
    checking_against = stops.union(punct, others)
    
    # Lemmatizing and lowercasing
    new = []
    for line in inputfile.readlines()[1:]: # e.g. '19\t1.0\ttroops\tNNS\tO\n'
        line = line.replace('\n', '')
        sep = line.split('\t') # e.g. ['19', '1.0', 'troops', 'NNS', 'O']
        sep[2] = sep[2].lower()
        if sep[2] not in checking_against:
            if sep[3].startswith('N'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'n')
            elif sep[3].startswith('A'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'a')
            elif sep[3].startswith('V'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'v')
            elif sep[3].startswith('R'):
                sep[2] = lemmatizer.lemmatize(sep[2], pos = 'r')
            else:
                sep[2] = sep[2].lower()
            new.append(sep)

    return new


# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)


def create_instances(data):
    """
    Creates the context of a Named Entity

    Given the result of preprocess, it looks for named entities
    and returns the type of NE and its context: five tokens be-
    fore and five tokens after. Other NE are not considered.

    Args:
        data: the result of the function preprocess (a list of
        lists).
    
    Returns:
        instances: a list of instances of a class containing the 
        type of NE and its context (10 words arround it).
    """
    instances = []
    max_size = len(data)
    for count, line in enumerate(data):
        features = []
        if line[-1].startswith('B'):
            neclass = line[-1].split('-')[1]
            current_sent = int(float(line[1]))
            
            # Getting 5 previous words
            m = 6
            n = 1 
            while n < m:
                if (count-n) < 0: # Taking into account the first five elements
                    for i in reversed(range(1, 6)):
                        if '<s'+str(i)+'>' not in features:
                            features.append('<s'+str(i)+'>')
                            break
                else:
                    if current_sent == int(float(data[count-n][1])):
                        if data[count-n][-1] == 'O':
                            features.append(data[count-n][2])
                        else:
                            m+=1  # Adding one to the range if there are NE in the surroundings
                    else:
                        for i in reversed(range(1, 6)):
                            if '<s'+str(i)+'>' not in features:
                                features.append('<s'+str(i)+'>')
                                break
                n+=1

            # Getting 5 following words
            p = 6
            n = 1 
            while n < p:
                if (count+n) >= max_size: # Taking into account the last five elements
                    for i in reversed(range(1, 6)):
                        if '<e'+str(i)+'>' not in features:
                            features.append('<e'+str(i)+'>')
                            break
                else:
                    if current_sent == int(float(data[count+n][1])):
                        if data[count+n][-1] == 'O':
                            features.append(data[count+n][2])
                        else:
                            p+= 1 # Adding one to the range if there are NE in the surroundings
                    else:
                        for i in reversed(range(1, 6)):
                            if '<e'+str(i)+'>' not in features:
                                features.append('<e'+str(i)+'>')
                                break
                n+=1
            
            instances.append(Instance(neclass, features))

    return instances


# Code for part 3
def create_table(instances):
    """
    Creates DataFrame of counts of features

    Given the result of create_instances, creates a DataFrame 
    containing the counts of the words that appear as the fea-
    ture of each instance of NE.

    Args:
        instances: the result of the function create_instances
        (a list of instances of a class).
    
    Returns:
        df: a pandas DataFrame the counts of the words that appear 
        as the feature of each instance of NE, as well as the type
        of NE.
    """

    # Getting the words that will be each column
    context = []
    classes = []
    for each in instances:
        classes.append(each.neclass)
        for word in each.features:
            if word not in context:
                context.append(word)
    
    # Creating an empty matrix
    a_lot_of_zeros = np.zeros((len(instances), len(context)))
    df = pd.DataFrame(data=a_lot_of_zeros, columns=context)

    # Filling the matrix
    row_index = 0
    for each in instances:
        for word in each.features:
            df.loc[row_index, word] += 1
        row_index += 1

    # Changing the name of the columns to integers
    new_index = list(range(len(context)))
    df.columns = new_index
    
    # Adding the column with the class of NE 
    df['class'] = classes
    cols = list(df.columns.values)
    df = df[[cols[-1]] + cols[0:-2]]

    return df

def ttsplit(bigdf):
    """
    Splits data into training and testing data

    Splits the NE data (classes and context) into training 
    (80%) and testing (20%) data with sklearn_test_split.

    Args:
        bigdf: a pandas DataFrame the counts of the words that 
        appear as the feature of each instance of NE, as well 
        as the type of NE.
    Returns:
        train_X: training data of the matrix of the context
            of instances of NE.
        train_y: NE correct labels of the training data.
        test_X: testing data of the matrix of the context
            of instances of NE.
        test_y: NE correct labels of the testing data.
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(bigdf, test_size=0.2)

    train_y = train['class']
    train_y = train_y.reset_index(drop=True)

    train_X = train.drop(columns = ['class'])
    train_X = train_X.reset_index(drop=True)
    
    test_y = test['class']
    test_y = test_y.reset_index(drop=True)
    
    test_X = test.drop(columns = ['class'])
    test_X = test_X.reset_index(drop=True)

    return train_X, train_y, test_X, test_y

# Code for part 5
def confusion_matrix(truth, predictions):
    """
    Returns a confusion matrix (from sklearn)

    Args:
        truth: the true values
        predictions: the predicted values

    """
    from sklearn.metrics import confusion_matrix
    l = list(dict.fromkeys(truth))
    l.sort()
    numpy_data = confusion_matrix(truth, predictions, labels = l)
    df = pd.DataFrame(data=numpy_data, index=l, columns=l)
    return df

# Code for bonus part B
def bonusb(filename):
    pass
