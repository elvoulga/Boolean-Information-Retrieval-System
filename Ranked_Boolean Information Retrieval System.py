#!/usr/bin/env python

from __future__ import print_function
import sys
import re
import itertools
import ast
import os.path
import math
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from itertools import groupby
from operator import itemgetter


CRAN_COLL = '/home/mscuser/Datasets/cran/cran.all.1400'
INDEX_FILE = 'cran.ind'

SYMBOLS = '!@#$%^&*()[]{};\':",.<>/?`~-_=+'


def parse_documents(cran_file=CRAN_COLL):
    """Parse the document body and title fields of the Cranfield collection.
    Arguments:
        cran_file: (str) the path to the Cranfield collection file
    Return:
        (body_kwds, title_kwds): where body_kwds and title_kwds are
        dictionaries of the form {docId: [words]}.
    """
    id_term_list_body = []
    id_term_list_title = [] 
    in_text_body = 0
    in_text_title = 0  
    body_kwds = {}
    title_kwds = {}
    
    # Open the cran file, read the titles and the bodies without the lines with the delimeters and store them in lists
 
    with open(cran_file) as cran_file:
        listoflines = cran_file.read().splitlines()
        for each_line in listoflines:
              if each_line.startswith('.I'):
                  in_text_body = 0
                  in_text_title = 0
                  document_id = int(re.search(r'\d+', each_line).group())
                  continue
              else:
                  if each_line.startswith('.T'):
                      in_text_title = 1
                      continue
              if in_text_title == 1:
                  if each_line.startswith('.A'):
                      in_text_title = 0
                      continue
                  words_title = each_line.split()
                  title_tuples = [(document_id,each_word) for each_word in words_title]
                  for each_title_tuple in title_tuples:
                      id_term_list_title.append(each_title_tuple)
              else:
                  if each_line.startswith('.W'):
                      in_text_body = 1
                      continue
              if in_text_body == 1:
                  if each_line.startswith('.I'):
                      in_text_body = 0
                      continue
                  words_body = each_line.split()
                  body_tuples = [(document_id,each_word) for each_word in words_body]
                  for each_body_tuple in body_tuples:
                      id_term_list_body.append(each_body_tuple)

    # Convert the lists into dictionaries

    for a, b in id_term_list_title:
        title_kwds.setdefault(a, []).append(b)

    for a, b in id_term_list_body:
        body_kwds.setdefault(a, []).append(b)
   
    return (title_kwds, body_kwds)
   
def pre_process(words):
    """Preprocess the list of words provided.
    Arguments:
        words: (list of str) A list of words or terms
    Return:
        a shorter list of pre-processed words
    """
    # Get list of stop-words and instantiate a stemmer:
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Make all lower-case:
    words = map(lambda x:x.lower(),words)
    
    # Remove symbols:
    words = [each_string.strip(SYMBOLS) for each_string in words]
    
    # Remove words <= 3 characters:
    words = [each_string for each_string in words if len(each_string) > 3]
   
    # Remove stopwords:
    words = [each_string for each_string in words if each_string not in stop_words]
  
    # Stem terms:
    words = map(lambda x: (stemmer.stem(x)), words)

    return (words)
 

def create_inv_index(bodies, titles):
    """Create a single inverted index for the dictionaries provided. Treat
    all keywords as if they come from the same field. In the inverted index
    retail document and term frequencies per the form below.
    Arguments:
        bodies: A dictionary of the form {doc_id: [terms]} for the terms found
        in the body (.W) of a document
        titles: A dictionary of the form {doc_id: [terms]} for the terms found
        in the title (.T) of a document
    Return:
        index: a dictionary {docId: [df, postings]}, where postings is a
        dictionary {docId: tf}.
        E.g. {'word': [3, {4: 2, 7: 1, 9: 3}]}
               ^       ^   ^        ^
               term    df  docid    tf
    """
    # Create a joint dictionary with pre-processed terms 
    
    joint_dictionary = {doc_id_titles : lst_titles + lst_bodies for doc_id_titles, lst_titles in titles.items() for doc_id_bodies, lst_bodies in bodies.items() if doc_id_titles == doc_id_bodies}

    # Create a list terms[] with the unique terms of the dictionary 

    terms = []
    for lst in joint_dictionary.values():
        for term in lst:
            terms.append(term)
    terms = set(terms)
    terms = list(terms)
    
    # Create the inverted indexes and call the function that writes them in a file
    
    tf = 0
    dictionary_instance = {}

    for term in terms:
        for number, lst in joint_dictionary.items():
            for word in lst:
                if word == term:
                    tf += 1
                    dictionary_instance[number] = tf
            tf = 0
        df = len(dictionary_instance.keys())
        inv_index = {term : [df, dictionary_instance]}
        dictionary_instance = {}
        write_inv_index(inv_index, INDEX_FILE)

       
def load_inv_index(filename=INDEX_FILE):
    """Load an inverted index from the disk. The index is assummed to be stored
    in a text file with one line per keyword. Each line is expected to be
    `eval`ed into a dictionary of the form created by create_inv_index().

    Arguments:
        filename: the path of the inverted index file
    Return:
        a dictionary containing all keyworks and their posting dictionaries
    """
    whole_dictionary = {}
    
    # Open the file with the inverted indexes and store them in a dictionary

    with open(filename) as index_file:
        lines = index_file.read().splitlines()
        for each_line in lines:
            each_line = ast.literal_eval(each_line)
            whole_dictionary.update(each_line)

    return (whole_dictionary)

def write_inv_index(inv_index, outfile=INDEX_FILE):
    """Write the given inverted index in a file.
    Arguments:
        inv_index: an inverted index of the form {'term': [df, {doc_id: tf}]}
        outfile: (str) the path to the file to be created
    """

    # Open a file in which the inverted indexes are written

    f = open(outfile, 'a')
    f.write(str(inv_index) + '\n')
    


def eval_conj(inv_index, terms):
    """Evaluate the conjunction given in list of terms. In other words, the
    list of terms represent the query `term1 AND term2 AND ...`
    The documents satisfying this query will have to contain ALL terms. 
    Arguments:
        inv_index: an inverted index
        terms: a list of terms of the form [str]
    Return:
        a set of (docId, score) tuples -- You can ignore `score` by
        substituting it with None
    """
    num_ids = []
    tf_idf_list = []
    conj_ids = set()
    total_documents = 1400

    # Create a list num_ids which contains lists of the document ids that every terms appears in

    for word in terms:
        for term, value in inv_index.iteritems():
            if word == term:
                num_ids.append(value[1].keys())
      
    # Create the set conj_ids with the ids that are common for each term 

    if len(num_ids)>=1:
        conj_ids = set(num_ids[0]).intersection(*num_ids[1:]) 
    else:
        conj_ids = set()

    # Calculates the tf_idf for every term for every mutual document id and stores it together with the id as a tuple in a list 

    for word in terms:
        for term, value in inv_index.iteritems():
            if word == term:
                idf = math.log10(total_documents / value[0])
                for oneid in conj_ids:
                    for key, freq in value[1].iteritems():
                        if oneid == key:
                            tf = 1 + math.log10(freq)
                            tf_idf = tf * idf
                            tup = (oneid, tf_idf)
                            tf_idf_list.append(tup)

    # Groups the same ids and then sums the tf_idf scores

    tf_idf_list.sort(key=itemgetter(0)) 
    grouped_by = groupby(tf_idf_list, key=itemgetter(0))
    numbers = [(key, [t[1] for t in items]) for key, items in grouped_by]
    conj_results = [(key, sum(items)) for key, items in numbers]

    return (conj_results)


def eval_disj(conj_results):
    """Evaluate the conjunction results provided, essentially ORing the
    document IDs they contain. In other words the resulting list will have to
    contain all unique document IDs found in the partial result lists.
    Arguments:
        conj_results: results as they return from `eval_conj()`, i.e. of the
        form [(doc_id, score)], where score can be None for non-ranked
        retrieval. 
    Return:
        a set of (docId, score) tuples - You can ignore `score` by substituting
        it with None
    """
    # Flattens the list of lists of tuples which are returned from the eval_conj(s)

    conj_results = [y for x in conj_results for y in x]   
    
    # Groups the same ids and then takes the max tf_idf score

    conj_results.sort(key=itemgetter(0)) 
    grouped_by = groupby(conj_results, key=itemgetter(0))
    numbers = [(key, [t[1] for t in items]) for key, items in grouped_by]
    disj_results = [(key, max(items)) for key, items in numbers]

    return (disj_results)


def main():
    """Load or create an inverted index. Parse user queries from stdin
    where words on each line are ANDed, while whole lines between them are
    ORed. Match the user query to the Cranfield collection and output matching
    documents as "ID: title", each on its own line, on stdout.
    """
    conj_eval = []

    # If an index file exists load it; otherwise create a new inverted index and write it into a file

    if os.path.exists("cran.ind") == False:
        titles, bodies = parse_documents()
        titles_parsed = {docid : pre_process(lst) for docid, lst in titles.items()}
        bodies_parsed = {docid : pre_process(lst) for docid, lst in bodies.items()}
        create_inv_index(bodies_parsed, titles_parsed)

    # Load the inverted index into a dictionary

    whole_dictionary = load_inv_index(INDEX_FILE) 
      
    # Gets and evaluates user queries from stdin. Terms on each line are ANDed, while results between lines are ORed.
    # The output is a space-separated list of, reverse-sorted by score, document IDs.
    
    user_input = sys.stdin.readlines()
    for lst_and_words in user_input:
        lst_and_words = lst_and_words.splitlines()
        lst_and_words = lst_and_words[0].split()
        lst_and_words = pre_process(lst_and_words)
        conj_eval.append(eval_conj(whole_dictionary, lst_and_words))
            
    disj_eval = eval_disj(conj_eval)
    disj_eval.sort(key=itemgetter(1), reverse=True)
    for answer in disj_eval:
        print (str(answer[0]), end=' ')
    print ("\n")
        

if __name__ == '__main__':
    main()
