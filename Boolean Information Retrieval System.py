#!/usr/bin/env python
import sys
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

CRAN_COLL = '/Users/iraklis/Datasets/cran/cran.all.1400'
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
    running_idx = -1
    in_text = False
    in_title = False
    id_body_list = {}
    id_title_list = {}
    f = open(cran_file)
    for line in f:
        if line.startswith('.I'):
            in_text = False
            running_idx = int(line[2:])
        elif line.startswith('.A'):
            in_title = False
        elif line.startswith('.W'):
            in_text = True
            line = line[2:]  # Do away with 2-char delimiters on 1st lines
            id_body_list[running_idx] = []
        elif line.startswith('.T'):
            in_title = True
            line = line[2:]
            id_title_list[running_idx] = []
        else:
            pass  # Good practice
        if in_text:
            id_body_list[running_idx] += line.split()
        if in_title:
            id_title_list[running_idx] += line.split()
    f.close()
    # assert len(id_body_list) == len(id_title_list)
    # assert sorted(id_body_list.keys()) == sorted(id_title_list.keys())
    return id_body_list, id_title_list


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
    lst = [t.lower() for t in words]
    # Remove symbols:
    lst = [''.join([c if c not in SYMBOLS else '' for c in t]) for t in lst]
    # Remove words <= 3 characters:
    lst = [t for t in lst if len(t) > 3]
    # Remove stopwords:
    lst = [t for t in lst if t not in stop_words]
    # Stem terms:
    lst = map(stemmer.stem, lst)

    return lst


def create_inv_index(bodies, titles):
    """Create a single inverted index for the dictionaries provided. Treat
    all keywords as if they come from the same field.
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
    inv_ind = {}
    for doc_id in bodies:
        all_terms = bodies[doc_id] + titles[doc_id]
        joint_terms = pre_process(all_terms)
        for t in joint_terms:
            if t not in inv_ind:
                inv_ind[t] = [1, {doc_id: 1}]  # initial entry for term
            else:
                if doc_id not in inv_ind[t][1]:
                    inv_ind[t][1][doc_id] = 1
                else:
                    inv_ind[t][1][doc_id] += 1
                inv_ind[t][0] = len(inv_ind[t][1])  # Update df
    return inv_ind


def load_inv_index(filename=INDEX_FILE):
    """Load an inverted index from the disk. The index is assummed to be stored
    in a text file with one line per keyword. Each line is expected to be
    `eval`ed into a dictionary of the form created by create_inv_index().

    Arguments:
        filename: the path of the inverted index file
    Return:
        a dictionary containing all keyworks and their posting dictionaries
    """
    inv_ind = {}
    f = open(filename)
    for line in f:
        inv_ind.update(eval(line))
    f.close()
    return inv_ind


def write_inv_index(inv_index, outfile=INDEX_FILE):
    """Write the given inverted index in a file.
    Arguments:
        inv_index: an inverted index of the form {'term': [df, {doc_id: tf}]}
        outfile: (str) the path to the file to be created
    """
    f = open(outfile, 'w')
    for t in inv_index:
        f.write({t: inv_index[t]}.__str__() + '\n')
    f.close()


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
    matches = {}
    # Get the posting "lists" for each of the ANDed terms
    postings = {}
    for t in terms:
        if t not in inv_index:
            postings[t] = {}
        else:
            postings[t] = inv_index[t][1]
    # Basic boolean - no scores:
    docid_sets = [set(postings[t].keys()) for t in postings]
    matches = set.intersection(*docid_sets)
    return {(docid, None) for docid in matches}


def eval_disj(conj_results):
    """Evaluate the conjunction results provided, essentially ORing the
    document IDs they contain. In other words the resulting list will have to
    contain all unique document IDs found in the partial result lists.
    Arguments:
        conj_results: results as they return from `eval_conj()`, i.e. of the
        form {(doc_id, score)}, where score can be None for non-ranked
        retrieval.
    Return:
        a set of (docId, score) tuples - You can ignore `score` by substituting
        it with None
    """
    # Basic boolean - no scores:
    return set.union(*conj_results)


def main():
    """Load or create an inverted index. Parse user queries from stdin
    where words on each line are ANDed, while whole lines between them are
    ORed. Match the user query to the Cranfield collection and output matching
    documents as "ID: title", each on its own line, on stdout.
    """

    # If an index file exists load it; otherwise create a new inverted index
    # and write it into a file
    inv_index = None
    if not os.path.isfile(INDEX_FILE):
        doc_kwds, title_kwds = parse_documents()
        inv_index = create_inv_index(doc_kwds, title_kwds)
        write_inv_index(inv_index)
    else:
        inv_index = load_inv_index()

    # Get and evaluate user queries from stdin. Terms on each line should be
    # ANDed, while results between lines should be ORed.
    # The output should be a space-separated list of document IDs. In the case
    # of unranked boolean retrieval they should be sorted by document ID, in
    # the case of ranked solutions they should be reverse-sorted by score
    # (documents with higher scores should appear before documents with lower
    # scores).
    replies = []
    for l in sys.stdin:
        replies.append(eval_conj(inv_index, pre_process(l.split())))
    if len(replies) > 0:
        or_replies = eval_disj(replies)
        print reduce(lambda x, y: str(x) + ' ' + str(y), sorted([x[0] for x in
                     or_replies]), '')
    else:
        print


if __name__ == '__main__':
    main()
