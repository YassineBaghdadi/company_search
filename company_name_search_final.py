''' 
Company Longname Search
Notebook performs the splitting of bios based on sentences containing the company name. 
Requisite files: 

- CSV containing all corporate bios and company names. In notebook below, 
  you will be able to specify the path to the CSV, the name of the column 
  containing the bios, and the name of the column containing company names. 
  The variables to reset are INPUT_PATH, BIO_COL, COMPANY_COL. 
- "all_prefix.json" - JSON file containing common courtesies
- "common_company_endings.json" - JSON file containing common company suffixes
- "common_suffixes.json" - JSON file containing an extended list of company suffixes 
(this one is used by the search function, not the sentence segmentation). 
'''
# ignore python warning
import platform
import warnings

from PyQt5.QtCore import QThread, QObject
from PyQt5.QtWidgets import QHeaderView
from plyer import notification

warnings.simplefilter(action = "ignore", category = UserWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning, append = True)

# load dependencies 
import pandas as pd 
import numpy as np 
import spacy 
import re
import json
from tqdm import tqdm
from collections import Counter
from ast import literal_eval
# these are the partial matchers 
# using Levenshtein distance 
from fuzzywuzzy.fuzz import partial_ratio
from fuzzywuzzy.process import extract

from PyQt5 import QtGui, QtCore, QtWidgets, uic
import os, sys
from subprocess import Popen

#list of common prefixes (courtesies)
PREFIX_LOC = "./all_prefix.json"

#list of extremely common company endings (Inc., LLC., etc.)
COMMON_ENDINGS_LOC = "./common_company_endings.json"

with open(PREFIX_LOC, "r+") as f:
    
    prefixes = json.load(f)

with open(COMMON_ENDINGS_LOC, "r+") as f:

    endings = json.load(f)
    
prefixes += ["He", "She"]



def the_golden_rule(doc, pre = prefixes, post = endings):
    '''
    Using the spaCy dependency based sentencizer that comes with with the
    "en_core_web_sm" model, this code adds additional rules to the pipeline
    to try to account for common errors in sentence tokenization for this
    body of work. This is not a lightweight implementation: It requires 
    most of the spaCy NLP pipeline, except the Named Entity Recognition. 
    But it's highly effective.

    Please see the attached doc I sent for an explanation of each rule
    '''
    periods_removed = [i.replace(".", "") for i in pre]
    
    for token in doc[:-1]:
        
        #RULE A: BLANKET CAPITALIZATION RULE
        if token.text[0].islower():
            
            doc[token.i].is_sent_start = False

        #RULE B: BLANKET PUNCTUATION RULE  
        if token.i >= 1:
            
            if doc[token.i - 1].text[-1] != ".":
                
                doc[token.i].is_sent_start = False
        
        #RULE C: LOWERCASE PREPOSITION RULE
        if token.pos_ == "ADP":
            
            if token.text[0].islower():
            
                doc[token.i].is_sent_start = False

                doc[token.i+1].is_sent_start = False
                
        #RULE D: THE COORDINATING CONJUNCTION RULE
        elif token.pos_ == "CCONJ":
            
            doc[token.i+1].is_sent_start = False
            
            doc[token.i].is_sent_start = False
            
        
        #RULE E: THE ADVERB RULE
        elif token.pos_ == "ADV": 
            
            if token.text[0].isupper() and doc[token.i + 1].pos_ == "ADP":
                
                doc[token.i].is_sent_start = True
        
        #RULE F: THE COMMON TITLES RULE
        elif token.text in pre and \
        doc[token.i -1].pos_ in ["PROPN", "NOUN"]:
            
            if doc[token.i -1].text[-1] == ".":
            
                doc[token.i].is_sent_start = True
                
        #RULE G: COMMON TITLES, MISSING PERIODS
        elif token.text in periods_removed and doc[token.i + 1].pos_ == "PUNCT":
            
            if token.i+2 < len(doc):
            
                doc[token.i+2].is_sent_start = False

        #RULE H: THE INC RULE
        elif token.text in post:

            doc[token.i].is_sent_start = False
                
        #RULE I: THE "THE" RULE
        elif token.pos_ == "DET" and token.text[0].isupper():
            
            if token.i -1 >=0:
                
                if doc[token.i - 1].text[-1] == ".":
                    
                    doc[token.i].is_sent_start = True

        #RULE J: THE PUNCTUATION START RULE
        elif token.pos_ == "PUNCT":
            
            if token.text not in [".", ")"]:
            
                doc[token.i].is_sent_start = False
                
                doc[token.i + 1].is_sent_start = False
            
    return(doc)

def build_nlp_object(disable_ner = True):
    '''
    Instantiate spacy NLP pipeline, add custom rule to pipeline.
    '''
    if disable_ner:
    
      nlp = spacy.load("en_core_web_sm", disable = ["ner"])

    else:

      nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe(the_golden_rule, before = "parser")

    return(nlp)

# this is punctuation to strip
# we are going to stripping everything that is not a hyphen or an ampersand 
punct = "!#$%\'()*+,./:;<=>?@[\\]^_`{|}~"

# function to remove text and 
def preprocess_text(x, punct_list = punct):

  return x.translate(str.maketrans("", "", punct_list)).lower()

with open("./common_suffixes.json", "r+") as f:

  common_suffixes = json.load(f)

# list of lowercase prefixes (used in the end of the search)
lower_pre = [preprocess_text(pre) for pre in prefixes]

# for complex cases, we will use POS tagger 
nlp = build_nlp_object()

# dict object of common suffix abbreviations in this text 
abbrevs = {
    
    "aktieselskab" : ["as"],
    "aktiengesellschaft" : ["ag"],
    "corporation" : ["corp", "corps"],
    "bancorporation" : ["bancorp"],
    "limited" : ["ltd"],
    "company" : ["co"],
    "incorporated" : ["inc"]
}

# location of input CSV
INPUT_PATH = ""

# location of output CSV (or excel) note: boolean flag will add suffix 
OUTPUT_PATH = ""
# OUTPUT_PATH = "divided_company_bios"

# the name of the biography column/ company name column
# has not been uniform between all
# datasets, so it can be updated here without changing the code 
BIO_COL = "persontext"
COMPANY_COL = "longname"

#these are the names of the output columns for the function
OUTPUT_COLS = [
    "tokenized_sentences",
    "sentences_with_longname",
    "sentences_without_longname",
    "total_sentences_containing",
    "total_occurences"
]


def master(
    input_path = INPUT_PATH, 
    output_path = OUTPUT_PATH,
    bio_col = BIO_COL, 
    company_col = COMPANY_COL, 
    output_cols = OUTPUT_COLS,
    to_csv = True,
    nlp_object = nlp
    ):

    print("Reading in Corporate Bios...")
    print()

    df = pd.read_csv(input_path)
    #progress bar for pandas .apply function,
    #useful for keeping track of longer executions 
    tqdm.pandas(desc = "company_search")

    print("Segmenting into sentences and searching for company names...")
    print()
    #call progress_apply (leveraging the tqdm )
    df[output_cols] = \
        df.progress_apply(
            lambda x: process_bio(x, bio_col, company_col, nlp_object),
            axis = 1,
            result_type = "expand"
        )
    
    #if CSV desired, write to csv
    if to_csv:

        output_path += ".csv"

        print("Writing to CSV...")

        df.to_csv(output_path, index = False)

        print("Successfully written")
    #otherwise, write to excel
    else:

        output_path += ".xlsx"

        print("Writing to Excel...")

        df.to_excel(output_path, header = True, index = False)

        print("Successfully written")

    notification.notify(title='Operation Completed successfully',
                        message=f'The Output file saved on {output_path}', timeout=5)
    return output_path

def process_bio(x, bio_col, company_col, nlp_object):
  '''
  Worker function that processes each bio. The process is as follows:
    1. Remove all whitespaces from company name and bio text
    2. Remove all punctuation (except for "&" and "-", these are frequent
       partial match offenders) from the company name 
    3. Segment bio into sentences 
    4. Iterate through all sentences:
      I. Check for exact matches. If there are exact matches, count the number
         of matches and add the sentence to the appropriate data structure
      II. If there is not an exact match, calculate partial match score using 
          Levenshtein Distance. If the score is 97, this means that only one character
          is off and we are going to assume this is a typo. This catches more mistakes
          than it will mistakenly flag. 
          If the score is over a threshold (70), then we will further 
          inspect the sentence:
            a. First, check for "-" and "&" in the sentence and the name, and if they are present,
               use the extract function to determine which character combination is present.
            b. Second, check for commonly abbreviated words and replace them with the 
               abbreviated (or unabbreviated) version that yields the highest match score
            c. Recalculate the score. If this produces an exact match, a score of 100,
               add the sentence to the appropriate data structure and update counts. 
            d. If this still did not produce an exact match, locate all occurences of the 
               first word in the company name in the sentence (matching in chronological
               order is critical, thus if the words are not in the correct order we know it
               is not a match)
            e. Check if all but the last word matches. If the last word of the company name
               is a common suffix and the next word of the potential match is not a proper
               noun, this is a match.
  '''
  text, company_name = " ".join(x[bio_col].split()), " ".join(x[company_col].split())

  proc_name = preprocess_text(company_name)

  sents = [sent.text for sent in nlp(text).sents]

  #initialize data structures / accumulators
  related, unrelated, total, individual = \
      [], [], 0, 0
  
  for sent in sents:

    proc_sent = preprocess_text(sent)

    exact_matches = re.findall(proc_name, proc_sent)

    if exact_matches:

      individual += len(exact_matches)

      total += 1

      related.append(sent)

    else:

      score = partial_ratio(proc_name, proc_sent)

      if score >= 97:

        individual += 1

        total += 1

        related.append(sent)

      elif score >= 70:

        if "&" in proc_name or "&" in proc_sent or "-" in proc_name:

          proc_name = process_compound_punctuation(proc_name, proc_sent, score)

        proc_name = replace_abbreviation(proc_sent, proc_name)

        score = partial_ratio(proc_name, proc_sent)

        if score == 100:

          individual += 1

          total += 1

          related.append(sent)

        else:

          name_words = proc_name.split()

          search_locs = find_starting_points(name_words[0], proc_sent, len(name_words))

          if search_locs:

            n_occurences = 0

            for loc in search_locs:

              res = SEARCH(name_words, loc, nlp_object)

              if res:

                n_occurences += 1

            if n_occurences:

              individual += n_occurences

              total += 1

              related.append(sent)

            else:

              unrelated.append(sent)

          else:

            unrelated.append(sent)

      else:

        unrelated.append(sent)

  related, unrelated = " ".join(related), " ".join(unrelated)

  return [sents, related, unrelated, total, individual]

def SEARCH(name_words, potential_match, nlp_object, suffixes = common_suffixes, prefixes = lower_pre):
  '''
  If the only thing missing is the suffix 
  of the company name and everything else matches, we will proceed as if it is 
  a match. The exception to this case is if the last word of the potential match
  is a proper noun, then this is not considered a match.
    Args:
      name_words[list] : list of individual words in the company name  
      potential_match[list] : list of words in the potential match for the company name 
      nlp_object[spacy.lang] : spacy english language model
      suffixes[list] : list of common suffixes
      prefixes[list] : list of common prefixes
    Returns:
      [boolean] : whether or not this end of case scenario matches
  '''
  # EOS case when the name is at the end of the sentence 
  if len(name_words) > len(potential_match):
    
    if name_words[-1] in suffixes: 

      for i, word in enumerate(potential_match):
        
        if word != name_words[i] and word.replace("s", "", -1) != name_words[i]:

          return False 

  elif len(name_words) == len(potential_match):

    POS = [i.pos_ for i in nlp_object(" ".join(potential_match))]

    for i, (potential, actual) in enumerate(zip(potential_match, name_words)):

      if potential != actual and potential.replace("s", "", -1) != actual:

        if i == len(name_words) - 1:

          if actual in suffixes:

            if POS[i] == "PROPN" and potential not in prefixes:

              return(False)

          else:

            return(False)

        else:

          return(False)

  return True

def find_starting_points(start_word, sentence, n_words):
    '''
    Many company names share common words (Financial, Capital, Trust, etc.)
    Partial string matching does not discriminate between which words are 
    similar, the score for Levenshtein distance is determined by the number
    of characters that would need to be changed to make the substrings match.
    Because of this, it's necessary to find all potential starting positions 
    and check the individual words in order. 
        Args:
            start_word[str] : First word of the company name
            sentence[str] : processeds sentence string 
            n_words[int] : The number of words in the company name 
    '''
    #data structure to hold all matches
    to_search = []
    
    #search for all potential starting points 
    potential_starts = [i.start() for i in re.finditer(start_word, sentence)]

    # iterate through all potential starting points 
    for start in potential_starts:

      to_search.append(sentence[start:].split()[:n_words])
        
    return to_search 

def process_compound_punctuation(name, sentence, score):
  '''
  Common case throwing off the search is the handling of ampersands and 
  hyphens. There are several possible combinations for these particular
  characters, so we will use the fuzzywuzzy extract function to 
  check which case is the best fit.
    Args:
      name[str] : processed company name string with all punctuation except 
                  ampersands and hyphens removed. 
      sentence[str] : processed sentence string with all punctuation except 
                      ampersands and hyphens removed
      score[int] : the score of the partial match
    Returns:
      [str] : processed company name that had the highest Levenshtein distance
              score. 
  '''
  # check ampersands
  if "&" in name:

    name = extract(sentence, [
        name, 
        name.replace("&", "and"),
        name.replace(" & ", "&"),
        name.replace("&", " & ")      
    ],
    scorer = partial_ratio
        
    )[0][0]

  # check hyphen
  if "-" in name:

    if partial_ratio(name.replace("-", " "), sentence) > score:

      name = name.replace("-", " ")

  return name

def replace_abbreviation(sentence, longname, abbrevs = abbrevs):
  '''
  Utility function to swap out and identify abbreviated words.
    Args:
      sentence[str] : full sentence
      longname[str] : company name
      abbrevs[dict] : dictionary object of common abbreviations.
    Returns:
      [str] : the company name with the best match
  '''
  for key in abbrevs.keys():

    if key in longname:
      
      all_possible = [longname]

      for abbr in abbrevs[key]:

        all_possible.append(longname.replace(key, abbr, 1))

      best_match = extract(sentence, all_possible, scorer = partial_ratio)[0][0]

      return best_match

    elif longname.split()[-1] in abbrevs[key]:

      targ = longname.split()[-1]

      all_possible = [longname]

      all_possible.append(longname.replace(targ, key, 1))

      for i in abbrevs[key]:

        if i != targ:

          all_possible.append(longname.replace(targ, i, 1))

      best_match = extract(sentence, all_possible, scorer = partial_ratio)[0][0]

      return best_match

  return longname
DESKTOP = os.path.join(os.path.join(os.environ['USERPROFILE']),
                       'Desktop') if platform.system() == 'Windows' else os.path.join(
    os.path.join(os.path.expanduser('~')), 'Desktop')

class Main(QtWidgets.QWidget):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi( os.path.join(os.path.dirname(__file__), 'ui/main.ui'), self)
        self.browse_.clicked.connect(self.browse_path)
        self.proc.setEnabled(False)
        self.proc.clicked.connect(self.start)
        self.paths_list = []
        self.path_txt.currentTextChanged.connect(self.path_changed)
        self.current_prccess_file = ''
        self.setWindowTitle('SPGI-Focus')
        self.loading = Loading()
        self.change_content(self.loading)

    def browse_path(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(caption='Choose File',
                                                              filter="csv (*.csv)", directory=DESKTOP)[0]
        if self.filename:
            if not QtCore.QFileInfo(self.filename).suffix():
                self.filename += '.csv'
            self.proc.setEnabled(True)
            self.paths_list.insert(0, self.filename)

            self.path_txt.clear()

            self.path_txt.addItems(self.paths_list)
            self.path_txt.setCurrentIndex(0)




    def path_changed(self):
        if self.path_txt.currentText() != self.current_prccess_file:
            self.proc.setEnabled(True)


    def start(self):

        if os.path.isfile(self.path_txt.currentText()):
            self.current_prccess_file = self.path_txt.currentText()
            self.proc.setEnabled(False)
            self.browse_.setEnabled(False)
            self.path_txt.setEnabled(False)
            OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(self.path_txt.currentText())), f'divided_company_bios')

            try:

                oo = master(input_path=self.path_txt.currentText(), output_path=OUTPUT_PATH)
                self.view = View(oo)
                self.change_content(self.view)
                self.proc.setEnabled(True)
                self.browse_.setEnabled(True)
                self.path_txt.setEnabled(True)
            except Exception as e:
                print('#' * 50)
                print(e)
                notification.notify(title='Operation Failed', message=str(e), timeout=5)



        else:
            notification.notify(title='invalid file', message='', timeout=5)

    def change_content(self, wdgt):
        for i in reversed(range(self.contents.count())):
            self.contents.itemAt(i).widget().setParent(None)
        self.contents.addWidget(wdgt)


class Loading(QtWidgets.QWidget):
    def __init__(self, gif='proc.gif'):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui/loading.ui'), self)
        self.gif = QtGui.QMovie(gif)
        self.label.setMovie(self.gif)
        self.gif.start()


class View(QtWidgets.QFrame):
    def __init__(self, out):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui/view.ui'), self)
        self.out = out
        data = pd.read_csv(out)
        table_header = [i for i in data.columns]
        self.table.setColumnCount(len(table_header))
        # self.table.setRowCount(len(data))
        self.table.setHorizontalHeaderLabels(table_header)
        self.table.resizeColumnsToContents()

        for i in range(len(table_header)):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)

        [self.table.removeRow(0) for _ in range(self.table.rowCount())]

        for r in range(len(data)):
            rrr = []
            self.table.insertRow(r)
            for c in range(len([i for i in data.columns])):

                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(data.iloc[r][c])))
                rrr.append(str(data.iloc[r][c]))
            print(rrr)
        self.o_e.clicked.connect(self.open_e)
        self.o_x.clicked.connect(self.open_x)
    
    def open_x(self):
      o = Popen(r''.join(self.out), shell = True)
      
    def open_e(self):
      path = r''.join(self.out)
      Popen(f'explorer /select, "{self.out}"')






if __name__ == "__main__":

    # master()
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec()
