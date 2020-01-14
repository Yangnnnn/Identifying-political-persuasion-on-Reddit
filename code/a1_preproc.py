import sys
import argparse
import os
import json
import html
import re
import spacy
import string

#global variables
indir = '/Users/chulong/Desktop/CSC401A1/data';
nlp = spacy.load('en', disable=['parser', 'ner'])

"""Helper functions"""
#step1
def remove_newline(comment):
    """Remove all newline characters."""
    comment = comment.replace("\n"," ").replace("\r"," ")
    comment = re.sub(r"\s{2,}"," ",comment)
    return comment
#step2
def html_char(comment):
    """Replace HTML character codes with their ASCII equivalent"""
    comment = html.unescape(comment)
    return comment
#step3
def remove_url(comment):
    """ Remove all URLs"""
    comment = re.sub(r'(https?://|www.)[^\s]*'," ",comment,flags=re.IGNORECASE)
    comment = re.sub(r"\s{2,}"," ",comment)
    return comment
#step4
    
def get_abbreviations_regex():
    """making regex pattern for abbreviations"""
    files = "/Users/chulong/Desktop/CSC401A1/Wordlists/abbrev.english"
    lst=[]
    a=[ ]
    s=""    
    with open(files) as file:
        for i in file:
            lst.append(i.strip())
    
    for i in lst:
        a.append(r'(?<!'+i[:-1]+r')')
    for i in a:
        s=s+i
        
    s=s+r'(?<![\W_])'
    return s+r'\.'


def split_pun(comment):
    """ Split each punctuation into its own token using whitespace"""
    abb_regex = get_abbreviations_regex()
    
    comment = re.sub(r'([\W_]+)',r' \1 ',comment,flags=re.IGNORECASE)
    
    comment = re.sub(r' \' ','\'',comment,flags=re.IGNORECASE)
    
    comment = re.sub(r' \. ','.',comment,flags=re.IGNORECASE)
    
    comment = re.sub(abb_regex,r" . ",comment,flags=re.IGNORECASE)
    
    comment = re.sub(r'\s{2,}'," ",comment,flags=re.IGNORECASE)
    
    return comment

#step5


def clitics(comment):
    """Split clitics using whitespace."""
    comment=re.sub(r'([\w]+)(\'(d|n|ve|re|ll|m|re|s))(\Z|\s)',r'\1 \2\4',comment,flags=(re.IGNORECASE))
    comment=re.sub(r'([\w]+)(\w\'t)(\Z|\s)',r'\1 \2\3',comment,flags=(re.IGNORECASE))
    comment=re.sub(r'(\s?)(t|y)(\'[\w]+)(\Z|\s)',r'\1\2 \3\4',comment,flags=(re.IGNORECASE))
    comment=re.sub(r'([\w]+)(s)(\')(\Z|\s)',r'\1\2 \3\4',comment,flags=(re.IGNORECASE))
    comment = re.sub(r'\s{2,}'," ",comment,flags=re.IGNORECASE)
    return comment
#step6
def tag(comment):
    """Each token is tagged with its part-of-speech using spaCy"""
    words = comment.split()
    doc = spacy.tokens.Doc(nlp.vocab, words)
    doc = nlp.tagger(doc)
    comment=""
    for i in doc:
        comment = comment +" "+ i.text+"/"+i.tag_
    comment=comment[1:]
    return comment
#step7
def stopwords(comment):
    """Remove stopwords"""
    stoplst=[]
    files = "/Users/chulong/Desktop/CSC401A1/Wordlists/StopWords"
    with open(files) as file:
        for i in file:
            stoplst.append(i.strip())    
    s = ""
    s = "|".join(stoplst)
    r = r'(^|\s)('+s +r')/(\S+)' 
    comment=re.sub(r,r'',comment,flags=re.IGNORECASE)
    comment = re.sub(r'\s{2,}'," ",comment,flags=re.IGNORECASE)
    return r
#step8
def lemma(comment):
    """Apply lemmatization using spaCy"""
    words = comment.split()
    tags = []
    lemma_words = []
    result=[]
    for i in range(len(words)):
            tags.append(words[i][words[i].rindex("/"):])
            words[i] = words[i][:words[i].rindex("/")]
    doc = spacy.tokens.Doc(nlp.vocab, words)
    doc = nlp.tagger(doc)
    for i in doc:
        if i.lemma_[0] =="-" and i.string[0]!="-":
            lemma_words.append(i.text)
        else:
            lemma_words.append(i.lemma_)
                 
    for i in range(len(lemma_words)):
    
        result.append(lemma_words[i]+tags[i])
        
    return " ".join(result)

#step9
def eos(comment):
    """Add a newline between each sentence"""
    comment=re.sub(r'(\W+/.)',r'\1 \n',comment,flags=re.IGNORECASE)
    return comment

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
        
    '''
    modComm = ''
    
    if 1 in steps: 
        comment = remove_newline(comment) 
        
    if 2 in steps: 
        comment = html_char(comment) 
       
    if 3 in steps:
        comment = remove_url(comment) 
        
    if 4 in steps: 
        comment = split_pun(comment) 

    if 5 in steps: 
        comment = clitics(comment) 
   
    if 6 in steps:
        comment = tag(comment)  
        
    if 7 in steps:
        comment = stopwords(comment)  
    
    if 8 in steps:
        comment = lemma(comment) 
        
    if 9 in steps: 
        comment = eos(comment) 
        
    if 10 in steps: 
        comment=re.sub(r'\b(\w+)(/)',lambda match: r'{}/'.format(match.group(1).lower()),comment) 
        
    modComm = comment
    return modComm 

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            count=0
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            data = json.load(open(fullFile))
            # TODO: select appropriate args.max lines
            start = args.ID[0]%len(data)
            end = start+int(args.max)
            selected_lines = data[start:end]
            lines =[]
            # TODO: read those lines with something like `j = json.loads(line)`
            for i in selected_lines:
                print(count)
                count=count+1
                j = json.loads(i)
                
            # TODO: choose to retain fields from those lines that are relevant to you
                keys=["body","id"]
                temp = {}
                for k in keys:
                    temp[k]=j[k]
                j = temp 
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                j["cat"]= file
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                preprocessed_text = preproc1(j['body'])
            # TODO: replace the 'body' field with the processed text
                j['body'] = preprocessed_text 
            # TODO: append the result to 'allOutput'
                allOutput.append(j)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
