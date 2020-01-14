import numpy as np
import sys
import argparse
import os
import json
import re
import csv

"""Helpers"""
def read_csv_norms():
    #convert norm file to a list. 
    words = []
    aoa = []
    img = []
    fam = []
    []
    with open("/Users/chulong/Desktop/CSC401A1/wordlists/BristolNorms+GilhoolyLogie.csv", "r") as norms:
        csv_file = csv.reader(norms)
        for i in csv_file:
            words.append(i[1])
            aoa.append(i[3])
            img.append(i[4])
            fam.append(i[5])
        words = words[1:]
        aoa = aoa[1:]
        img = img[1:]
        fam = fam[1:]
    return[words,aoa,img,fam]

def read_csv_norms_Warringer():
    #convert norm file to a list.
    words = []
    vmeansum = []
    ameansum = []
    dmeansum = []
    with open("/Users/chulong/Desktop/CSC401A1/wordlists/Ratings_Warriner_et_al.csv", "r") as norms:
        csv_file = csv.reader(norms)
        for i in csv_file:
            words.append(i[1])
            vmeansum.append(i[2])
            ameansum.append(i[5])
            dmeansum.append(i[8])
        words = words[1:]
        vmeansum = vmeansum[1:]
        ameansum = ameansum[1:]
        dmeansum = dmeansum[1:]
    return[words,vmeansum,ameansum,dmeansum]


#globals
norms1002576848 = read_csv_norms()
norms_Warringer1002576848 = read_csv_norms_Warringer()  

def extract1( comment ):
    ''' This function extracts features from a single comment


    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    #calling helper 
  
    #creat an array
    feats = np.zeros( (1, 173))
    
    #1 Number of first-person pronouns.  done
    first=re.findall(r'\b(I|me|my|mine|we|us|our|ours)/',comment)
    feats[0,0] = len(first)
    #2 Number of second-person pronouns. u ur urs are not tagged with PRP PRP$
    second=re.findall(r'\b(you|your|yours|u|ur|urs)/',comment)
    feats[0,1] = len(second)
    #3 Number of third-person pronouns
    third=re.findall(r'\b(he|him|his|she|her|hers|it|its|they|them|their|theirs)/',comment)
    feats[0,2] = len(third)
    #4 Number of coordinating conjunctions tagged with CC
    cc = re.findall(r'\b(\w)+/(CC)',comment)
    feats[0,3] = len(cc)
    #5 Number of past-tense verbs tagged with VBD
    past_tense = re.findall(r'\b(\w)+/(VBD)',comment)
    feats[0,4] = len(past_tense)
    #6 Number of future-tense verbs 
    future_tense = re.findall(r'\b(\'ll|will|gonna|going/VBG to/TO \w+/VB)',comment)
    feats[0,5] = len(future_tense)
    #7 Number of commas
    comma = re.findall(r'/,',comment)
    feats[0,6] = len(comma)
    #8 Number of multi-character punctuation tokens
    multi = re.findall(r'\W{2,}/',comment)
    feats[0,7] = len(multi)
    #9 Number of common nouns
    common = re.findall(r'/(NN|NNS)\b',comment)
    feats[0,8] = len(common)
    #10 Number of proper nouns # did not test 
    proper = re.findall(r'/(NNP|NNPS)\b',comment)
    feats[0,9] = len(proper)
    #11 Number of adverbs # did not test 
    adverb = re.findall(r'/(RB|RBR|RBS)\b',comment)
    feats[0,10] = len(adverb)
    #12 Number of wh- words # did not test 
    wh_words = re.findall(r'/(WDT|WP|WP$|WRB)\b',comment)
    feats[0,11] = len(wh_words)  
    #13 Number of slang acronyms
    slangs = re.findall(r'\b(smh|fwb|lmfao|lmao|lms|tbh|rofl|wtf|bff|wyd|lylc|brb|atm|imao|sml|btw|bw|imho|fyi|ppl|sob|ttyl|imo|ltr|thx|kk|omg|omfg|ttys|afn|bbs|cya|ez|f2f|gtr|ic|jk|k|ly|ya|nm|np|plz|ru|so|tc|tmi|ym|ur|u|sol|fml)/',comment)
    feats[0,12] = 0
    # 14. Number of words in uppercase (≥ 3 letters long)
    upper = re.findall(r'\b[A-Z]{3,}/',comment)
    feats[0,13] = len(upper)
    #15. Average length of sentences, in tokens
     #number of sentences
    num_sentences = comment.count("\n") + 1
     #number of tokens
    tokens =  re.findall(r'\S+/\S+',comment)
    feats[0,14] = len(tokens)/num_sentences
    #16. Average length of tokens, excluding punctuation-only tokens, in characters
    tokens_without_puns=re.findall(r'(\w+)/',comment)
    char_num = len("".join(tokens_without_puns))
    if len(tokens_without_puns) !=0 :
        feats[0,15] = char_num/len(tokens_without_puns)
    else:
        feats[0,15] = 0
    #17. Number of sentences. already did this in step 15
    feats[0,16] = comment.count("\n") + 1
    #18-23 need more tests
    # we got all tokens excluding punctuations in step 16.
    # we got norms in the helper functions section
    aoa1 = []
    img1 = []
    fam1 = []
    for i in tokens_without_puns:
        if i in norms1002576848[0]:
            index1 = norms1002576848[0].index(i)
            aoa1.append(int(norms1002576848[1][index1]))
            img1.append(int(norms1002576848[2][index1]))
            fam1.append(int(norms1002576848[3][index1]))
        
    if aoa1:
        feats[0,17] = np.nanmean(aoa1)
        feats[0,20] = np.nanstd(aoa1)      
    if img1:
        feats[0,18] = np.nanmean(img1)
        feats[0,21] = np.nanstd(img1)
    if fam1:
        feats[0,19] = np.nanmean(fam1)
        feats[0,22] = np.nanstd(fam1)
   

    #24-29
    vmean = []
    amean = []
    dmean = []
    for i in tokens_without_puns:
        if i in norms_Warringer1002576848[0]:
            index1 = norms_Warringer1002576848[0].index(i)
            vmean.append(float(norms_Warringer1002576848[1][index1]))
            amean.append(float(norms_Warringer1002576848[2][index1]))
            dmean.append(float(norms_Warringer1002576848[3][index1]))
        
    if vmean:
        feats[0,23] = np.nanmean(vmean)
        feats[0,26] = np.nanstd(vmean)
     
    if amean:
        feats[0,24] = np.nanmean(amean)        
        feats[0,27] = np.nanstd(amean)
  
    if dmean:
        feats[0,25] = np.nanmean(dmean) 
        feats[0,28] = np.nanstd(dmean)
 
    return feats
        

#def main( args ):
    ##need to be modified

    #data = json.load(open(args.input))
    #feats = np.zeros( (len(data), 173+1))
    #path = "/u/cs401/A1/feats/"
    ##open file
    #alt_file = open(path+"Alt_IDs.txt","r").read()
    #center_file = open(path+"Center_IDs.txt","r").read()
    #right_file = open(path+"Right_IDs.txt","r").read()
    #left_file = open(path+"Left_IDs.txt","r").read()
    ##convert file to a list 
    #alt_list = alt_file.split("\n")
    #center_list = center_file.split("\n")
    #right_list = right_file.split("\n")
    #left_list = left_file.split("\n")
    #alt_feats = np.load(path+"Alt_feats.dat.npy")
    #center_feats = np.load(path+"Center_feats.dat.npy")
    #right_feats = np.load(path+"Right_feats.dat.npy")
    #left_feats = np.load(path+"Left_feats.dat.npy")

    ## TODO: your code here #testing
    ## read input
    #i=0
    #for j in data:
        #if j["cat"] == "Alt":
            #feats[i][-1] = 3
            #feats[i][:29] = extract1(j["body"])[0][:29]
            #index=alt_list.index(j["id"])
            #feats[i][29:-1] = alt_feats[index]
            
        #if j["cat"] == "Center":
            #feats[i][-1] = 1
            #feats[i][:29] = extract1(j["body"])[0][:29]
            #index=center_list.index(j["id"])
            #feats[i][29:-1] = center_feats[index]
            
        #if j["cat"] == "Right":
            #feats[i][-1] = 2
            #feats[i][:29] = extract1(j["body"])[0][:29]
            #index=right_list.index(j["id"])
            #feats[i][29:-1] = right_feats[index]
            
        #if j["cat"] == "Left":
            #feats[i][-1] = 0
            #feats[i][:29] = extract1(j["body"])[0][:29]
            #index=left_list.index(j["id"])
            #feats[i][29:-1] = left_feats[index]
        #print(i)
        #i=i+1

    #np.savez_compressed(args.output, feats)

    
#if __name__ == "__main__": 

    #parser = argparse.ArgumentParser(description='Process each .')
    #parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    #parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    #args = parser.parse_args()
                 

    #main(args)

