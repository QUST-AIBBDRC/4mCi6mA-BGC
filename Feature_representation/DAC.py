import os
import sys,re
import pickle
from collections import Counter
from functools import reduce
import itertools
import pandas as pd
import numpy as np
import platform

ALPHABET='ACGT'

def readDNAFasta(file):
	with open(file) as f:
		records = f.read()
	if re.search('>', records) == None:
		print('Error,the input DNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGT-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta



def acc_property(k):

    property_value = generate_property_value(k) 
    return property_value



def DAC(input_data,k,lag):
    phyche_value = acc_property(k)   
    fastas=readDNAFasta(input_data)   
    vector,_=make_ac_vector(fastas, lag, phyche_value, k)    
    return vector


input_vector=DAC('Sample.txt')
