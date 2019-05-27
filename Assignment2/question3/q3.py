#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:29:05 2019

@author: ashish
"""

import os
input_path = ""
while(1):
    print ("Enter FASTA file path")
    input_path = input()
    if(os.path.isfile(input_path)):
        break
    else:
        print ("Invalid path to file")
output_path = "/ashish/question3/outputs/"

print("Running abcpred...")
os.system("perl /gpsr/standalone/abcpred/abcpred.pl -i "+input_path+" -o "+output_path+"abcpredres.txt")
print("Running ctlpred...")
os.system("perl /gpsr/standalone/ctlpred/ctlpred.pl -i "+input_path+" -o "+output_path+"ctlpredres.txt")
print("Running propred...")
os.system("perl /gpsr/standalone/propred/propred.pl -i "+input_path+" -o "+output_path+"propredres.txt")
print("Running toxinpred...")
t = input("Enter Threshold for SVM example(0.5): ")
os.system("perl /gpsr/standalone/toxinpred/toxinpred.pl -i "+input_path+" -o "+output_path+"toxinpredres.txt -t "+t +" -m 1")

print ("outputs generated in "+output_path + " directory")
