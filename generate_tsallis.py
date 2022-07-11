import pandas as pd
import numpy as np
import subprocess, os, shutil
import argparse

def generate_datasets():

    labels = ['sRNA', 'tRNA', 'rRNA']
    q = 0.1

    for n in range(100):
        for label in labels:
                q = round(q, 1)
                subprocess.run(['python', 'MathFeature/methods/TsallisEntropy.py', '-i', 'Briefings/pre_' + label + '.fasta', '-o', 'Tsallis/' + str(q) + '.csv', '-l', label, '-k', '24', '-q', str(q)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        q += 0.1

generate_datasets()