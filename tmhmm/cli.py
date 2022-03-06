import os
import argparse
import itertools
import textwrap
import logging

from tqdm import tqdm

from .api import predict
from .api import DEFAULT_MODEL
from .model import parse
from .utils import (
    dump_posterior_file,
    load_posterior_file,
    load_fasta_file,
)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

has_matplotlib = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    has_matplotlib = False


PRETTY_NAMES = {
    'i': 'inside',
    'M': 'transmembrane helix',
    'o': 'outside',
    'O': 'outside'
}


def summarize(path):
    """
    Summarize a path as a list of (start, end, state) triples.
    """
    for state, group in itertools.groupby(enumerate(path), key=lambda x: x[1]):
        group = list(group)
        start = min(group, key=lambda x: x[0])[0]
        end = max(group, key=lambda x: x[0])[0]
        yield start, end, state


def plot(posterior_file, outputfile):
    inside, membrane, outside = load_posterior_file(posterior_file)

    plt.figure(figsize=(16, 8))
    plt.title('Posterior probabilities')
    plt.suptitle('tmhmm.py')
    plt.plot(inside, label='inside', color='blue')
    plt.plot(membrane, label='transmembrane', color='red')
    plt.fill_between(range(len(inside)), membrane, color='red')
    plt.plot(outside, label='outside', color='black')
    plt.legend(frameon=False, bbox_to_anchor=[0.5, 0],
               loc='upper center', ncol=3, borderaxespad=1.5)
    plt.tight_layout(pad=3)
    plt.savefig(outputfile)



class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def cli():
    # argparse
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument('-f', '--fasta', dest='sequence_file',
                        type=argparse.FileType('r'), required=True,
                        help='path to file in fasta format with sequences')
    parser.add_argument('-m', '--model', dest='model_file',
                        default=DEFAULT_MODEL,
                        help='path to the model to use')
    parser.add_argument('-o', '--output-dir', dest='output', type=str, required=True,
                        default='tmhmm_output',
                        help='Output prefix')
    if has_matplotlib:
        parser.add_argument('-p', '--plot', dest='plot_posterior',
                            action='store_true',
                            help='plot posterior probabilies')

    args = parser.parse_args()

    # output
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # processing fasta files
    tsv_out = os.path.join(args.output, 'annotations.tsv')
    annot_out = os.path.join(args.output, 'annotations.fasta')
    with open(tsv_out, 'w') as outF1, open(annot_out, 'w') as outF2:
        # tsv
        header = ['entry', 'start', 'end', 'state']
        outF1.write('\t'.join(header) + '\n')

        # processing sequences
        seqs = load_fasta_file(args.sequence_file)        
        for seqid,entry in tqdm(seqs.items(), miniters=10):
            # prediction
            path, posterior = predict(entry['seq'], args.model_file)
            ## summary
            for start, end, state in summarize(path):
                line = [str(x) for x in [seqid, start, end, PRETTY_NAMES[state]]]
                outF1.write('\t'.join(line) + '\n')
            ## annotation
            line = ' '.join(['>' + seqid, entry['desc']])
            outF2.write(line + '\n')
            for line in textwrap.wrap(path, 79):
                outF2.write(line + '\n')

    # finish up
    logging.info(f'File written: {tsv_out}')
    logging.info(f'File written: {annot_out}')
