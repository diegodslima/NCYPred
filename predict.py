from Bio import SeqIO
import pandas as pd
import argparse
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

'''
@function
'''
def make_argument_parser():
	parser = argparse.ArgumentParser(description='print text')

	parser.add_argument('-i', required=True, help='input file path', metavar='FILE')
	parser.add_argument('-o', required=True, help='output file name', metavar='FILE')

	return parser


'''
@function
'''
def remove(df):
	for i in range(len(df.seq)):
		for j in ['N', 'Y', 'K', 'W', 'R', 'H', 'M', 'S', 'D', 'V', 'B']:
			if j in df.seq[i]:
				df = df.drop(index=i)
				break

	df = df.reset_index(drop=True)
	return df


'''
@function
'''
''' input: seq_list = list of nucleotide sequences      
	output: list of lists of k-mers derived from sequences '''
def seq_to_3mer(seq_list):
	print('Processing {} sequences'.format(len(seq_list)))
	
	main_list = []
	
	for _, i in enumerate(seq_list):
		# print('type(i): ', type(i))
		# print('type([i]): ', type([i]))
		# print('type(list(i)): ', type(list(i)))
		seq = list(i)
		seq_kmer = []

		for j, _ in enumerate(seq):
			if j < len(seq) - 2:
				seq_kmer.append(seq[j] + seq[j+1] + seq[j+2])
			else:
				continue

		main_list.append(seq_kmer)

	return main_list 


'''
@function
'''
def token_pad(sentences, max_len, prefix):
	print('Zero-padding sequences to {} and tokenizing'.format(max_len))

	with open('./tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	tokens = tokenizer.texts_to_sequences(sentences)
	all_pad = pad_sequences(tokens, max_len, padding=prefix)

	return all_pad


'''
@function
'''
def argmax_to_label(predictions):
	label_list = ['5.8S-rRNA', '5S-rRNA', 'CD-box', 'HACA-box', 'Intron-gp-I',
		'Intron-gp-II', 'Leader', 'Riboswitch', 'Ribozyme', 'Y-RNA',
		'Y-RNA-like', 'miRNA ', 'tRNA']

	argmax_values = range(13)
	pred_labels = []

	for p in predictions:
		for n, l in zip(argmax_values, label_list):
			if p == n:
				pred_labels.append(l)

	return pred_labels


'''
@method
'''
def main():
	parser = make_argument_parser()
	args = parser.parse_args()

	# read input with biopython
	input_file = args.i
	output_file = args.o

	# dataframe to store data
	df = pd.DataFrame()

	input_id = []
	input_seq = []

	for seq_record in SeqIO.parse(input_file, 'fasta'):
		input_id.append(seq_record.id)

		if 'U' in seq_record.seq:
			dna_seq = seq_record.seq.back_transcribe()
			input_seq.append(dna_seq)

		else:
			input_seq.append(seq_record.seq)

	df['id'] = input_id
	df['seq'] = input_seq

	# remove sequences with unallowed characters
	df = remove(df)

	# decompose sequence into 3-mers
	x = df['seq']
	x = seq_to_3mer(x)

	# tokenization and zero-padding
	x_pad = token_pad(x, 498, 'post')

	# load model
	print('Loading model...')
	model = keras.models.load_model('./trained-model/', compile=False)
	print('Predicting...')
	predictions = model.predict(x_pad, verbose=0)
	argmax = np.argmax(predictions, axis=1)
	pred_labels = argmax_to_label(argmax)

	df['prediction'] = pred_labels

	print('Done.')
	# save output
	print('saving results...')
	df.to_csv('./{}.csv'.format(output_file), sep=';')



if __name__ == '__main__':
	main()