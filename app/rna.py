from Bio import SeqIO
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


# REMOVE SEQUÊNCIAS COM CARACTERES NÃO AUTORIZADOS
def remove(df):
	for i in range(len(df.seq)):
		for j in ['N', 'Y', 'K', 'W', 'R', 'H', 'M', 'S', 'D', 'V', 'B']:
			if j in df.seq[i]:
				df = df.drop(index=i)
				break

	df = df.reset_index(drop=True)
	return df


# DECOMPÕE A SEQUÊNCIA EM 3-MER
def seq_to_3mer(seq_list):
	main_list = []

	for _, i in enumerate(seq_list):
		seq = list(i)
		seq_kmer = []

		for j, _ in enumerate(seq):
			if j < len(seq) - 2:
				seq_kmer.append(seq[j] + seq[j+1] + seq[j+2])
			else:
				continue

		main_list.append(seq_kmer)

	return main_list


# TOKENIZAÇÃO
def token_pad(sentences, max_len, prefix):
	with open('app/static/tokenize.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	tokens = tokenizer.texts_to_sequences(sentences)
	all_pad = pad_sequences(tokens, max_len, padding=prefix)

	return all_pad


# CONVERTE ARGMAX EM UM DICIONÁRIO
def argmax_to_label(predictions):
	label_list = ['5.8S-rRNA', '5S-rRNA', 'CD-box', 'HACA-box', 'Intron-gp-I', 'Intron-gp-II',
		'Leader', 'Riboswitch', 'Ribozyme', 'Y-RNA', 'Y-RNA-like', 'miRNA ', 'tRNA']

	argmax_pred = np.argmax(predictions, axis=1)
	argmax_values = range(13)
	pred_labels = []

	for i in argmax_pred:
		for j, k in zip(argmax_values, label_list):
			if i == j:
				pred_labels.append(k)

	return pred_labels, label_list


# MODELO DE PREDIÇÃO
def calculate(filename):
	# RENOMEIA O CAMINHO DOS ARQUIVOS
	input = 'app/static/input/' + filename + '.fasta'
	output = 'app/static/output/' + filename

	# DATAFRAME PARA ARMAZENAR OS DADOS
	df = pd.DataFrame()

	input_id, input_seq = [], []

	for seq_record in SeqIO.parse(input, 'fasta'):
		input_id.append(seq_record.id)

		if 'U' in seq_record.seq:
			dna_seq = seq_record.seq.back_transcribe()
			input_seq.append(dna_seq.upper())

		else:
			input_seq.append(seq_record.seq.upper())

	df['id'] = input_id
	df['seq'] = input_seq

	# REMOVE SEQUÊNCIAS COM CARACTERES NÃO PERMITIDOS
	df = remove(df)

	# DECOMPÕE A SEQUÊNCIA EM 3-MERS
	x = df['seq']
	x = seq_to_3mer(x)

	x_pad = token_pad(x, 498, 'post')

	# CARREGA O MODELO
	model = keras.models.load_model('app/static/trained-model/', compile=False)
	predictions = model.predict(x_pad, verbose=0)
	pred_labels, label_list = argmax_to_label(predictions)

	df['prediction'] = pred_labels
	df_softmax = pd.DataFrame(data=predictions, columns=label_list)
	df_final = pd.concat([df, df_softmax], axis=1)

	# SALVA E RETORNA A SAÍDA
	df_final.to_csv('{}.csv'.format(output), sep=';', float_format='%.3f')
	return output + '.csv'