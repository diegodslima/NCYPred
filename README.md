# NCYPred: a bidirectional long short-term memory network with attention mechanism for small non-coding RNA and Y-RNA classification

Python 3.7

Dependencies:
- pandas 1.1.1
- biopython 1.78
- numpy 1.19
- tensorflow 2.3+

Usage:
```
python predict.py -i input_file.fasta -o output_file
```

## About

  NCYPred is a deep learning model trained to classify non-coding RNA transcripts into 13 distinct classes: 5.8S rRNA, 5S rRNA, CD-box, HACA-box, Intron-gpI, Intron-gpII, Leader, miRNA, Riboswitch, Ribozyme, tRNA, Y RNA (vertebrates), sbRNA and CeY RNA (nematodes), sbRNA (insects), Y RNA like (bacterial). The model receives as input a non-coding RNA sequence in FASTA format and outputs the predicted class. NCYPred was trained on a dataset consisting of 31,000 non-coding RNA sequences obtained from Rfam 14.0 (https://rfam.xfam.org/). It is capable of extracting features directly from nucleotides sequences, learn sequence representations and classify them in 1 of 13 classes.

  The model architecture was built based on bidirectional long short-term memory networks (biLSTM) with attention mechanism, using TensorFlow and Keras. Before feeding it to the neural network, input sequences are preprocessed: I) Sequences are decomposed into overlapping 3-mers (subsequences of 3 nt); II) Each unique 3-mer is “tokenized”. III) Sequences are concatenated with zeros until the maximum length is reached (500 nt). After this process, input sequences are fed to an Embedding layer, which maps each token into a 10-dimensional representation that is optimized during training. Then, the biLSTM layers encode sequential information, and the Attention layer assigns importance to each position, embedding the most relevant information into a context vector. This final sequence representation is used by a 128-layer feed-forward neural network to classify each sequence into 1 of 13 classes.

## Web-server
  NCYPred is also available at: https://www.gpea.uem.br/ncypred/

  NCYPred  web-server allows the user to send RNA strings (single or multiple) in FASTA format. If you want to get forecast results by email, please provide your email address.
Sequences can be sent via a file or by inserting the RNA sequences in the text box. If both are sent, only the file will be considered and the informed text will be ignored.
