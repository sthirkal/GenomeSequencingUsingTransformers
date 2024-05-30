# Genome Sequencing using a Decoder-only Transformer Model.

Generating mutated genome sequence using Transformers and Multiheaded Attention Mechanism.

## Tech Stacks Used
1. Pytorch
2. Numpy

# What it does

The ongoing evolution of viruses poses a significant challenge to public health, necessitating the development of predictive models to predict genetic mutations in virus DNA. This paper presents a novel approach to predicting mutations in virus DNA sequences through the utilization of Multiheaded Attention, a deep learning mechanism originally designed for natural language processing tasks. By adapting this architecture to the genomics domain, we aim to enhance our ability to forecast viral mutation events accurately and use it to our advantage.

# Tests conducted

This model has been tested with the genomic sequences of COVID-19 and its variants. The output was then run through BLAST(A genome analysis tool) and analysed, uncovering some major similarities and mutations in the generated sequences.

# Changes to the Original Transformer Model

1)This model contains only a decoder from the original transformer with a low training cost.
2)Codon Tokenizer: The Tokenizer for this model is designed to tokenize and differentiate between different codons found in the input genomic sample.

# Things to expect

Version 2 for this model is a work-in-progress with a team that works on the biological aspect of the project like methods to improve the tokenizer and finding a way to analyze the output of the Model.
