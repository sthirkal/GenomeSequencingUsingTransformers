# Genome Sequencing using a Decoder-only Transformer Model.

Generating mutated genome sequence using Transformers and Multiheaded Attention Mechanism.

## Tech Stacks Used
1. Pytorch
2. Numpy

# What does it do...

This is a novel approach to the existing field of genome sequencing. Using State-of-the-Art Architecture for Language Model, Transformers, We introduce scalability and efficiency in the field of Genome Sequencing.
This model analyzes all the input genome sequences of a certain virus and predicts the mutated genomic sequence of that virus that may occur in the foreseeable future. This allows the researchers in biotech to work on more data regarding the virus and work on the cure for that disease even before the mutation occurs. This AI model helps in the field of Vaccinology and BioInformatics.

# Tests conducted

This model is tested with the most trending disease of 2020 COVID-19, We fed the model the data of all the variants discovered on COVID-19. We ran the output through BLAST(A genome analysis tool) and we found some major similarities in both the Sequences.

# Changes to the Original Transformer Model

1)This model contains only a decoder from the original transformer with a low training cost. (Lack of funding, So didn't work on the Encoder Part).
2)Codon Tokenizer: The Tokenizer for this model is designed to tokenize and differentiate between different codons found in the input genomic sample.

# Things to expect from this project

Version 2 for this model is a work-in-progress with a team that works on the biological aspect of the project like methods to improve the tokenizer and finding a way to analyze the output of the Model.
