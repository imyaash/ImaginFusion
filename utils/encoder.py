from freqencoder import FreqEncoder

def encoder(inDim = 3, multiRes = 6):
    encoder = FreqEncoder(input_dim = inDim, degree = multiRes)
    return encoder, encoder.output_dim