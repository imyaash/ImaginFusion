from freqencoder import FreqEncoder

def getEncoder(inputDim = 3, multires = 6):
    encoder = FreqEncoder(input_dim = inputDim, degree = multires)
    return encoder, encoder.output_dim