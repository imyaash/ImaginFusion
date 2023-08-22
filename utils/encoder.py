from freqencoder import FreqEncoder

def Encoder(input_dim=3, multires=6):
    encoder = FreqEncoder(input_dim=input_dim, degree=multires)
    return encoder, encoder.output_dim