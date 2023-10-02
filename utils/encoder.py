from modules.freqencoder import FreqEncoder

def encoder(inDim = 3, multiRes = 6):
    """
    Create a frequency encoder

    Args:
        inDim (int, optional): The input dimension. Defaults to 3.
        multiRes (int, optional): The degree of multi-resolution encoding. Defaults to 6.

    Returns:
        Tuple: A tuple containing the frequency encoder and its output dimension.
    """
    encoder = FreqEncoder(input_dim = inDim, degree = multiRes)
    return encoder, encoder.output_dim