def activation_memory(
    a, # attention heads
    b, # micro batch size
    h, # hidden dimension size
    h_ff, # feedforward dimension size (often h_ff = 4h)
    L, # number of layers
    s, # sequence length
    mixed=True,
    recomputation="none"
    ):
    
    # https://arxiv.org/pdf/2205.05198
    if mixed:
        bytes_per_value = 2 
    else:
        bytes_per_value = 4

    one_layer_attention = s * b * h * (bytes_per_value * 5 + 1) + ((2 * bytes_per_value + 1) * a * s * s * b) # eq (2)
    one_layer_feedforward_mlp = (s * b * h * bytes_per_value + (s * b * h_ff * bytes_per_value)   # inputs of 1st/2nd linear layers
         + s * b * h_ff * bytes_per_value # inputs of activation function (not really necessary for Relu though)
            + s * b * h)  # dropout
    one_layer_feedforward_swiglu = (s * b * h * bytes_per_value + (s * b * h_ff * bytes_per_value)   # inputs of input/output linear layers
         + s * b * h_ff * bytes_per_value * 3 # inputs of activation function
            + s * b * h)  # dropout (note that dropout is lower-precision - boolean)


    if recomputation == "none":
        one_layer = one_layer_attention # eq (2)
    elif recomputation =="selective":
        one_layer = s * b * h * 34 # eq (6)
    elif recomputation =="full":
        one_layer = s * b * h * 2
    else:
        raise ValueError()
    
    input_dropout = 0  # s * b * h # section 4.3

    total = L * one_layer + input_dropout
        
    return total


def param_grads_opt(
    h, # hidden dimension size
    L, # number of layers
    s, # sequence length
    v, # vocab size
    k=8, # parameters for optimizer (Adam: 8 = 4 bytes moments + 4 bytes variance)
    mixed=True # mixed precision training
    ):
    
    # https://michaelwornow.net/2024/01/18/counting-params-in-transformer
    # note: this is without GQA or MQA
    
    emb = h*(v+s)
    one_layer = 12 * h**2 + 13*h
    other = 2*h

    n = emb + L * one_layer + other
    
    # 3.1 https://arxiv.org/pdf/1910.02054
    
    if mixed:
        k += 4 # additional full precision weights
        bytes_per_paramter = 2
    else:
        bytes_per_paramter = 4
    
    return bytes_per_paramter*n, bytes_per_paramter*n, k*n
