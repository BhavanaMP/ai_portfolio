GPT is Generative Pretrained Transformer. This is a decoder-only transformer that can predict next token given the sequence.

ChatGPT is based on GPT, but after pretraining the GPT with next token generation objective, several further fientuinigns will takeplace like - prompt engineering based training, reward based RLHF(Reinforcement learning from human feedback) training to make the model more like chat based GPT. GPT has versions till GPT 4 as of 2024.

Self attention = softmax(Q @ K.T / sqrt(head_dim)) @ V = softmax( (x_i.W_Q) @ (x_j.W_K).T / sqrt(head_dim)) @ (x_k.W_V)

The objective of this code to generate shakespeare like text given the input context. As the name suggests, this is GPT model that contains only tansformer decoder.

This code also contains KV cache for better memory optimization.

We used simple tokenizer to tokenize the input sentences. One can use any other char based tokenizers or even Word piece Tokenizer / BPE(sub word tokenizer) / Tiktoken from OpenAI.

Total Vocab is of size 65. The input corpus is in data/shakespeare.txt file

Positional Embeddings can be learned or even use the fixed sine or cosine encodings/embeddings.
- Static sinusoidal vectors: the positional embeddings are precomputed.
- Trained positional embeddings: the positional embeddings are learned.

3 Types of Positional Embeddings
- Absolute
    - Fixed vectors added to the token embeddings to represent absolution position of each token in the sentence.
- Relative
    - Deals with 2 tokens at a time. Calculated directly during attention computation as the purpose of attn is to find out the intensity of how the 2 tokens are related. Relative positional encosing basically tells the attention mechanism, the distance between 2 words that are involved in attention. So, given 2 tokens, we create a vector that represents their distance. SelfAttn = softmax(Q @ (K + p).T / sqrt(head_dim)) @ V = softmax( (x_i.W_Q) @ (x_j.W_K + a_ij_K).T / sqrt(head_dim)) @ (x_k.W_V) where a_ij is the vector that represents distance between words i and j
- Rotational
    - In between Absolute and Relative. It calculates positional embedding for each token liken in absolute and add it to the attention mechanism by calculating distance between 2 tokens. But it uses complex numbers, eulerformular, and rotational mechanism of vectors in dimensional space to get the positional embedding. Refer to the links given.

Absolute Positional Embedding is used in original transformer.

Note: Fixed kind of positional embeddings are computed only once. It will be created only once for the first sentence and will be reused for all other sentences during training and inference. That is, every first position token in any sentence will have the same first positional encoding vector, every second position token in any sentence will have the same second positional encoding vector, every third position token in any sentence will have the same third positional encoding vector and so on.. These are precomputed and doesnt depend on word. Just the position of the word. It is also just like a look up table. Left side will be the index positions (0, 1, 2, ....) that represents the positions and each index will have a some d dimensional vector. SO, each row is positional vector of respective position stated in index column (think of table structure)



Ref Link - https://stackoverflow.com/questions/76825022/why-nn-embedding-layer-is-used-for-positional-encoding-in-bert
Ref Link - https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

The code basically be able to generate text given the input context. The generated text will be saved in generation_out.txt.

All the logs while executing the code will be saved in gpt_nano.log file.

The code also supports distributed training

The trained checkpoint will be saved as chargpt_nano.pt

Ref Link to Andrey Karpathy Tutorial Video: https://youtu.be/kCc8FmEb1nY?si=4dY0ZRctmxKytiI_
Ref Links to understand Transformers at Inference and at Training: https://youtu.be/IGu7ivuy1Ag?si=_uO0BiQYEADE79RP

This code is inspired from the Andrej Karpathy Nano GPT tutorial.

# Fixme
- write save chckpoint
- ddp
