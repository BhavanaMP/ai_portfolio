 Llama from scratch in Pytorch.

 

 Large Language Model Meta AI, is a family of autoregressive large language models (LLMs) released by Meta AI. Llama is a decoder-only language model, which takes the input sentence as ordered tokens and predicts the next token.Givent the context, model will predict the next token in the sequence. More like generating text based on previous context so far.

 In this module, we implemented

 - Changes between Vanilla Transformer and Llama
 - RMS Normalization (instead of Layer Normalization)
 - RoPE (instead of Absolute or Position Embeddings) 
 - KV cache
 - SwiGLU Activation function
 - Multi Query Attention
 - Grouped Multi Query Attention (GQA)
 - Inference methods: Greedy, Beam Search, Temperature Scaling, Random Sampling, Top K, Top P

 Rotary Positional Embeddings
 ----------------------------
- Positional Embeddings are computed after RMS norm of token embeddings and getting Q, K, V's just before performing self attention, unlike usual transformer which computed pos embeddings after the token embeddings and then Layer Norm is applied and Q, K, V is calculated
- Computed only for Q and K matrices but not on V. 

 Ref Link - https://youtu.be/Mn_9W1nCFLo?si=hfduLDa2OtLpW7DV