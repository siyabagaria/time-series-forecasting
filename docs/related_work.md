"Are Transformers Effective for Time Series Forecasting?" (DLinear, Zeng et al)
  What problem it says Transformers have on time series:
    Transformers rely on self-attention, which is permutation-invariant (the output of a function or system does not change when the order or labels of the input elements are
    shuffled or rearranged), so it does not naturally preserve the temporal order of data. Although positional/temporal embeddings add some ordering information, the authors
    argue that Transformers still lose important temporal information and often fail to capture useful long-term temporal relationships. 
 What it proposes to fix it:
    Instead of using complicated Transformer architectures, the paper proposes LTSF-Linear, a simple one-layer linear model that directly maps past time-series values to 
    future values using Direct Multi-Step (DMS) forecasting. It also proposes DLinear, which separately models trend and seasonal components, and NLinear, which handles 
    distribution shifts through simple normalization.
 What it claims in the results:
    On 9 real-world datasets, LTSF-Linear outperforms existing Transformer-based models in most cases, with improvements of roughly 20–50% over the SOTA FEDformer in many
    multivariate forecasting settings. It also finds that Transformer performance often does not improve—and can worsen—as the look-back window becomes longer, whereas 
    LTSF-Linear generally improves.

    
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
  What problem does it say Transformers have on time series?
    The paper says standard Transformers have three major problems when the time-series sequence is very long:
    - Quadratic self-attention: self-attention requires \(O(L^2)\) computation and memory, so it becomes extremely expensive as sequence length \(L\) increases.
    - High memory usage: stacking multiple encoder/decoder layers makes handling very long input sequences difficult.
    - Slow prediction: the standard encoder-decoder uses step-by-step decoding, so predicting a long future sequence becomes slow.     
  2. What does it propose to fix it?
      It proposes Informer, with 3 main changes:
        1. ProbSparse Self-Attention
           Instead of calculating every attention relationship, it focuses on the important ones.
           This reduces complexity from:
           O(L^2) to O(Llog L)
        2. Self-Attention Distilling
           It progressively reduces the sequence length between encoder layers, keeping the important information while reducing memory usage.
        3. Generative-style decoder
           Instead of predicting:
           t+1 → t+2 → t+3 → ...
           one at a time, it predicts the whole future sequence in one forward pass, making long-range prediction much faster.     
    3. What does it claim in the results?
          The paper tests Informer on 4 datasets and compares it with Transformer variants, RNN-based models, and traditional forecasting methods.     
          Its main claims are:
            - Informer gets the best result in 32 out of 50 univariate comparison cases, compared with 12 for its standard-attention version.     
            - Compared with LSTM, Informer reduces MSE by:
              - 26.8% for prediction length 168
              - 52.4% for prediction length 336
              - 60.1% for prediction length 720.     
            - Compared with DeepAR, ARIMA and Prophet, it reports average MSE reductions of:
              - 49.3% at 168
              - 61.1% at 336
              - 65.1% at 720.     
            - For multivariate forecasting, it also reports better results than the compared RNN-based methods; for example, MSE decreases relative to LSTMa by 26.6%, 28.2%, and 34.3% at prediction lengths 168, 336, and 720.     2012.07436v3
              
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.(Wu et al., 2021)
    1. Problem with Transformers on time series
      - Long-term time-series patterns are complex/entangled, so Transformers may not reliably discover the correct temporal dependencies.
      - Standard Transformer self-attention has \(O(L^2)\) complexity, making it computationally expensive for long sequences.
      - Existing sparse-attention Transformers such as Informer make attention more efficient, but because they connect only selected individual time points, they can lose          useful information.     
    2. What Autoformer proposes
    Autoformer proposes two main changes:
    - Progressive decomposition: It decomposes the time series into trend-cyclical + seasonal components inside the model itself, rather than doing decomposition only as          preprocessing.   
    - Auto-Correlation mechanism: It replaces self-attention with a mechanism that finds periodic dependencies and similar sub-series instead of connecting individual             points. It achieves \(O(L\log L)\) complexity.     
    3. What does it claim in the results?
    Autoformer claims state-of-the-art long-term forecasting performance and an overall 38% average reduction in MSE across its benchmark settings.     2106.13008v5
    Some specific comparisons against previous best results:
    - ETT: 74% lower MSE
    - Electricity: 18% lower MSE
    - Exchange: 61% lower MSE
    - Traffic: 15% lower MSE
    - Weather: 21% lower MSE
    - ILI: 43% lower MSE
