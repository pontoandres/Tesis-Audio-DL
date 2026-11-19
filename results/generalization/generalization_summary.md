# Generalization evaluation summary

Lower is better for ESR, MAE, MR-STFT and LUFS error.

| Effect | Architecture | ESR | MAE | MR-STFT | LUFS error (dB) |
|---|---|---:|---:|---:|---:|
| starcaster | CNN | 0.088174 | 0.013746 | 1.534211 | 1.251583 |
| starcaster | LSTM | 0.120346 | 0.017254 | 3.056460 | 1.490259 |
| starcaster | TCN | 0.125942 | 0.017296 | 1.433975 | 1.444384 |
| dynacomp | CNN | 1.585805 | 0.044117 | 2.910304 | 6.565575 |
| dynacomp | LSTM | 1.048534 | 0.036068 | 2.123175 | 14.768368 |
| dynacomp | TCN | 1.312389 | 0.041225 | 2.667515 | 9.551330 |
