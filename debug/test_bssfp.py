#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
from signalmodel_bloch import calculate_signal_bssfp_bloch

#%%
fa = np.load('initialization/fa_cao.npy')
tr = np.load('initialization/tr_cao.npy')

# %%
t1 = 1000
t2 = 100
m0 = 1
beats = 1
shots = len(fa)
ph = np.zeros_like(fa)
ph[::2] = 180
prep = [1]
ti = [20]
t2te = [0]
te = 1

# %%
plt.figure()
for df in [0, 5, 10, 20]:    
    signal = calculate_signal_bssfp_bloch(t1, t2, m0, beats, shots, fa, tr, ph, prep, ti, t2te, te, df=df)      
    plt.plot(np.imag(signal), label=str(df))
plt.legend()
plt.show()
# %%
