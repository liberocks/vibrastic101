---
marp: true
paginate: true
---

# <!--fit--> Vibrastic 101 : Signal Procssing Crash Course
<div style="text-align:center">Tirtadwipa Manunggal</div>
<div style="text-align:center;font-size:16pt">Machine Learning Enthusiast</div>
<br>
<div style="text-align:center;font-size:18pt">tirtadwipa.manunggal@gmail.com</div>


---

# Waves

Let's take a look at this audio. At a glance, this audio seems chaotic and likely hard to extract the information. But when we zoom bigger at certain range of index, we can obviously see a pattern. The signal has a **periodic** pattern and smoothly sweeps into other **periodic** pattern.

---

# **Import scipy, matplotlib, and Audio**

```python
!pip install scipy matplotlib

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from IPython.display import Audio
```

---

# **Read audio**

```python
!wget https://raw.githubusercontent.com/linerocks/vibrastic101/master/data/audio.wav
fs, x = wav.read('audio.wav')
Audio(x, rate=fs)
```

---

# <!--fit--> Thank you