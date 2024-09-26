##Name: Capacitor_model.py Author: Alex Epps
# Models the X and Y capacitor impedance given specific measurements

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

class Cap():
    def __init__(self, path:str) -> None:
        contents = pd.read_csv(path)
        contents.apply(pd.to_numeric)

        self.raw_freq = contents.iloc[:,0].to_numpy()
        self.raw_impedance = contents.iloc[:,1].to_numpy()
        return
    def piecewise_linear(self, freq: np.ndarray):
        results = np.zeros_like(freq)

        measurements = np.rec.fromarrays([self.raw_freq, self.raw_impedance], names=["Freq", "Impedance"])
        measurements.sort()

        for i in range(1, len(measurements.Freq)):
            start_f = measurements.Freq[i-1]
            stop_f = measurements.Freq[i]
            start_z = measurements.Impedance[i-1]
            stop_z = measurements.Impedance[i]

            mask = np.logical_and((freq <= stop_f), (freq >= start_f))
            slope = (stop_z - start_z)/(stop_f - start_f)

            results = results - results * mask +((freq - start_f)*slope+start_z)*mask

        self.freq = freq
        self.impedance = results

def open_template():
    path = os.path.abspath("./templates/capacitor.csv")

    contents = pd.DataFrame(columns=["Frequency", "Impedance"])
    contents.to_csv(path, index=False)

    os.startfile(path)

    


