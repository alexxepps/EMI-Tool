## Name: choke_model.py    Author: Abel Keeley
# Meant to model CM choke impedance given specific measurements
# Changes in V2
    # Exploring using a few measurements, plus line of best fit to generate impedance curve

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

# Represents a CM choke's impedance curve. Is created with a path to a .CSV file containing a set of Freq, Impedance pairs that characterize the curve. The model is expanded to cover an entire frequency range using the piecewise_linear function.
class Choke():
    def __init__(self, path: str) -> None:
        # read in the file specified by path to a pandas dataframe.
        contents = pd.read_csv(path)
        # convert file's contents from string to numeric
        contents.apply(pd.to_numeric)

        # assumes first col freq, second col impedance
        # store the original data points in numpy arrays
        self.raw_freq = contents.iloc[:,0].to_numpy()
        self.raw_impedance = contents.iloc[:,1].to_numpy()
        return
    
    # expand model to cover the entire frequency range that is passed in
    def piecewise_linear(self, freq: np.ndarray):
        # dummy variable to hold results while we work on them
        results = np.zeros_like(freq)

        # make sure the measurements are sorted in ascending frequency
        measurements = np.rec.fromarrays([self.raw_freq, self.raw_impedance], names=["Freq","Impedance"])
        measurements.sort()

        # step through each raw measurement and calculate the linear interpolate between it and the previous measurement.
        for i in range(1, len(measurements.Freq)):
            start_f = measurements.Freq[i-1]
            stop_f = measurements.Freq[i]
            start_z = measurements.Impedance[i-1]
            stop_z = measurements.Impedance[i]
            # isolate elements in freq between start and stop
            mask = np.logical_and((freq <= stop_f), (freq >= start_f))
            slope = (stop_z - start_z) / (stop_f - start_f)
            # calculate values and store in result
            results = results - results * mask + ((freq - start_f) * slope + start_z) * mask
        
        # save results
        self.freq = freq
        self.impedance = results

# Creates an empty choke data file template and then launches it for the user.
def open_template():
    # for some reason I can't seem to get the file to launch unless an absolute path is used
    path = os.path.abspath("./templates/CM Choke.csv")

    # setup template headers
    contents = pd.DataFrame(columns=["Frequency", "Impedance"])
    # create file
    contents.to_csv(path, index=False)

    # launch for user with default application
    os.startfile(path)
    
# If this file is launched directly, opens a template and displays some dummy data. Use this for quickly testing changes.
if __name__ == "__main__":
    open_template()
    path = "./templates/CM Choke.csv"
    example = Choke(path)
    full_freq = np.linspace(150000,30000000, 1000)
    example.piecewise_linear(full_freq)

    fig, axs = plt.subplots(1)

    axs.scatter(example.raw_freq, example.raw_impedance)
    axs.plot(example.freq, example.impedance)
    plt.show()