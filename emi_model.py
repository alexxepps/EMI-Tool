## Name: emi_model.py    Author: Abel Keeley
# Handles spectrum measurements, noise impedance calculations, and noise estimations for EMI filter design. Important changes in v6:
#   Updated |Zs| calcs...

import numpy as np
import math
import pandas as pd

from choke_model import Choke

# setup global vars
CM_base = None
CM_filter = None
CM_noise_impedance = None
DM_base = None
DM_filter = None
DM_noise_impedance = None
Noise_floor = None
Limit = None

# Holds measurements from spectrum analyzer, handles converting dBm to dBuV.
class Spectrum_Measurement:
    # load measurements from file
    def __init__(self, filepath: str = None, scope_impedance: float = 50) -> None:
        self.scope_impedance = scope_impedance

        # if creating a dummy object, return before trying to access filepath
        if filepath is None:
            return
        
        # load in data from csv file
        array = np.loadtxt(filepath, delimiter=",", dtype=str, encoding="utf-8-sig")
        headers = array[0]
        data = array[1:]
        # load data as float
        self.freq = np.asfarray(data[:,0])
        meas_dbm = np.asfarray(data[:,1])
        # convert dBm to dBuV
        self.measurement = meas_dbm + np.full_like(meas_dbm, 10*math.log10(scope_impedance) + 90)
    
    # returns the N-pt filtered measurements
    def get_n_filter(self, N: int) -> np.ndarray:
        # calculate the average for the center data points, will create transients for points within N/2 of bounds
        result =  np.convolve(self.measurement, np.ones(N)/N, mode='same')

        # clean up those transients
        edge = math.ceil(N/2)
        for index in range(edge):
            result[index] = np.sum(self.measurement[0:index+edge])/(index+edge)
        for index in range(self.measurement.size-edge, self.measurement.size):
            result[index] = np.sum(self.measurement[index-edge:])/(self.measurement.size-index+edge)
        return result

    # returns a new Spectrum_Measurement object, whose measurements are sufficiently far from the provided floor
    def get_de_floored(self, floor, pts = 3, delta = 3, window = 6):
        # run each measurement through an N point filter
        filtered_measurement = self.get_n_filter(pts)
        filtered_floor = floor.get_n_filter(pts)

        # find what data points are less than <delta> above the floor
        diff = filtered_measurement-filtered_floor
        below_floor = diff<delta

        # if any measurements within sliding window are below floor, mark entire window of measurements for removal
        remove = np.full_like(below_floor, False)
        for index in range(len(below_floor)):
            # handle edge cases
            if (index - math.floor(window/2)) < 0:
                # window is just at the beginning of array
                trail = 0
                lead = min(index - (index - math.floor(window/2)), len(below_floor)-1)
            elif (index + math.floor(window/2)) > (len(below_floor)-1):
                # window is just at the end of array
                lead = len(below_floor) - 1
                trail = index - (index + math.floor(window/2) - (len(below_floor)-1))
            else:
                lead = min(index + math.floor(window/2), len(below_floor)-1)
                trail = max(index - math.floor(window/2), 0)
            remove[index] = below_floor[trail:lead].sum() > 0

        # create dummy spectrum measurement object and copy over only freq, measurement pairs that are not marked for removal
        result = Spectrum_Measurement()
        result.freq = np.delete(self.freq, np.where(remove))
        result.measurement = np.delete(self.measurement, np.where(remove))
        return result

# Models CM/DM Noise Impedance
class Noise_Impedance:
    # requires a baseline measurement set, a filtered measurement set, and a measurement of the noise floor
    def __init__(self, baseline: Spectrum_Measurement, filtered: Spectrum_Measurement, floor: Spectrum_Measurement, min_freq: float = 150000, max_freq: float = 30000000) -> None:
        # copy baseline frequency
        self.freq = baseline.freq

        # remove any measurements too close to the noise floor
        baseline_no_floor = baseline.get_de_floored(floor)
        filtered_no_floor = filtered.get_de_floored(floor)

        #trim to min/max freq
        remove_mask_min_max = np.logical_or(self.freq > max_freq, self.freq < min_freq)

        #remove any freqs/measurements not in both baseline and filtered
        freq_in_base = np.isin(self.freq,baseline_no_floor.freq)
        freq_in_filtered = np.isin(self.freq,filtered_no_floor.freq)
        remove_mask_unique = np.logical_not(np.logical_and(freq_in_base, freq_in_filtered))
        self.remove_mask = np.logical_or(remove_mask_min_max, remove_mask_unique)

        self.freq = np.delete(self.freq, np.where(self.remove_mask))
        self.baseline_measurement = np.delete(baseline.measurement, np.where(self.remove_mask))
        self.filtered_measurement = np.delete(filtered.measurement, np.where(self.remove_mask))

        # calculate the attenuation between baseline and filtered
        self.attenuation = np.abs(10 ** ((self.baseline_measurement - self.filtered_measurement)/20))

    # helper function for paralleling circuit impedances
    def parallel_eqv(self, A, B):
        return A*B / (A + B)

    # calculate min, max, geo mean noise impedance for Common Mode, requires model of CM choke used. See documentation for background on calculations
    def CM_calculation(self, choke:Choke, freq:np.ndarray):
        Z_choke =  np.delete(choke.impedance, np.where(self.remove_mask))

        twentyfive = np.full_like(Z_choke, 25)
        one = np.ones_like(self.freq)

        #back-calculate noise source impedance
        self.min = np.abs(np.abs(Z_choke / (self.attenuation + one)) - twentyfive)
        self.max = np.abs(np.abs(Z_choke / (self.attenuation - one)) + twentyfive)

        self.fill_min = np.abs(np.average(self.min) - np.std(self.min))
        self.fill_max = np.abs(np.average(self.max) + np.std(self.max))

        new_min = np.full_like(freq, self.fill_min)
        new_max = np.full_like(freq, self.fill_max)

        for index in range(len(freq)):
            # check if this particular freq already has a value
            if freq[index] in freq:
                index_in_existing_arrays = np.flatnonzero(self.freq == freq[index])
                if index_in_existing_arrays.size:
                    new_min[index] = self.min[index_in_existing_arrays[0]]
                    new_max[index] = self.max[index_in_existing_arrays[0]]

        self.freq = freq
        self.min = new_min
        self.max = new_max
        self.geo_mean = np.sqrt(self.min*self.max)
        return
    
    # calculate min, max, geo mean noise impedance for Differential Mode, requires X capacitor used. See documentation for background on calculations
    def DM_calculation(self, capacitance, freq:np.ndarray):
        # usually fine to use ideal cap model
        x_cap = 1/(2*math.pi*capacitance*self.freq)

        onehundred = np.full_like(self.freq, 100)
        one = np.ones_like(self.freq)

        #back-calculate noise source impedance
        alpha_plus = np.abs(1 / (np.abs(x_cap*self.attenuation) + np.abs(x_cap)))
        alpha_minus = np.abs(1 / (np.abs(x_cap*self.attenuation) - np.abs(x_cap)))
        self.min = np.abs(one / (alpha_minus + np.abs(1/onehundred)))
        self.max = np.abs(one / (alpha_plus - np.abs(1/onehundred)))

        self.fill_min = np.abs(np.average(self.min) - np.std(self.min))
        self.fill_max = np.abs(np.average(self.max) + np.std(self.max))

        self.min = np.full_like(freq, self.fill_min)
        self.max = np.full_like(freq, self.fill_max)
        self.geo_mean = np.sqrt(self.min*self.max)
        self.freq = freq
        return

# Models EMI performance
class Estimate:
    def __init__(self, R_lisn) -> None:
        self.R_lisn = R_lisn

    # load in noise impedance model
    def add_noise(self, noise: Noise_Impedance):
        self.min_noise = noise.min
        self.max_noise = noise.max
        self.mean_noise = noise.geo_mean
        self.freq = noise.freq

    # helper function for paralleling circuit impedances
    def parallel_eqv(self, A, B):
        return A*B / (A + B)
    
    # calculates how estimated attenuation affects baseline noise
    def calculate_noise(self, baseline: Spectrum_Measurement):
        self.worst_estimate = 20 * np.log10((10 ** (baseline.measurement/20))/self.worst_attenuation)#------------where does this formula come fro
        self.mean_estimate = 20 * np.log10((10 ** (baseline.measurement/20))/self.mean_attenuation)

# Models CM EMI performance
class Common_Mode_Estimate(Estimate):
    # setup
    def __init__(self) -> None:
        super().__init__(25)

    # Helps calculate the attenuation for a specific noise source impedance.
    def CL_topology_helper(self, Z_noise_source: np.ndarray) -> np.ndarray:
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        filter = self.parallel_eqv(self.R_lisn, self.Z_cap) / (self.parallel_eqv(self.R_lisn, self.Z_cap) + self.Z_choke + Z_noise_source)
        attenuation = no_filter/filter
        #print (attenuation)
        return np.abs(attenuation)
    
    def CL_topology_math(self, y_cap: float, cm_choke: Choke):
        # Recall that in the CM equivalent circuit, both y caps are in parallel...
        self.Z_cap = -1j / (2*math.pi*self.freq*y_cap*2)
        self.Z_choke = cm_choke.impedance
    
    # handles calculations for a CL CM EMI circuit
    def CL_topology(self, y_cap: float, cm_choke: Choke):
        # Recall that in the CM equivalent circuit, both y caps are in parallel...
        self.Z_cap = -1j / (2*math.pi*self.freq*y_cap*2)
        self.Z_choke = cm_choke.impedance

        # Calculate attenuation for CL topology using mean noise
        self.mean_attenuation = self.CL_topology_helper(self.mean_noise)

        # Calculate attenuation for CL topology, using max/min noise and choosing the worst case result
        attenuation_w_min_noise = self.CL_topology_helper(self.min_noise)
        attenuation_w_max_noise = self.CL_topology_helper(self.max_noise)
        self.worst_attenuation = np.minimum(np.abs(attenuation_w_min_noise),np.abs(attenuation_w_max_noise))
        return
    
    def LC_topology_helper(self, Z_noise_source: np.ndarray) -> np.ndarray:
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        Z_lisn_choke = self.R_lisn + self.Z_choke
        Z_lisn_choke_y = self.parallel_eqv(Z_lisn_choke, self.Z_cap)
        filter = (Z_lisn_choke_y/(Z_lisn_choke_y + Z_noise_source))*(self.R_lisn/(self.R_lisn + self.Z_choke))
        attenuation = no_filter/filter
        return np.abs(attenuation)
    
    def LC_topology_math(self, y_cap: float, cm_choke: Choke):
        # Recall that in the CM equivalent circuit, both y caps are in parallel...
        self.Z_cap = -1j / (2*math.pi*self.freq*y_cap*2)
        self.Z_choke = cm_choke.impedance
        return
    
    # handles calculations for a LC CM EMI circuit
    def LC_topology(self, y_cap: float, cm_choke: Choke):
        # Recall that in the CM equivalent circuit, both y caps are in parallel...
        self.Z_cap = -1j / (2*math.pi*self.freq*y_cap*2)
        self.Z_choke = cm_choke.impedance

        # Calculate attenuation for CL topology using mean noise
        self.mean_attenuation = self.LC_topology_helper(self.mean_noise)

        # Calculate attenuation for CL topology, using max/min noise and choosing the worst case result
        attenuation_w_min_noise = self.LC_topology_helper(self.min_noise)
        attenuation_w_max_noise = self.LC_topology_helper(self.max_noise)
        self.worst_attenuation = np.minimum(np.abs(attenuation_w_min_noise),np.abs(attenuation_w_max_noise))
        return

    # Helps calculate the necessary choke impedance for a specific noise source impedance.
    def find_Z_choke_helper(self, Z_noise_source: np.ndarray) -> np.ndarray:
        A = np.abs((self.needed_attenuation * (self.R_lisn + Z_noise_source) * self.parallel_eqv(self.R_lisn, self.Z_cap)) / self.R_lisn)
        B = np.abs(self.parallel_eqv(self.R_lisn, self.Z_cap) + Z_noise_source)

        return np.abs(A - B)
    
    def find_Z_choke_helper_LC(self, Z_noise_source: np.ndarray) -> np.ndarray:
        z_root_list = []

        print(self.Z_cap)
        for noise_pt, attenuation_pt, z_cap_pt in zip(Z_noise_source, self.needed_attenuation, self.Z_cap):
            #variables for quadratic
            C = z_cap_pt * (self.R_lisn)**3
            B = z_cap_pt * (self.R_lisn)**2
            A = z_cap_pt * self.R_lisn
            CC = C + noise_pt * (self.R_lisn)**3 + noise_pt * z_cap_pt * (self.R_lisn)**2
            BB = 2 * B + 2 * (noise_pt * (self.R_lisn)**2) + noise_pt * z_cap_pt * self.R_lisn
            AA = abs(A + noise_pt * self.R_lisn)
            DD = B + noise_pt * z_cap_pt * self.R_lisn
            EE = C + noise_pt * z_cap_pt * (self.R_lisn)**2
            X = attenuation_pt * EE
            CCC = abs(CC - X)
            BBB = abs(BB  - (attenuation_pt * DD))
            discriminant = BBB**2 - 4 * AA * CCC
            #print(discriminant)

            #does quadratic math
            if discriminant > 0:
                root1 = abs(-BBB + math.sqrt(discriminant)) / (2 * AA)
                root2 = abs(-BBB - math.sqrt(discriminant)) / (2 * AA)
                z_root_list.append(max(root1, root2))
                
            elif discriminant == 0:
                root = abs(-BBB / (2 * AA))
                z_root_list.append(root)
                
            else:
                real_part = -BBB / (2*AA)
                imaginary_part = math.sqrt(-discriminant) / (2*AA)
                root1 = abs(complex(real_part, imaginary_part))
                root2 = abs(complex(real_part, -imaginary_part))
                z_root_list.append(max(root1, root2))
        return z_root_list
                
        #overall equation
        #0 = Zchoke**2 (AA) + Zchoke (BB-Attenuation*DD) + CC - (Attenuation * EE) 

    # Back-calculate the min choke impedance needed to meet a given limit
    def find_Z_choke(self, baseline: Spectrum_Measurement, limit: np.ndarray):
        # Calculate necessary attenuation to meet limit
        self.needed_attenuation = 10 ** (baseline.measurement/20) / 10 ** (limit/20)

        # find Z_choke with mean noise
        self.needed_mean_Z_choke = self.find_Z_choke_helper(self.mean_noise)

        # find Z_choke with worst case (min/max) noise
        self.needed_worst_Z_choke = np.maximum(self.find_Z_choke_helper(self.min_noise),self.find_Z_choke_helper(self.max_noise))

        return
    
    def find_Z_choke_LC(self, baseline: Spectrum_Measurement, limit: np.ndarray):
        # Calculate necessary attenuation to meet limit
        self.needed_attenuation = 10 ** (baseline.measurement/20) / 10 ** (limit/20)

        # find Z_choke with mean noise
        self.needed_mean_Z_choke = self.find_Z_choke_helper_LC(self.mean_noise)

        # find Z_choke with worst case (min/max) noise
        self.needed_worst_Z_choke_LC = np.maximum(self.find_Z_choke_helper_LC(self.min_noise), self.find_Z_choke_helper_LC(self.max_noise))
        return

    
    def save_Z_choke(self, file):
        # save worst Z_choke to file
        contents = pd.DataFrame({"Frequency":self.freq, "Impedance":self.needed_worst_Z_choke})
        
        # seems to insert blank lines unless lineterminator is specified...
        contents.to_csv(file, index=False, lineterminator='\n')
        return
    def save_Z_choke_LC(self, file):
        # save worst Z_choke to file
        
        contents = pd.DataFrame({"Frequency":self.freq, "Impedance":self.needed_worst_Z_choke_LC})
        # seems to insert blank lines unless lineterminator is specified...
        contents.to_csv(file, index=False, lineterminator='\n')
        return
   


# Models DM EMI performance
class Differential_Mode_Estimate(Estimate):
    # setup
    def __init__(self) -> None:
        super().__init__(100)
    
    # Helps calculate the attenuation for a specific noise source impedance.
    def C_topology_helper(self, Z_noise_source:np.ndarray):
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        R_Z_x = self.parallel_eqv(self.R_lisn, self.Z_x_cap)
        filter = R_Z_x / (R_Z_x + Z_noise_source)
        Attenuation = no_filter/filter
        return np.abs(Attenuation)
    
    # handles calculations for a C DM EMI circuit
    def C_topology(self, x_cap: float):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)

        # Calculate attenuation for C topology using mean noise
        self.mean_attenuation = self.C_topology_helper(self.mean_noise)

        # Calculate attenuation for C topology, using max/min noise and choosing the worst case result
        A_min_noise = self.C_topology_helper(self.min_noise)
        A_max_noise = self.C_topology_helper(self.max_noise)
        self.worst_attenuation = np.minimum(np.abs(A_min_noise),np.abs(A_max_noise))
        return
    
    # Helps calculate the attenuation for a specific noise source impedance.
    def PI_topology_helper(self, Z_noise_source:np.ndarray):
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        R_Z_y = self.parallel_eqv(self.R_lisn, self.Z_y_cap)
        Z1 = self.parallel_eqv((R_Z_y + self.Z_leakage), self.Z_x_cap)
        filter = (R_Z_y / (R_Z_y + self.Z_leakage)) * (Z1 / (Z1 + Z_noise_source))
        # // half leakage calcs
        alpha = R_Z_y + self.Z_leakage
        ZxAlpha = self.parallel_eqv(self.Z_x_cap, alpha)
        filter = (ZxAlpha / (ZxAlpha + Z_noise_source)) * (R_Z_y / alpha)
        # \\ half leakage calcs
        A = no_filter/filter
        return np.abs(A)
    
    # handles calculations for a PI DM EMI circuit, using y caps and leakage inductance
    def PI_topology(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance

        # Calculate attenuation for PI topology using mean noise
        self.mean_attenuation = self.PI_topology_helper(self.mean_noise)

        # Calculate attenuation for PI topology, using max/min noise and choosing the worst case result
        A_min_noise = self.PI_topology_helper(self.min_noise)
        A_max_noise = self.PI_topology_helper(self.max_noise)
        self.worst_attenuation = np.minimum(np.abs(A_min_noise),np.abs(A_max_noise))
        return
    
    def PI_topology_math(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance

     # Helps calculate the attenuation when two x caps are used ----------------------------------------------------------------------
    def PI2_topology_helper(self, Z_noise_source:np.ndarray):
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        R_Z_x = self.parallel_eqv(self.R_lisn, self.Z_x_cap)
        Zxy = self.parallel_eqv(self.Z_y_cap, self.Z_x_cap) #really self._x_2_cap
        Z_l_x_leak = R_Z_x + self.Z_leakage 
        Z_l_x_leak_y_x = self.parallel_eqv(Z_l_x_leak, Zxy)
        filter = (Z_l_x_leak_y_x / (Z_l_x_leak_y_x + Z_noise_source))*(R_Z_x / (R_Z_x + self.Z_leakage))
        A = no_filter / filter
        
        return np.abs(A)
    
    # handles calculations for a PI DM EMI circuit, using y caps and leakage inductance
    def PI2_topology(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance

        # Calculate attenuation for PI topology using mean noise
        self.mean_attenuation = self.PI2_topology_helper(self.mean_noise)

        # Calculate attenuation for PI topology, using max/min noise and choosing the worst case result
        A_min_noise = self.PI2_topology_helper(self.min_noise)
        A_max_noise = self.PI2_topology_helper(self.max_noise)
        self.worst_attenuation = np.minimum(np.abs(A_min_noise),np.abs(A_max_noise))
        return
    
    def PI2_topology_math(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance
    
    def CCL_topology_helper(self, Z_noise_source:np.ndarray):
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        parallel = (self.R_lisn*self.Z_y_cap*self.Z_x_cap)/(self.R_lisn+self.Z_y_cap+self.Z_x_cap)
        filter = parallel / (parallel + Z_noise_source + self.Z_leakage)
        A = no_filter / filter
        return np.abs(A)

    def CCL_topology(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance

        # Calculate attenuation for PI topology using mean noise
        self.mean_attenuation = self.CCL_topology_helper(self.mean_noise)

        # Calculate attenuation for PI topology, using max/min noise and choosing the worst case result
        A_min_noise = self.CCL_topology_helper(self.min_noise)
        A_max_noise = self.CCL_topology_helper(self.max_noise)
        
        self.worst_attenuation = np.minimum(np.abs(A_min_noise),np.abs(A_max_noise))
        return
    
    def CCL_topology_math(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance
        return
    
    def LCC_topology_helper(self, Z_noise_source:np.ndarray):
        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        Z2 = self.parallel_eqv(self.Z_y_cap, self.Z_x_cap)
        Z1 = self.R_lisn + self.Z_leakage
        Z3 = self.parallel_eqv(Z1, Z2)
        filter = Z3 / (Z3 + Z_noise_source)
        A = no_filter / filter
        return np.abs(A)
    
    def LCC_topology(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance

        # Calculate attenuation for PI topology using mean noise
        self.mean_attenuation = self.LCC_topology_helper(self.mean_noise)

        # Calculate attenuation for PI topology, using max/min noise and choosing the worst case result
        A_min_noise = self.LCC_topology_helper(self.min_noise)
        A_max_noise = self.LCC_topology_helper(self.max_noise)
        
        self.worst_attenuation = np.minimum(np.abs(A_min_noise),np.abs(A_max_noise))
        return
    
    def LCC_topology_math(self, x_cap: float, y_cap:float, choke_leakage:Choke):
        self.Z_x_cap = -1j / (2*math.pi*self.freq*x_cap)
        # y caps will be in series for the DM equiv circuit...
        self.Z_y_cap = -1j / (2*math.pi*self.freq*y_cap/2)
        self.Z_leakage = choke_leakage.impedance
        return

    # Helps calculate the X cap impedance for a specific noise source impedance.
    def find_Z_x_helper(self, Z_noise_source:np.ndarray):
        # Assuming PI topology...
        R_Z_y = self.parallel_eqv(self.R_lisn, self.Z_y_cap)
        alpha = np.abs(self.Z_leakage + R_Z_y)

        no_filter = self.R_lisn / (self.R_lisn + Z_noise_source)
        filter = no_filter / self.needed_attenuation
        A = np.abs(filter * Z_noise_source * alpha)
        B = np.abs((alpha + Z_noise_source) * filter)
        return np.abs(A / (np.abs(R_Z_y) - B))

    # Back-calculate the max X cap impedance needed to meet a given limit
    def find_Zx(self, base: Spectrum_Measurement, limit: np.ndarray):
        # Calculate necessary attenuation to meet limit
        self.needed_attenuation = 10 ** (base.measurement/20) / 10 ** (limit/20)

        # find Z_x with worst case (min/max) noise
        self.needed_Z_x = np.minimum(self.find_Z_x_helper(self.min_noise),self.find_Z_x_helper(self.max_noise))
        return
    
    def find_Z_x_helper_PI2(self, Z_noise_source:np.ndarray):
        z_x_result_array = []
        check = None
        for noise_pt, attenuation_pt, z_cap_y_pt, z_leak in zip(Z_noise_source, self.needed_attenuation, self.Z_y_cap, self.Z_leakage):
            for Z_x_cap_random in range(0, 10000, 1):
                no_filter = self.R_lisn / (self.R_lisn + noise_pt)
                r_z_x = self.parallel_eqv(self.R_lisn, Z_x_cap_random)
                zxy = self.parallel_eqv(z_cap_y_pt, Z_x_cap_random) #really self._x_2_cap
                z_l_x_leak = r_z_x + z_leak
                z_l_x_leak_y_x = self.parallel_eqv(z_l_x_leak, zxy)
                filter = (z_l_x_leak_y_x / (z_l_x_leak_y_x + noise_pt))*(r_z_x / (r_z_x + z_leak))
                result_math = abs(no_filter / (filter * attenuation_pt))
            
                if result_math < 5:
                    z_x_result_array.append(result_math)
                    check = 1
                    temp = result_math
                    break
                else:
                    check = 0
                if Z_x_cap_random == 1000 and check == 0:
                    z_x_result_array.append(temp)
                    break
        #print ('z_x_result_array')
        #print (z_x_result_array)
        return z_x_result_array
    
    def find_Zx_PI2(self, base: Spectrum_Measurement, limit: np.ndarray):
        self.needed_attenuation = 10 ** (base.measurement/20) / 10 ** (limit/20)

        self.needed_Z_x_PI2 = np.maximum(self.find_Z_x_helper_PI2(self.min_noise), self.find_Z_x_helper_PI2(self.max_noise))
        return
    

    def find_Z_x_helper_CCL(self, Z_noise_source:np.ndarray):
        
        numerator1 = -self.Z_leakage * self.R_lisn**2 * self.needed_attenuation - self.Z_leakage * self.Z_y_cap * self.R_lisn * self.needed_attenuation - Z_noise_source * self.R_lisn**2 * self.needed_attenuation
        numerator2 = -Z_noise_source * self.Z_y_cap * self.R_lisn * self.needed_attenuation - self.Z_leakage * self.R_lisn * Z_noise_source * self.needed_attenuation - self.Z_leakage * Z_noise_source * self.Z_y_cap * self.needed_attenuation
        numerator3 = -Z_noise_source**2 * self.R_lisn * self.needed_attenuation - Z_noise_source**2 * self.Z_y_cap
        numerator = numerator1 + numerator2 + numerator3

        denominator1 = self.R_lisn**2 * self.Z_y_cap * self.needed_attenuation + self.Z_leakage * self.R_lisn * self.needed_attenuation
        denominator2 = Z_noise_source * self.R_lisn + self.Z_y_cap * self.R_lisn * Z_noise_source * self.needed_attenuation + Z_noise_source * self.Z_leakage * self.needed_attenuation
        denominator3 = Z_noise_source**2 * self.needed_attenuation - self.R_lisn**2 * self.Z_y_cap
        denominator = denominator1 + denominator2 + denominator3

        return np.abs(numerator/denominator)
    
    def find_Z_x_helper_LCC(self, Z_noise_source:np.ndarray):

        numerator1 = Z_noise_source * self.Z_leakage * self.R_lisn * self.Z_y_cap + Z_noise_source * self.R_lisn**2 * self.Z_y_cap 
        numerator2 = Z_noise_source * self.Z_leakage**2 * self.Z_y_cap + Z_noise_source * self.Z_leakage * self.R_lisn * self.Z_y_cap
        numerator = numerator1 + numerator2

        denominator1 = self.needed_attenuation * self.Z_y_cap * self.Z_leakage * self.R_lisn + self.needed_attenuation * self.Z_y_cap * self.Z_leakage * Z_noise_source
        denominator2 = self.needed_attenuation * self.Z_y_cap * self.R_lisn**2 + self.needed_attenuation * self.Z_y_cap * self.R_lisn * Z_noise_source
        denominator3 = - self.Z_y_cap * self.Z_leakage * self.R_lisn - self.Z_y_cap * self.R_lisn**2 - Z_noise_source * self.Z_y_cap * self.R_lisn
        denominator4 = - Z_noise_source * self.Z_leakage * self.R_lisn - Z_noise_source * self.R_lisn**2 - self.Z_y_cap * self.Z_leakage**2 - self.Z_y_cap * self.Z_leakage * self.R_lisn
        denominator5 = - Z_noise_source * self.Z_leakage**2 - Z_noise_source * self.Z_leakage * self.R_lisn
        denominator = denominator1 + denominator2 + denominator3 + denominator4 + denominator5

        return np.abs(numerator/denominator)

    def find_Zx_CCL(self, base: Spectrum_Measurement, limit: np.ndarray):
        self.needed_attenuation = 10 ** (base.measurement/20) / 10 ** (limit/20)

        self.needed_Z_x_CCL = np.maximum(self.find_Z_x_helper_CCL(self.min_noise), self.find_Z_x_helper_CCL(self.max_noise))
        return

    def find_Zx_LCC(self, base: Spectrum_Measurement, limit: np.ndarray):
        self.needed_attenuation = 10 ** (base.measurement/20) / 10 ** (limit/20)

        self.needed_Z_x_LCC = np.maximum(self.find_Z_x_helper_LCC(self.min_noise), self.find_Z_x_helper_LCC(self.max_noise))

# Holds the noise limits in a form appropriate for calculations and display.
class Noise_Limit:
    def __init__(self, freq_range: np.ndarray) -> None:
        self.freq = freq_range
        low_range = 79 # <- 79dBuV, from FCC
        high_range = 73 # <- 73dBuV, from FCC

        self.upper_mask = freq_range >= 0.5 * 10**6 # <- high range is anything >= 500kHz
        self.FCC = self.calc_limit(high_range,low_range)

    # calculates an actual limit
    def calc_limit(self, upper, lower) -> np.ndarray:
        upper_val = np.full_like(self.freq, upper)
        lower_val = np.full_like(self.freq, lower)
        return np.where(self.upper_mask, upper_val, lower_val)
    
    # finds a modified limit based on a margin from the FCC limits.
    def add_margin(self, margin):
        self.limit = self.FCC - margin

# Given path to merged data file, reads data into provided CM/DM baseline, CM/DM filtered, and Noise Floor objects
def merge_file_load(path, CM_base: Spectrum_Measurement, CM_filtered: Spectrum_Measurement, Noise_floor: Spectrum_Measurement, DM_base: Spectrum_Measurement, DM_filtered: Spectrum_Measurement):
    if path is None:
            return
            
    # extract data
    array = np.loadtxt(path, delimiter=",", dtype=str, encoding="utf-8-sig")
    headers = array[0]
    data = array[1:]

    # load data into appropriate Spectrum_Measurements
    CM_base.freq = np.asfarray(data[:,0])
    CM_base.measurement = np.asfarray(data[:,1])
    CM_filtered.freq = np.asfarray(data[:,0])
    CM_filtered.measurement = np.asfarray(data[:,2])
    Noise_floor.freq = np.asfarray(data[:,0])
    Noise_floor.measurement = np.asfarray(data[:,3])
    DM_base.freq = np.asfarray(data[:,0])
    DM_base.measurement = np.asfarray(data[:,4])
    DM_filtered.freq = np.asfarray(data[:,0])
    DM_filtered.measurement = np.asfarray(data[:,5])

    # convert dBm measurements to dBuV
    CM_base.measurement = CM_base.measurement + np.full_like(CM_base.measurement, 10*math.log10(CM_base.scope_impedance) + 90)
    CM_filtered.measurement = CM_filtered.measurement + np.full_like(CM_filtered.measurement, 10*math.log10(CM_filtered.scope_impedance) + 90)
    Noise_floor.measurement = Noise_floor.measurement + np.full_like(Noise_floor.measurement, 10*math.log10(Noise_floor.scope_impedance) + 90)
    DM_base.measurement = DM_base.measurement + np.full_like(DM_base.measurement, 10*math.log10(DM_base.scope_impedance) + 90)
    DM_filtered.measurement = DM_filtered.measurement + np.full_like(DM_filtered.measurement, 10*math.log10(DM_filtered.scope_impedance) + 90)

    return