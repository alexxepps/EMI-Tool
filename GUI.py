## Name: GUI.py    Author: Abel Keeley
# Handle UI for EMI filter design tool. Important changes in v8:
#   Cleanup, change CM choke to work with choke_model v2

import tkinter as tk
from tkinter import Misc, ttk
from tkinter import filedialog, messagebox
import numpy as np
import os
import pandas as pd

import emi_model as emi
import choke_model
from freq_graph import Embedded_Graph

# global vars
CM_base = None
CM_filtered = None
Noise_Floor = None
DM_base = None
DM_filtered = None
filter_choke = None
actual_choke = None
leakage_choke = None
CM_Zs = None
DM_Zs = None
CM_est = None
DM_est = None
Cap_Dict = {"pF" : (10 ** -12), "nF" : (10 ** -9), "uF" : (10 ** -6)}
Ind_Dict = {"uH" : (10 ** -6), "mH" : (10 ** -3), "H" : (10 ** 0)}
Freq_Dict =  {"Hz" : (10 ** 0), "kHz" : (10 ** 3), "MHz" : (10 ** 6)}
Imp_Dict = {"Ohm" : (10 ** 0), "kOhm" : (10 ** 3), "MOhm" : (10 ** 6)}
CM_Topo = None
DM_Topo = None

# Main GUI wrapper
class GUI():
    def __init__(self, root):
        self.root = root
        self.root.title("EMI Design")

        # themes -> ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
        style = ttk.Style()
        style.theme_use('clam')

        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(side="top", fill='x')

        self.windows = []

        # initial frame to display
        self.file_selector()

    def create_nav_frame(self):
        # navigates between frames
        for widget in self.nav_frame.winfo_children():
            widget.pack_forget()

        nav_button = ttk.Button(self.nav_frame, text="Back", command=self.previous_state)
        nav_button.pack(side="left")
        text_frame = ttk.Frame(self.nav_frame)
        label = ttk.Label(text_frame, text=self.current_state.title)
        label.pack(anchor="center")
        text_frame.pack(side="left",expand=True,fill="x")
        nav_button = ttk.Button(self.nav_frame, text="Next", command=self.next_state)
        nav_button.pack(side="right")

    # handles frame switching
    def next_state(self):
        # if in file select, go to noise impedance
        if type(self.current_state).__name__ == "FileSelectWindow":
            self.noise_impedance()
            return
        
        # if in noise impedance, go to CM filter
        if type(self.current_state).__name__ == "NoiseImpedanceWindow":
            self.filter_CM()
            return
        
        # if in CM filter, go to DM filter
        if type(self.current_state).__name__ == "FilterCM":
            self.filter_DM()
            return
    
    def previous_state(self):
        # if in noise impedance, go back to file select
        if type(self.current_state).__name__ == "NoiseImpedanceWindow":
            self.file_selector()
        
        # if in CM filter, go back to noise impedance
        if type(self.current_state).__name__ == "FilterCM":
            self.noise_impedance()

        # if in DM filter, go back to CM filter
        if type(self.current_state).__name__ == "FilterDM":
            self.filter_CM()

    # file selector window
    def file_selector(self):
        # assume window needs to be created
        create_window = True
        for window in self.windows:
            if (type(window).__name__) == "FileSelectWindow":
                #file window already exists
                create_window = False
                file = window
            #hide all windows
            window.root.pack_forget()
        
        if create_window:
            file_frame = ttk.Frame(self.root)
            file = FileSelectWindow(file_frame, self.next_state)
            self.windows.append(file)
        
        #lastly, show file_selector window
        file.root.pack(side="top", fill="both", expand=1)
        self.current_state = file
        # updates nav frame
        self.create_nav_frame()

    # noise impedance window
    def noise_impedance(self):
        # assume window needs to be created
        create_window = True
        for window in self.windows:
            if (type(window).__name__) == "NoiseImpedanceWindow":
                #window already exists
                create_window = False
                noise = window
            #hide all windows
            window.root.pack_forget()
        
        if create_window:
            noise_frame = ttk.Frame(self.root)
            noise = NoiseImpedanceWindow(noise_frame)
            noise.create_input_frame()
            noise.create_visual_frame()
            self.windows.append(noise)

        #lastly, show noise impedance window
        noise.root.pack(side="top", fill="both", expand=1)
        self.current_state = noise
        self.create_nav_frame()

    # CM filter window
    def filter_CM(self):
        # assume window needs to be created
        create_window = True
        for window in self.windows:
            if (type(window).__name__) == "FilterCM":
                # window already exists
                create_window = False
                CM_generate = window
            #hide all windows
            window.root.pack_forget()
        
        if create_window:
            CM_generate_frame = ttk.Frame(self.root)
            CM_generate = FilterCM(CM_generate_frame)
            self.windows.append(CM_generate)

        #lastly, show CM filter window
        CM_generate.root.pack(side="top", fill="both", expand=1)
        self.current_state = CM_generate
        self.create_nav_frame()

    # DM filter window
    def filter_DM(self):
        # assume window needs to be created
        create_window = True
        for window in self.windows:
            if (type(window).__name__) == "FilterDM":
                #window already exists
                create_window = False
                DM_generate = window
            #hide all windows
            window.root.pack_forget()
        
        if create_window:
            DM_generate_frame = ttk.Frame(self.root)
            DM_generate = FilterDM(DM_generate_frame)
            self.windows.append(DM_generate)

        #lastly, show DM filter window
        DM_generate.root.pack(side="top", fill="both", expand=1)
        self.current_state = DM_generate
        self.create_nav_frame()

# Base class for windows, holds parent and a number of frames
class Window():
    def __init__(self, parent_frame: ttk.Frame) -> None:
        self.root = parent_frame
        self.frames = {}
        self.title = "Null"

# UI to allow user to select a measurement file
class FileSelectWindow(Window):
    def __init__(self, parent_frame: ttk.Frame, next_state) -> None:
        super().__init__(parent_frame)

        self.title = "Measurement File Select"

        self.next_state = next_state

        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill="y")

        # prompt for file selection
        select_button = ttk.Button(input_frame, text="Select a Data File", command=self.select_file)
        select_button.pack(side="left")
        label = ttk.Label(input_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(input_frame, text="View Template", command=self.open_template)
        select_button.pack(side="left")

        # create some instructions for the user
        label = ttk.Label(self.root, text = "Note - accepts a .csv file, see template for expected layout.")
        label.pack(side="top", anchor='center')
        label = ttk.Label(self.root, text = "There should be a data measurement of each kind for each frequency.")
        label.pack(side="top", anchor='center')

    # allows user to select a measurement file to load
    def select_file(self):
        global CM_base, CM_filtered, Noise_Floor, DM_base, DM_filtered
        # popup prompt for a .csv file
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])
        
        # check if the user selected a file
        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        # now we know that a file was selected        
        
        # setup empty measurement objects
        CM_base = emi.Spectrum_Measurement(None)
        CM_filtered = emi.Spectrum_Measurement(None)
        Noise_Floor = emi.Spectrum_Measurement(None)
        DM_base = emi.Spectrum_Measurement(None)
        DM_filtered = emi.Spectrum_Measurement(None)
        # fill measurement objects from measurement data file
        emi.merge_file_load(file_path, CM_base, CM_filtered, Noise_Floor, DM_base, DM_filtered)
        # automatically transition the GUI to the next window
        self.next_state()

    # creates and launches a template file, so the user knows what the tool expects
    def open_template(self):
        path = os.path.abspath("./templates/Measurements.csv")

        contents = pd.DataFrame(columns=["Frequency [Hz]", "Common Mode Baseline Data [dBm]", "Common Mode Filtered Data [dBm]","Noise Floor Data [dBm]","Differential Mode Baseline Data [dBm]","Differential Mode Filtered Data [dBm]"])
        contents.to_csv(path, index=False)

        # launch in default program
        os.startfile(path)

# UI to allow user to specify what known filter elements were used in the measurements
class NoiseImpedanceWindow(Window):
    def __init__(self, parent_frame: ttk.Frame) -> None:
        super().__init__(parent_frame)
        self.title = "Noise Impedance"


    # create the inputs for the user
    def create_input_frame(self):
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(side="left", fill="y")

        option_frame = ttk.Frame(self.input_frame)
        option_frame.pack()

        # user instructions
        label = ttk.Label(option_frame, text="The CM Choke used for the measurements needs to be modeled.")
        label.pack()
        label = ttk.Label(option_frame, text="Please load in a .csv file with a sufficient number of Frequency, Impedance pairs (see template) to model the Choke.")
        label.pack()
        label = ttk.Label(option_frame, text="Note - the CM Choke tab on the right can be used to visualize the model after noise impedance has been calculated.")
        label.pack()

        #added option for LC or CL filter for CM emi filter------------------------------------------------------------------------------
        label = ttk.Label(option_frame, text="Was a CM CL or LC filter used? C is the Ycap.")
        label.pack()
        #-----------------------------------------------------------------------------------------------------------------------------------


        #enter the filter details
        filter_frame = ttk.Frame(option_frame)

        #buttons that choose CM LC or CL filter------------------------------------------------------------------------------------------
        #command button will option which calculation to make
        select_button = ttk.Button(filter_frame, text="LC", command=self.topology_select_LC)
        select_button.pack(side="left")
        label = ttk.Label(filter_frame, text = "or")
        select_button = ttk.Button(filter_frame, text="CL", command=self.topology_select_CL)
        select_button.pack(side="right")
        #--------------------------------------------------------------------------------------------------------------------------------

        select_button = ttk.Button(filter_frame, text="Select CM Choke Data File", command=self.select_file)
        select_button.pack(side="left")
        label = ttk.Label(filter_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(filter_frame, text="Generate Template", command=self.gen_template)
        select_button.pack(side="left")

        filter_frame.pack()

        # more instructions
        label = ttk.Label(option_frame, text="The X Capacitor used for the measurements needs to be modeled.")
        label.pack()

        # enter the filter details
        filter_frame = ttk.Frame(option_frame)

        self.x_cap, self.x_cap_unit = label_entry_unit(filter_frame, "X Capacitor - Capacitance", 1000, Cap_Dict, "pF")
        
        filter_frame.pack()

        button(filter_frame, "Calculate Noise Impedance", self.calculate_noise_impedance)

        # add helpful visuals
        img_frame = ttk.Frame(self.input_frame)
        self.CM_img = tk.PhotoImage(file="./images/CM Impedance Test Circuits.png").subsample(2)
        self.DM_img = tk.PhotoImage(file="./images/DM Impedance Test Circuits.png").subsample(2)

        label = ttk.Label(img_frame, image=self.CM_img)
        label.pack()
        label = ttk.Label(img_frame, image=self.DM_img)
        label.pack()

        img_frame.pack()
        
    def create_visual_frame(self):
        self.visual_frame = ttk.Frame(self.root)
        self.visual_frame.pack(side="right", fill="both", expand=1)

        self.switcher_frame = ttk.Frame(self.visual_frame, height=100)
        self.switcher_frame.pack(side="top", anchor="n")
        baseline_button = ttk.Button(self.switcher_frame, text="Baseline", command=self.show_baseline)
        baseline_button.pack(side='left')
        filtered_button = ttk.Button(self.switcher_frame, text="Filtered", command=self.show_filtered)
        filtered_button.pack(side='left')
        filter_button = ttk.Button(self.switcher_frame, text="CM Choke", command=self.show_choke)
        filter_button.pack(side='left')
        impedance_button = ttk.Button(self.switcher_frame, text="CM Impedance", command=self.show_impedance_CM)
        impedance_button.pack(side='left')
        impedance_button = ttk.Button(self.switcher_frame, text="DM Impedance", command=self.show_impedance_DM)
        impedance_button.pack(side='left')

        self.graph_frame = tk.Frame(self.visual_frame)
        self.graph_frame.pack(side="bottom", fill="both", expand=1)

    def calculate_noise_impedance(self):
        global filter_choke, CM_base, CM_filtered, Noise_Floor, CM_Zs, DM_base, DM_filtered, DM_Zs

        x_cap = float(self.x_cap.get()) * self.x_cap_unit.get()

        if filter_choke is None:
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if CM_Topo is None:
            CM_Zs = emi.Noise_Impedance(CM_base, CM_filtered, Noise_Floor)
            CM_Zs.CM_calculation(filter_choke, CM_base.freq)
        if CM_Topo is 1:
            CM_Zs = emi.Noise_Impedance(CM_base, CM_filtered, Noise_Floor)
            CM_Zs.CM_calculation(filter_choke, CM_base.freq)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        DM_Zs = emi.Noise_Impedance(DM_base, DM_filtered, Noise_Floor)
        DM_Zs.DM_calculation(x_cap, DM_base.freq)

        #destroy existing frames
        for key in self.frames.keys():
            self.frames[key].destroy()

        #create replacement frames
        self.frames["Baseline"] = ttk.Frame(self.graph_frame)
        self.frames["Filtered"] = ttk.Frame(self.graph_frame)
        self.frames["CM Choke"] = ttk.Frame(self.graph_frame)
        self.frames["CM Impedance"] = ttk.Frame(self.graph_frame)
        self.frames["DM Impedance"] = ttk.Frame(self.graph_frame)

        baseline = Embedded_Graph(1)
        baseline.plot(0, CM_base.freq, CM_base.measurement, "CM")
        baseline.plot(0, DM_base.freq, DM_base.measurement, "DM")
        baseline.prettify(0, 'Noise\n[dB$\mu$V]', 'Baseline Measurements')
        baseline.draw(self.frames["Baseline"])

        defloored_CM = CM_filtered.get_de_floored(Noise_Floor)
        defloored_DM = DM_filtered.get_de_floored(Noise_Floor)
        filtered = Embedded_Graph(1)
        filtered.plot(0, defloored_CM.freq, defloored_CM.measurement, "CM")
        filtered.plot(0, defloored_DM.freq, defloored_DM.measurement, "DM")
        filtered.prettify(0, 'Noise\n[dB$\mu$V]', 'Filtered Measurements - Noise Floor Removed')
        filtered.draw(self.frames["Filtered"])

        CM_Choke_graph = Embedded_Graph(1)
        CM_Choke_graph.plot(0, filter_choke.freq, filter_choke.impedance, "Model")
        CM_Choke_graph.plot(0, filter_choke.raw_freq, filter_choke.raw_impedance, "Data")
        CM_Choke_graph.prettify(0, 'Impedance\n[$\Omega$]', 'CM Choke')
        CM_Choke_graph.draw(self.frames["CM Choke"])

        impedance_CM = Embedded_Graph(1)
        impedance_CM.plot(0, CM_Zs.freq, CM_Zs.geo_mean, "")
        impedance_CM.prettify(0, 'Impedance\n[$\Omega$]', 'CM Noise Impedance')
        impedance_CM.draw(self.frames["CM Impedance"])

        impedance_DM = Embedded_Graph(1)
        impedance_DM.plot(0, DM_Zs.freq, DM_Zs.geo_mean, "")
        impedance_DM.prettify(0, 'Impedance\n[$\Omega$]', 'DM Noise Impedance')
        impedance_DM.draw(self.frames["DM Impedance"])

        self.show_frame("Baseline")

    def show_frame(self, frame_name):
        for key in self.frames.keys():
            self.frames[key].pack_forget()
            if key == frame_name:
                self.frames[key].pack(side="top", fill="both", expand=1)   

    def show_baseline(self):
        self.show_frame("Baseline")
       
    def show_filtered(self):
        self.show_frame("Filtered")
        
    def show_choke(self):
        self.show_frame("CM Choke")
    
    def show_impedance_CM(self):
        self.show_frame("CM Impedance")

    def show_impedance_DM(self):
        self.show_frame("DM Impedance")

    def gen_template(self):
        choke_model.open_template()

    def select_file(self):
        global filter_choke, CM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])
        
        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        # load measurements into choke
        filter_choke = choke_model.Choke(file_path)
        filter_choke.piecewise_linear(CM_base.freq)

        #-----------------------------------------------------------------------------------------------------
    def topology_select_CL(self):
        global CM_Topo
        CM_Topo = None
        return

    def topology_select_LC(self):
        global CM_Topo
        CM_Topo = 1
        return
    
    def topology_select_CyLCx(self):
        global DM_Topo
        DM_Topo = None
        return

    def topology_select_CxLCxCy(self):
        global DM_Topo
        DM_Topo = 1
        return
        #-----------------------------------------------------------------------------------------------------


class FilterCM(Window):
    def __init__(self, parent_frame: ttk.Frame) -> None:
        super().__init__(parent_frame)

        self.title = "CM Filter Design"

        self.interface_frame = ttk.Frame(self.root)
        self.interface_frame.pack(side='left', fill="y")
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side='right', fill='both', expand=1)

        self.setup_interface()

    def setup_interface(self):

        # clear existing frame
        for widget in self.interface_frame.winfo_children():
            widget.destroy()

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        image_frame = ttk.Frame(self.interface_frame)
        image_frame.pack()

        label = ttk.Label(options_frame, text="To find the minimum impedance curve for the CM choke,")
        label.pack()
        label = ttk.Label(options_frame, text="enter the desired margin from the FCC noise limit and the")
        label.pack()
        label = ttk.Label(options_frame, text="Y cap that will be used.")
        label.pack()

        # noise limit
        self.margin = label_entry(options_frame, "Noise Limit Margin [dBuV]", 6)

        # Y Cap
        self.y_cap, self.y_cap_unit = label_entry_unit(options_frame, "Y Cap Value:", 680, Cap_Dict, "pF")

        button(options_frame, "Calculate Min Impedance Curve", self.show_suggested_impedance)

        button(options_frame, "Save Impedance Curve", self.save_curve)

        # Inductor
        label = ttk.Label(options_frame, text="To see the estimated CM performance of a particular choke")
        label.pack()
        label = ttk.Label(options_frame, text="in a LC filter, select data file here:")
        label.pack()

        filter_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(filter_frame, text="Select File", command=self.select_file)
        select_button.pack(side="left")
        label = ttk.Label(filter_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(filter_frame, text="Generate Template", command=self.gen_template)
        select_button.pack(side="left")

        filter_frame.pack()

        button(options_frame, "Est. Choke Performance", self.show_actual_impedance)

    def show_suggested_impedance(self):
        global CM_est, y_cap

        margin = float(self.margin.get())
        y_cap = float(self.y_cap.get()) * self.y_cap_unit.get()

        Limit = emi.Noise_Limit(CM_base.freq)
        Limit.add_margin(margin)

        CM_est = emi.Common_Mode_Estimate()
        CM_est.add_noise(CM_Zs)
        CM_est.CL_topology(y_cap, filter_choke)
        CM_est.find_Z_choke(CM_base, Limit.limit)

        needed_impedance = Embedded_Graph(2)
        needed_impedance.plot(0, CM_base.freq, CM_base.measurement, "Baseline")
        needed_impedance.plot(0, Limit.freq, Limit.FCC, "FCC Limit")
        needed_impedance.plot(0, Limit.freq, Limit.limit, "Limit with Margin")
        needed_impedance.prettify(0, 'Noise\n[dB$\mu$V]', 'CM Noise')
        needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke, "Suggested Choke Impedance")
        needed_impedance.log_scale(1)
        needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        needed_impedance.draw(self.graph_frame)
        return
    
    def show_actual_impedance(self):
        global actual_choke, CM_est, y_cap

        if actual_choke is None:
            # no choke loaded in yet...
            messagebox.showerror("No CM Choke Model Found", "Please select an appropriate CM Choke file.")
        
        margin = float(self.margin.get())
        y_cap = float(self.y_cap.get()) * self.y_cap_unit.get()

        Limit = emi.Noise_Limit(CM_base.freq)
        Limit.add_margin(margin)

        actual_impedance = Embedded_Graph(2)

        if CM_est == None:
            CM_est = emi.Common_Mode_Estimate()
            CM_est.add_noise(CM_Zs)
            CM_est.CL_topology(y_cap, filter_choke)
        else:
            actual_impedance.plot(0, CM_base.freq, CM_est.needed_worst_Z_choke, "Suggested Choke Impedance")
        actual_impedance.plot(0, actual_choke.freq, actual_choke.impedance, "Actual Choke Impedance")
        actual_impedance.log_scale(0)
        actual_impedance.prettify(0, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        CM_est.CL_topology(y_cap, actual_choke)
        CM_est.calculate_noise(CM_base)
        actual_impedance.plot(1, CM_est.freq, CM_est.mean_estimate, "Mean Noise Estimate")
        actual_impedance.plot(1, CM_est.freq, CM_est.worst_estimate, "Worst Noise Estimate")
        actual_impedance.plot(1, Limit.freq, Limit.FCC, "FCC Limit")
        actual_impedance.plot(1, Limit.freq, Limit.limit, "Limit with Margin")
        actual_impedance.prettify(1, 'Noise\n[dB$\mu$V]', 'CM Noise')

        actual_impedance.draw(self.graph_frame)

    def save_curve(self):
        global CM_est

        if CM_est is None:
            messagebox.showerror("No Min CM Curve Found", "Please Calculate Min Impedance Curve.")
            return

        f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        
        CM_est.save_Z_choke(f)

    def gen_template(self):
        choke_model.open_template()

    def select_file(self):
        global actual_choke, CM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])
        
        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        # load measurements into choke
        actual_choke = choke_model.Choke(file_path)
        actual_choke.piecewise_linear(CM_base.freq)

class FilterDM(Window):
    def __init__(self, parent_frame: ttk.Frame) -> None:
        super().__init__(parent_frame)

        self.title = "DM Filter Design"

        self.interface_frame = ttk.Frame(self.root)
        self.interface_frame.pack(side='left', fill="y")
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side='right', fill='both', expand=1)

        self.setup_interface()

    def setup_interface(self):
        # clear existing frame
        for widget in self.interface_frame.winfo_children():
            widget.destroy()

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        image_frame = ttk.Frame(self.interface_frame)
        image_frame.pack()

        label = ttk.Label(options_frame, text="To find the maximum impedance curve for the X Cap, enter the")
        label.pack()
        label = ttk.Label(options_frame, text="desired margin from the FCC noise limit and both the leakage")
        label.pack()
        label = ttk.Label(options_frame, text="impedance of the CM Choke and the Y cap that will be used.")
        label.pack()


        # noise limit
        self.margin = label_entry(options_frame, "Noise Limit Margin [dBuV]", 6)

        # Inductor - leakage
        label = ttk.Label(options_frame, text="CM Choke Leakage data File:")
        label.pack()

        filter_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(filter_frame, text="Select File", command=self.select_file)
        select_button.pack(side="left")
        label = ttk.Label(filter_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(filter_frame, text="Generate Template", command=self.gen_template)
        select_button.pack(side="left")

        filter_frame.pack()
        
        button(options_frame, "Calculate Max Impedance Curve", self.show_suggested_impedance)

        label = ttk.Label(options_frame, text="To see the estimated DM performance of a particular")
        label.pack()
        label = ttk.Label(options_frame, text="capacitor in a Pi filter, specify the capacitance:")
        label.pack()

        # X Cap
        self.x_cap, self.x_cap_unit = label_entry_unit(options_frame, "X Capacitor - Capacitance", 0.15, Cap_Dict, "uF")

        button(options_frame, "Est Cap Performance", self.show_actual_impedance)

    def show_suggested_impedance(self):
        global DM_est, leakage_choke

        margin = float(self.margin.get())

        Limit = emi.Noise_Limit(DM_base.freq)
        Limit.add_margin(margin)

        DM_est = emi.Differential_Mode_Estimate()
        DM_est.add_noise(DM_Zs)
        # use y cap as placeholder for x cap, doesn't matter for finding Zx
        DM_est.PI_topology(y_cap, y_cap, leakage_choke)
        DM_est.find_Zx(DM_base, Limit.limit)

        needed_impedance = Embedded_Graph(2)
        needed_impedance.plot(0, DM_base.freq, DM_base.measurement, "Baseline")
        needed_impedance.plot(0, Limit.freq, Limit.FCC, "FCC Limit")
        needed_impedance.plot(0, Limit.freq, Limit.limit, "Limit with Margin")
        needed_impedance.prettify(0, 'Noise\n[dB$\mu$V]', 'CM Noise')
        needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x, "Suggested X Capacitor Impedance")
        needed_impedance.log_scale(1)
        needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        needed_impedance.draw(self.graph_frame)
        return
    
    def show_actual_impedance(self):
        global leakage_choke, DM_est

        margin = float(self.margin.get())
        x_cap = float(self.x_cap.get()) * self.x_cap_unit.get()

        Limit = emi.Noise_Limit(DM_base.freq)
        Limit.add_margin(margin)

        actual_impedance = Embedded_Graph(2)

        if DM_est == None:
            DM_est = emi.Differential_Mode_Estimate()
            DM_est.add_noise(DM_Zs)
        else:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x, "Suggested X Capacitor Impedance")

        DM_est.PI_topology(x_cap, y_cap, leakage_choke)
        DM_est.calculate_noise(CM_base)

        actual_impedance.plot(0, DM_est.freq, np.abs(DM_est.Z_x_cap), "Actual X Capacitor Impedance")
        actual_impedance.log_scale(0)
        actual_impedance.prettify(0, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        actual_impedance.plot(1, DM_est.freq, DM_est.mean_estimate, "Mean Noise Estimate")
        actual_impedance.plot(1, DM_est.freq, DM_est.worst_estimate, "Worst Noise Estimate")
        actual_impedance.plot(1, Limit.freq, Limit.FCC, "FCC Limit")
        actual_impedance.plot(1, Limit.freq, Limit.limit, "Limit with Margin")
        actual_impedance.prettify(1, 'Noise\n[dB$\mu$V]', 'DM Noise')

        actual_impedance.draw(self.graph_frame)
    
    def gen_template(self):
        choke_model.open_template()

    def select_file(self):
        global leakage_choke, CM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])
        
        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        # load measurements into choke
        leakage_choke = choke_model.Choke(file_path)
        leakage_choke.piecewise_linear(CM_base.freq)

class UnitMenu(ttk.OptionMenu):
    def __init__(self, master: Misc | None, value: str | None = "?", value_dict: dict[str, float] | None = {"error" : 0.0}, **kwargs) -> None:
        self.value_dict = value_dict
        self.unit = tk.StringVar(master, value)
        super().__init__(master, self.unit, self.unit.get(), *self.value_dict.keys(), **kwargs)

    def get(self) -> float:
        return self.value_dict[self.unit.get()]
    
# creates a row frame packed inside parent with a label "TXT" and entry "VAL". Returns Entry.
def label_entry(parent:  ttk.Frame, TXT: str, VAL: float):
        row_frame = ttk.Frame(parent)
        label = ttk.Label(row_frame, text=TXT)
        label.pack(side='left', fill='both', expand=1)
        entry = ttk.Entry(row_frame)
        entry.insert(0, str(VAL))
        entry.pack(side='left')
        row_frame.pack(side='top', fill='both', expand=1)
        return entry

# creates a row frame packed inside parent with a button labeled "TXT" that calls function "FNC".
def button(parent:  ttk.Frame, TXT: str, FNC):
        row_frame = ttk.Frame(parent)
        button = ttk.Button(row_frame, text=TXT, command=FNC)
        button.pack()
        row_frame.pack(side='top', fill='both', expand=1)

# creates a row frame packed inside parent with a label "TXT" and entry "VAL" and unit menu with options "OPTS_DICT" and default value of "DEFAULT". Returns tuple of (Entry, UnitMenu).
def label_entry_unit(parent: ttk.Frame, TXT: str, VAL: float, OPTS_DICT: dict, DEFAULT: str):
    row_frame = ttk.Frame(parent)
    label = ttk.Label(row_frame, text=TXT)
    label.pack(side='left', fill='both', expand=1)
    entry = ttk.Entry(row_frame)
    entry.insert(0, str(VAL))
    entry.pack(side='left')
    menu = UnitMenu(row_frame, DEFAULT, OPTS_DICT)
    menu.pack(side='left')
    row_frame.pack(side='top', fill='both', expand=1)
    return (entry, menu)

# launch the GUI when file is run
if __name__ == '__main__':
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()