
## Name: GUI.py    Author: Abel Keeley/Alex Epps
# Handle UI for EMI filter design tool. Important changes in v8:
#   Cleanup, change CM choke to work with choke_model v2

import tkinter as tk
from tkinter import Misc, ttk
from tkinter import filedialog, messagebox
from tkinter import *
import numpy as np
import os
import pandas as pd

import emi_model as emi
import choke_model
import Capacitor_model
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
nonideal_toggle = None



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

    def select_file_cap_x(self):
        global impedance_xcap, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap = Capacitor_model.Cap(file_path)
        impedance_xcap.piecewise_linear(DM_base.freq)

    def select_file_cap_x_1(self):
        global impedance_xcap_1, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap_1 = Capacitor_model.Cap(file_path)
        impedance_xcap_1.piecewise_linear(DM_base.freq)


    def select_file_cap_actual(self):
        global impedance_xcap_actual, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap_actual = Capacitor_model.Cap(file_path)
        impedance_xcap_actual.piecewise_linear(DM_base.freq)

    def select_file_cap_actual_1(self):
        global impedance_xcap_actual_1, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap_actual_1 = Capacitor_model.Cap(file_path)
        impedance_xcap_actual_1.piecewise_linear(DM_base.freq)

    def select_file_cap_y(self):
        global impedance_ycap, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_ycap = Capacitor_model.Cap(file_path)
        impedance_ycap.piecewise_linear(DM_base.freq)

    def select_file_cap_y_1(self):
        global impedance_ycap_1, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_ycap_1 = Capacitor_model.Cap(file_path)
        impedance_ycap_1.piecewise_linear(DM_base.freq)

    def nonideal_sim(self):
        global nonideal_toggle_x, nonideal_toggle_y
        nonideal_toggle_x = 1
        nonideal_toggle_y = 1

    def ideal_sim(self):
        global nonideal_toggle_x, nonideal_toggle_y
        nonideal_toggle_x = None
        nonideal_toggle_y = None


# UI to allow user to select a measurement file
class FileSelectWindow(Window):
    def __init__(self, parent_frame: ttk.Frame, next_state) -> None:
        super().__init__(parent_frame)

        self.title = "Measurement File Select"

        self.next_state = next_state

        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(side="left", fill="y")

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
        label = ttk.Label(self.root, text = "Alex Rev")
        label.pack(side="top", anchor='center')

        img_frame = ttk.Frame(self.input_frame)
        
        self.logo_img = tk.PhotoImage(file="./images/ge-appliances-a-haier-company-logo-vector.png").subsample(2)

        label = ttk.Label(img_frame, image=self.logo_img)
        label.pack(side="bottom", anchor='center')

        img_frame.pack(side="bottom", anchor='center')


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

        #enter the filter details
        filter_frame = ttk.Frame(option_frame)

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

        #button(filter_frame, "Calculate Noise Impedance", self.calculate_noise_impedance)

        label = ttk.Label(option_frame, text="Select a file with the measured X capacitor impedance for the X capacitor that was used")
        label.pack()
        label = ttk.Label(option_frame, text="X capacitor impedance data file:")
        label.pack()

        bt2_frame = ttk.Frame(option_frame)

        bt3_frame = ttk.Frame(option_frame)

        #filter_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt2_frame, text="Select File", command=self.select_file_cap_x)
        select_button.pack(side="left")
        label = ttk.Label(bt2_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt2_frame, text="Generate Template", command=self.gen_template_cap)
        select_button.pack(side="left")

        label == ttk.Label(bt2_frame, text="")
        label.pack()

        select_button = ttk.Button(bt3_frame, text="Non ideal capacitor sim toggle", command=self.nonideal_sim)
        select_button.pack(side="left")
        label = ttk.Label(bt3_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt3_frame, text="Ideal capacitor sim toggle", command=self.ideal_sim)
        select_button.pack(side="left")
       
        bt2_frame.pack()

        bt3_frame.pack()

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
        global filter_choke, CM_base, CM_filtered, Noise_Floor, CM_Zs, DM_base, DM_filtered, DM_Zs, impedance_xcap, nonideal_toggle_x

        x_cap = float(self.x_cap.get()) * self.x_cap_unit.get()
        

        if filter_choke is None:
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return

        CM_Zs = emi.Noise_Impedance(CM_base, CM_filtered, Noise_Floor)
        CM_Zs.CM_calculation(filter_choke, CM_base.freq)

        if nonideal_toggle_x is None:
            DM_Zs = emi.Noise_Impedance(DM_base, DM_filtered, Noise_Floor)
            DM_Zs.DM_calculation(x_cap, DM_base.freq)

        elif nonideal_toggle_x == 1:
            DM_Zs = emi.Noise_Impedance(DM_base, DM_filtered, Noise_Floor)
            DM_Zs.DM_calculation_actual(impedance_xcap, DM_base.freq)

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

    """def select_file_cap(self):
        global impedance_xcap, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap = Capacitor_model.Cap(file_path)
        impedance_xcap.piecewise_linear(DM_base.freq)"""

    def gen_template_cap(self):
        Capacitor_model.open_template()

    """def nonideal_sim(self):
        global nonideal_toggle
        nonideal_toggle = 1

    def ideal_sim(self):
        global nonideal_toggle
        nonideal_toggle = None"""


class FilterCM(Window):
    global CM_Topo
    def __init__(self, parent_frame: ttk.Frame) -> None:
        super().__init__(parent_frame)

        self.title = "Common Mode Filter Design"

        self.interface_frame = ttk.Frame(self.root)
        self.interface_frame.pack(side='left', fill="y")
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side='right', fill='both', expand=1)

        self.setup_interface()

    def setup_interface(self):
        global CM_Topo
        # clear existing frame
        for widget in self.interface_frame.winfo_children():
            widget.destroy()

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        image_frame = ttk.Frame(self.interface_frame)
        image_frame.pack()

        label = ttk.Label(options_frame, text="To find the minimum impedance curve for the Common Mode choke(L),")
        label.pack()
        label = ttk.Label(options_frame, text="enter the desired margin from the FCC noise limit and the")
        label.pack()
        label = ttk.Label(options_frame, text="Y cap (Cy_1) that will be used.")
        label.pack()
        label = ttk.Label(options_frame, text="")
        label.pack()

        # noise limit
        self.margin = label_entry(options_frame, "Noise Limit Margin [dBuV]", 6)

        # Y Cap
        self.y_cap, self.y_cap_unit = label_entry_unit(options_frame, "Y Cap (Cy_1) Value:", 1000, Cap_Dict, "pF")

        bt4_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt4_frame, text="Non ideal capacitor sim toggle", command=self.nonideal_sim)
        select_button.pack(side="left")
        label = ttk.Label(bt4_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt4_frame, text="Ideal capacitor sim toggle", command=self.ideal_sim)
        select_button.pack(side="left")

        bt4_frame.pack()

        bt5_frame = ttk.Frame(options_frame)

        label = ttk.Label(options_frame, text="For non ideal capacitor sim upload a file with Y capacitor impedance measurements")
        label.pack()

        select_button = ttk.Button(bt5_frame, text="Select File", command=self.select_file_cap_y)
        select_button.pack(side="left")
        label = ttk.Label(bt5_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt5_frame, text="Generate Template", command=self.gen_template_cap)
        select_button.pack(side="left")

        bt5_frame.pack()

        label = ttk.Label(options_frame, text="Upload another file with Y capacitor impedance measurements if a second pair of y capacitors is added")
        label.pack()

        bt6_frame = ttk.Frame(options_frame)
        select_button = ttk.Button(bt6_frame, text="Select File", command=self.select_file_cap_y_1)
        select_button.pack(side="left")

        bt6_frame.pack()

        label = ttk.Label(options_frame, text="")
        label.pack()

        #added option for LC or CL filter for CM emi filter
        label = ttk.Label(options_frame, text="Press the topology button to find the minimum impedance curve to pass FCC.")
        label.pack()

        label = ttk.Label(options_frame, text="")
        label.pack()

        def combine_funcs(*funcs):
            def combined_func(*tp_select, **show):
                for f in funcs:
                    f(*tp_select, **show)
            return combined_func

        #buttons that choose CM LC or CL filter
        #command button will option which calculation to make
        #button(options_frame, "Input - L - Cy - Noise Source", self.topology_select_LC)
        button_color(options_frame, "Topology 1: Input-L-Cy_1-Noise Source", combine_funcs(self.topology_select_LC, self.show_suggested_impedance), "blue")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 2: Input-Cy_1-L-Noise Source", combine_funcs(self.topology_select_CL, self.show_suggested_impedance), "blue")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 3: Input-Cy_2-L-Cy_1-Noise Source", combine_funcs(self.topology_select_CyLCy, self.show_suggested_impedance), "blue")
        self.y_cap_1, self.y_cap_1_unit = label_entry_unit(options_frame, "Y Cap (Cy_2) Value:", 1000, Cap_Dict, "pF")
        label = ttk.Label(options_frame, text="")
        label.pack()

        #button(options_frame, "Calculate Min Impedance Curve", self.show_suggested_impedance)
        label = ttk.Label(options_frame, text="Press the Compare button to compare all topologies.")
        label.pack()

        button(options_frame, "Compare", combine_funcs(self.CM_topology_compare, self.show_suggested_impedance))
    
        label = ttk.Label(options_frame, text="")
        label.pack()
        label = ttk.Label(options_frame, text="Press the save buttons to export suggested minimum choke impedance curve to excel.")
        label.pack()

        
        #save LC curve
        button(options_frame, "Save Topology 1", self.save_curve_LC)
        label = ttk.Label(options_frame, text="")
        label.pack()
        #save CL curve
        button(options_frame, "Save Topology 2", self.save_curve)
        label = ttk.Label(options_frame, text="")
        label.pack()
        #save CLC curve
        button(options_frame, "Save Topology 3", self.save_curve_CLC)
        label = ttk.Label(options_frame, text="")
        label.pack()


        # Inductor
        label = ttk.Label(options_frame, text="To see the estimated Common Mode Noise performance of a particular choke")
        label.pack()
        label = ttk.Label(options_frame, text="click the topology you want to test, select choke impedance file, and import file here:")
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
        global CM_est, y_cap, CM_Topo, y_cap_1, nonideal_toggle_y, impedance_ycap, impedance_ycap_1

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()

        margin = float(self.margin.get())
        #print(impedance_ycap)

        #if nonideal_toggle_y is None:
        y_cap = float(self.y_cap.get()) * self.y_cap_unit.get()
        y_cap_1 = float(self.y_cap_1.get()) * self.y_cap_1_unit.get()
        #elif nonideal_toggle_y == 1:
            #y_cap = impedance_ycap
            #y_cap_1 = impedance_ycap_1

        Limit = emi.Noise_Limit(CM_base.freq)
        Limit.add_margin(margin)

        Limit_avg = emi.Noise_Limit_Avg(CM_base.freq)
        Limit_avg.add_margin(margin)

        CM_est = emi.Common_Mode_Estimate()
        CM_est.add_noise(CM_Zs)

        if nonideal_toggle_y is None:
            if CM_Topo is None:
                CM_est.CL_topology(y_cap, filter_choke)
                CM_est.find_Z_choke(CM_base, Limit_avg.limit)
            elif CM_Topo == 1:
                CM_est.LC_topology(y_cap, filter_choke)
                CM_est.find_Z_choke_LC(CM_base, Limit_avg.limit)
            elif CM_Topo == 3:
                CM_est.CLC_topology(y_cap, y_cap_1, filter_choke)
                CM_est.find_Z_choke_CLC(CM_base, Limit_avg.limit)
            elif CM_Topo == 2:
            
                CM_est.CL_topology_math(y_cap, filter_choke)
                CM_est.find_Z_choke(CM_base, Limit_avg.limit)
            
                CM_est.LC_topology_math(y_cap, filter_choke)
                CM_est.find_Z_choke_LC(CM_base, Limit_avg.limit)

                CM_est.CLC_topology_math(y_cap, y_cap_1, filter_choke)
                CM_est.find_Z_choke_CLC(CM_base, Limit_avg.limit)

                CLCz = np.mean(CM_est.needed_worst_Z_choke_CLC)
                print("CLCz mean")
                print(CLCz)
                CLCz1 = np.max(CM_est.needed_worst_Z_choke_CLC)
                print("Max impedance required")
                print(CLCz1)
                print("")

                LCz = np.mean(CM_est.needed_worst_Z_choke_LC)
                print("LCz mean")
                print(LCz)
                LCz1 = np.max(CM_est.needed_worst_Z_choke_LC)
                print("Max impedance required")
                print(LCz1)
                print("")

                CLz = np.mean(CM_est.needed_worst_Z_choke)
                print("CLz mean")
                print(CLz)
                CLz1 = np.max(CM_est.needed_worst_Z_choke)
                print("Max impedance required")
                print(CLz1)
                print("")

                if CLz < LCz and CLz < CLCz:
                    label = ttk.Label(options_frame, text="Topology 2 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("CLz is better")
                elif LCz < CLz and LCz < CLCz:
                    label = ttk.Label(options_frame, text="Topology 1 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("LCz is better")
                elif CLCz < LCz and CLCz < CLz:
                    label = ttk.Label(options_frame, text="Topology 3 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("CLCz is better")
                else: 
                    label = ttk.Label(options_frame, text="No difference between topologies")
                    label.pack()
                
        elif nonideal_toggle_y == 1:
            if CM_Topo is None:
                CM_est.CL_topology_array_input(impedance_ycap, filter_choke)
                CM_est.find_Z_choke(CM_base, Limit_avg.limit)
            elif CM_Topo == 1:
                CM_est.LC_topology_array_input(impedance_ycap, filter_choke)
                CM_est.find_Z_choke_LC(CM_base, Limit_avg.limit)
            elif CM_Topo == 3:
                CM_est.CLC_topology_array_input(impedance_ycap, impedance_ycap_1, filter_choke)
                CM_est.find_Z_choke_CLC(CM_base, Limit_avg.limit)
            elif CM_Topo == 2:
                CM_est.CL_topology_math_array_input(impedance_ycap, filter_choke)
                CM_est.find_Z_choke(CM_base, Limit_avg.limit)
            
                CM_est.LC_topology_math_array_input(impedance_ycap, filter_choke)
                CM_est.find_Z_choke_LC(CM_base, Limit_avg.limit)

                CM_est.CLC_topology_math_array_input(impedance_ycap, impedance_ycap_1, filter_choke)
                CM_est.find_Z_choke_CLC(CM_base, Limit_avg.limit)

                CLCz = np.mean(CM_est.needed_worst_Z_choke_CLC)
                print("CLCz mean")
                print(CLCz)
                CLCz1 = np.max(CM_est.needed_worst_Z_choke_CLC)
                print("Max impedance required")
                print(CLCz1)
                print("")

                LCz = np.mean(CM_est.needed_worst_Z_choke_LC)
                print("LCz mean")
                print(LCz)
                LCz1 = np.max(CM_est.needed_worst_Z_choke_LC)
                print("Max impedance required")
                print(LCz1)
                print("")

                CLz = np.mean(CM_est.needed_worst_Z_choke)
                print("CLz mean")
                print(CLz)
                CLz1 = np.max(CM_est.needed_worst_Z_choke)
                print("Max impedance required")
                print(CLz1)
                print("")

            
                if CLz < LCz and CLz < CLCz:
                    label = ttk.Label(options_frame, text="Topology 2 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("CLz is better")
                elif LCz < CLz and LCz < CLCz:
                    label = ttk.Label(options_frame, text="Topology 1 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("LCz is better")
                elif CLCz < LCz and CLCz < CLz:
                    label = ttk.Label(options_frame, text="Topology 3 is the best topology for this noise source", foreground="green")
                    label.pack()
                    print("CLCz is better")
                else: 
                    label = ttk.Label(options_frame, text="No difference between topologies")
                    label.pack()

            
        needed_impedance = Embedded_Graph(2)
        needed_impedance.plot(0, CM_base.freq, CM_base.measurement, "Baseline")
        needed_impedance.plot(0, Limit.freq, Limit.FCC, "FCC Quasi Limit")
        needed_impedance.plot(0, Limit.freq, Limit.limit, "Quasi Limit with Margin")
        needed_impedance.plot(0, Limit_avg.freq, Limit_avg.FCC, "FCC Avg Limit")
        needed_impedance.plot(0, Limit_avg.freq, Limit_avg.limit, "Avg Limit with Margin")
        needed_impedance.prettify(0, 'Noise\n[dB$\mu$V]', 'Common Mode Noise without filter')
       
        if CM_Topo is None:
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke, "TP2: Suggested Choke Impedance")
        elif CM_Topo == 1:
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke_LC, "TP1: Suggested Choke Impedance")
        elif CM_Topo == 3: 
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke_CLC, "TP3: Suggested Choke Impedance")
        elif CM_Topo == 2:
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke, "TP2: Suggested Choke Impedance")
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke_LC, "TP1: Suggested Choke Impedance")
            needed_impedance.plot(1, CM_base.freq, CM_est.needed_worst_Z_choke_CLC, "TP3: Suggested Choke Impedance")
        needed_impedance.log_scale(1)
        needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        needed_impedance.draw(self.graph_frame)
        return 
    
    def show_actual_impedance(self):
        global actual_choke, CM_est, y_cap, CM_Topo

        if actual_choke is None:
            # no choke loaded in yet...
            messagebox.showerror("No CM Choke Model Found", "Please select an appropriate CM Choke file.")
        
        margin = float(self.margin.get())
        y_cap = float(self.y_cap.get()) * self.y_cap_unit.get()
        y_cap_1 = float(self.y_cap_1.get()) * self.y_cap_unit.get()
        

        Limit = emi.Noise_Limit(CM_base.freq)
        Limit.add_margin(margin)

        Limit_avg = emi.Noise_Limit_Avg(CM_base.freq)
        Limit_avg.add_margin(margin)

        actual_impedance = Embedded_Graph(2)

        if CM_est == None:
            CM_est = emi.Common_Mode_Estimate()
            CM_est.add_noise(CM_Zs)
            if nonideal_toggle_y is None:
                if CM_Topo is None:
                    CM_est.CL_topology(y_cap, filter_choke)
                elif CM_Topo == 1:
                    CM_est.LC_topology(y_cap, filter_choke)
                elif CM_Topo == 3:
                    CM_est.CLC_topology(y_cap, y_cap_1, filter_choke)
            elif nonideal_toggle_y == 1:
                if CM_Topo is None:
                    CM_est.CL_topology_array_input(impedance_ycap, filter_choke)
                elif CM_Topo == 1:
                    CM_est.LC_topology_array_input(impedance_ycap, filter_choke)
                elif CM_Topo == 3:
                    CM_est.CLC_topology_array_input(impedance_ycap, impedance_ycap_1, filter_choke)
        else:
            if CM_Topo is None:
                actual_impedance.plot(0, CM_base.freq, CM_est.needed_worst_Z_choke, "TP2: Suggested Choke Impedance")
            elif CM_Topo == 1:
                actual_impedance.plot(0, CM_base.freq, CM_est.needed_worst_Z_choke_LC, "TP1: Suggested Choke Impedance")
            elif CM_Topo == 3:
                actual_impedance.plot(0, CM_base.freq, CM_est.needed_worst_Z_choke_CLC, "TP3: Suggested Choke Impedance")
        actual_impedance.plot(0, actual_choke.freq, actual_choke.impedance, "Actual Choke Impedance Imported")
        actual_impedance.log_scale(0)
        actual_impedance.prettify(0, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        if nonideal_toggle_y is None:    
            if CM_Topo is None:
                CM_est.CL_topology(y_cap, actual_choke)
            elif CM_Topo == 1:
                CM_est.LC_topology(y_cap, actual_choke)
            elif CM_Topo == 3:
                CM_est.CLC_topology(y_cap, y_cap_1, actual_choke)
            CM_est.calculate_noise(CM_base)
            actual_impedance.plot(1, CM_est.freq, CM_est.mean_estimate, "Mean Noise Estimate")
            actual_impedance.plot(1, CM_est.freq, CM_est.worst_estimate, "Worst Noise Estimate")
            actual_impedance.plot(1, Limit.freq, Limit.FCC, "FCC Quasi Limit")
            actual_impedance.plot(1, Limit.freq, Limit.limit, "Quasi Limit with Margin")
            actual_impedance.plot(1, Limit_avg.freq, Limit_avg.FCC, "FCC Avg Limit")
            actual_impedance.plot(1, Limit_avg.freq, Limit.limit, "Avg Limit with Margin")
            actual_impedance.log_scale_x(1)
            actual_impedance.prettify(1, 'Noise\n[dB$\mu$V]', 'Common Mode Noise with filter')
        elif nonideal_toggle_y == 1:
            if CM_Topo is None:
                CM_est.CL_topology_array_input(impedance_ycap, actual_choke)
            elif CM_Topo == 1:
                CM_est.LC_topology_array_input(impedance_ycap, actual_choke)
            elif CM_Topo == 3:
                CM_est.CLC_topology_array_input(impedance_ycap, impedance_ycap_1, actual_choke)
            CM_est.calculate_noise(CM_base)
            actual_impedance.plot(1, CM_est.freq, CM_est.mean_estimate, "Mean Noise Estimate")
            actual_impedance.plot(1, CM_est.freq, CM_est.worst_estimate, "Worst Noise Estimate")
            actual_impedance.plot(1, Limit.freq, Limit.FCC, "FCC Quasi Limit")
            actual_impedance.plot(1, Limit.freq, Limit.limit, "Quasi Limit with Margin")
            actual_impedance.plot(1, Limit_avg.freq, Limit_avg.FCC, "FCC Avg Limit")
            actual_impedance.plot(1, Limit_avg.freq, Limit_avg.limit, "Avg Limit with Margin")
            actual_impedance.log_scale_x(1)
            actual_impedance.prettify(1, 'Noise\n[dB$\mu$V]', 'Common Mode Noise with filter')

        actual_impedance.draw(self.graph_frame)

        worst_est_max = np.max(CM_est.worst_estimate)
        limit_fcc_min = np.min(Limit.FCC)
        if worst_est_max > limit_fcc_min:
            messagebox.showwarning("FCC limit exceeded", "Try a different choke, Y capacitor, and or topology")
            return



    def save_curve(self):
        global CM_est

        if CM_est is None:
            messagebox.showerror("No Min CM Curve Found", "Please Calculate Min Impedance Curve.")
            return

        f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        
        CM_est.save_Z_choke(f)

    def save_curve_LC(self):
        global CM_est

        if CM_est is None:
            messagebox.showerror("No Min CM Curve Found", "Please Calculate Min Impedance Curve.")
            return

        f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        
        CM_est.save_Z_choke_LC(f)

    def save_curve_CLC(self):
        global CM_est

        if CM_est is None:
            messagebox.showerror("No Min CM Curve Found", "Please Calculate Min Impedance Curve.")
            return
        
        f = filedialog.asksaveasfile()
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        CM_est.save_Z_choke_CLC(f)



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

    def topology_select_CL(self):
        global CM_Topo
        CM_Topo = None
        self.show_suggested_impedance
        return
        
        
    def topology_select_LC(self):
        global CM_Topo
        CM_Topo = 1
        self.show_suggested_impedance
        return

    def CM_topology_compare(self):
        global CM_Topo
        CM_Topo = 2
        self.show_suggested_impedance
        return

    def topology_select_CyLCy(self):
        global CM_Topo
        CM_Topo = 3
        self.show_suggested_impedance
        return
    
    """def nonideal_sim(self):
        global nonideal_toggle_y
        nonideal_toggle_y = 1

    def ideal_sim(self):
        global nonideal_toggle_y
        nonideal_toggle_y = None"""

    """def select_file_cap(self):
        global impedance_xcap, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_ycap = Capacitor_model.Cap(file_path)
        impedance_ycap.piecewise_linear(DM_base.freq)"""

    def gen_template_cap(self):
        Capacitor_model.open_template()


class FilterDM(Window):
    global DM_Topo
    def __init__(self, parent_frame: ttk.Frame) -> None:
        super().__init__(parent_frame)

        self.title = "Differential Mode Filter Design"

        self.interface_frame = ttk.Frame(self.root)
        self.interface_frame.pack(side='left', fill="y")
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(side='right', fill='both', expand=1)

        self.setup_interface()

    def setup_interface(self):
        global DM_Topo
        # clear existing frame
        for widget in self.interface_frame.winfo_children():
            widget.destroy()

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        image_frame = ttk.Frame(self.interface_frame)
        image_frame.pack()

        label = ttk.Label(options_frame, text="Enter the desired margin from the FCC noise limit")
        label.pack()
        # noise limit
        self.margin = label_entry(options_frame, "Noise Limit Margin [dBuV]", 6)

        #label = ttk.Label(options_frame, text="Enter the X Capacitor that was used.")
        #label.pack()
        #self.x_cap0, self.x_cap_unit = label_entry_unit(options_frame, "X Capacitor (Cx) - Capacitance", 0.15, Cap_Dict, "uF")

        label = ttk.Label(options_frame, text="")
        label.pack()

        label = ttk.Label(options_frame, text="Select a file with the measured choke leakage impedance for the choke you want to simulate")
        label.pack()
        #label = ttk.Label(options_frame, text="impedance of the CM Choke and the Y cap that will be used.")
        #label.pack()

        def combine_funcs(*funcs):
            def combined_func(*tp_select, **show):
                for f in funcs:
                    f(*tp_select, **show)
            return combined_func
        # noise limit
        #self.margin = label_entry(options_frame, "Noise Limit Margin [dBuV]", 6)

        # Inductor - leakage
        label = ttk.Label(options_frame, text="CM Choke Leakage data File:")
        label.pack()

        bt_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt_frame, text="Select File", command=self.select_file)
        select_button.pack(side="left")
        label = ttk.Label(bt_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt_frame, text="Generate Template", command=self.gen_template)
        select_button.pack(side="left")
        

        bt_frame.pack()


        label = ttk.Label(options_frame, text="Enter X capacitor values to calculate differential mode noise without filter")
        label.pack()

        self.x_cap_1, self.x_cap_unit = label_entry_unit(options_frame, "X Capacitor input (Cx1) - Capacitance", 0.15, Cap_Dict, "uF")
        self.x_cap_2, self.x_cap_unit = label_entry_unit(options_frame, "X Capacitor output (Cx2) - Capacitance", 0.15, Cap_Dict, "uF")

        bt7_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt7_frame, text="Non ideal capacitor sim toggle", command=self.nonideal_sim)
        select_button.pack(side="left")
        label = ttk.Label(bt7_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt7_frame, text="Ideal capacitor sim toggle", command=self.ideal_sim)
        select_button.pack(side="left")

        bt7_frame.pack()

        bt8_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt8_frame, text="Select File", command=self.select_file_cap_x)
        select_button.pack(side="left")
        label = ttk.Label(bt8_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt8_frame, text="Generate Template", command=self.gen_template_cap)
        select_button.pack(side="left")

        bt8_frame.pack()

        label = ttk.Label(options_frame, text="Upload an additional file if a second x cap is used")
        label.pack()

        bt9_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt9_frame, text="Select File", command=self.select_file_cap_x_1)
        select_button.pack(side="left")

        bt9_frame.pack()

        label = ttk.Label(options_frame, text="Push the topology button to find the minimum impedance curve to pass FCC")
        label.pack()
        

        button_color(options_frame, "Topology 1: Input-Cy-L-Cx-Noise Source", combine_funcs(self.topology_select_CyLCx, self.show_suggested_impedance), "blue")
        select_button.pack(side="left")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 2: Input-Cx1-L-Cy-Cx2-Noise Source", combine_funcs(self.topology_select_CxLCyCx, self.show_suggested_impedance), "blue")
        select_button.pack(side="left")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 3: Input-Cy-Cx-L-Noise Source", combine_funcs(self.topology_select_CyCxL, self.show_suggested_impedance), "blue")
        select_button.pack(side="left")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 4: Input-L-Cy-Cx-Noise Source", combine_funcs(self.topology_select_LCyCx, self.show_suggested_impedance), "blue")
        select_button.pack(side="left")
        label = ttk.Label(options_frame, text="")
        label.pack()
        button_color(options_frame, "Topology 5: Input-Cx-L-Cy-Noise Source", combine_funcs(self.topology_select_CxLCy, self.show_suggested_impedance), "blue")
        select_button.pack(side="left")

        label = ttk.Label(options_frame, text="")
        label.pack()
        
        label = ttk.Label(options_frame, text="Push the Compare button to compare choke impedance per topology")
        label.pack()
        button(options_frame, "Compare", combine_funcs(self.DM_topology_compare, self.show_suggested_impedance))
        select_button.pack(side="left")

        label = ttk.Label(options_frame, text="")
        label.pack()
        

        label = ttk.Label(options_frame, text="To see the estimated Differential Mode performance of a particular")
        label.pack()
        label = ttk.Label(options_frame, text="X capacitor in this topology, click on the topology you want to test and specify the capacitance:")
        label.pack()

        # X Cap
        self.x_cap_0, self.x_cap_unit = label_entry_unit(options_frame, "X Capacitor - Capacitance", 0.15, Cap_Dict, "uF")

        button(options_frame, "Est Cap Performance", self.show_actual_impedance)

        label = ttk.Label(options_frame, text="Select a file with the measured X capacitor impedance for the X capacitor that was used")
        label.pack()
        label = ttk.Label(options_frame, text="X capacitor impedance data file:")
        label.pack()

        bt1_frame = ttk.Frame(options_frame)

        #filter_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt1_frame, text="Select File", command=self.select_file_cap_actual)
        select_button.pack(side="left")
        label = ttk.Label(bt1_frame, text = "or")
        label.pack(side="left")
        select_button = ttk.Button(bt1_frame, text="Generate Template", command=self.gen_template_cap)
        select_button.pack(side="left")

        label = ttk.Label(options_frame, text="Upload an additional X capacitor impedance measurement file if a second X capacitor is used")
        label.pack()

        bt1_frame.pack()

        bt9_frame = ttk.Frame(options_frame)

        select_button = ttk.Button(bt9_frame, text="Select File", command=self.select_file_cap_actual_1)
        select_button.pack(side="left")

        bt9_frame.pack()
        
        #button(options_frame, "Est Non Ideal Cap Performance", self.show_actual_impedance)

        

        
    def show_suggested_impedance(self):
        global DM_est, leakage_choke, DM_Topo, impedance_xcap, impedance_ycap

        #x_cap0 = float(self.x_cap0.get()) * self.x_cap_unit.get()

        #x_cap0 = emi.DM_calculation
        if nonideal_toggle_x is None:
            x_cap_1 = float(self.x_cap_1.get()) * self.x_cap_unit.get()
            x_cap_2 = float(self.x_cap_2.get()) * self.x_cap_unit.get()
            x_cap0 = float(self.x_cap_0.get()) * self.x_cap_unit.get()
        elif nonideal_toggle_x == 1:
            x_cap0 = impedance_xcap
            x_cap_1 = impedance_xcap
            x_cap_2 = impedance_xcap_1

        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        print("x_cap")
        #print(x_cap0)
        margin = float(self.margin.get())

        Limit = emi.Noise_Limit(DM_base.freq)
        Limit.add_margin(margin)

        DM_est = emi.Differential_Mode_Estimate()
        DM_est.add_noise(DM_Zs)
        # use y cap as placeholder for x cap, doesn't matter for finding Zx

        if nonideal_toggle is None:
            if DM_Topo is None:
                DM_est.PI_topology(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx(DM_base, Limit.limit)
            elif DM_Topo == 1:
                DM_est.PI2_topology(x_cap_1, x_cap_2, y_cap, leakage_choke)
                DM_est.find_Zx_PI2(DM_base, Limit.limit)
            elif DM_Topo == 2:
                DM_est.CCL_topology(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CCL(DM_base, Limit.limit)
            elif DM_Topo == 3:
                DM_est.LCC_topology(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_LCC(DM_base, Limit.limit)
            elif DM_Topo == 5:
                DM_est.CxLCy_topology(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CxLCy(DM_base, Limit.limit)
            elif DM_Topo == 4:
                DM_est.PI_topology_math(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx(DM_base, Limit.limit)
                DM_est.PI2_topology_math(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_PI2(DM_base, Limit.limit)
                DM_est.CCL_topology_math(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CCL(DM_base, Limit.limit)
                DM_est.LCC_topology_math(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_LCC(DM_base, Limit.limit)
                DM_est.CxLCy_topology_math(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CxLCy(DM_base, Limit.limit)
        elif nonideal_toggle == 1:
            if DM_Topo is None:
                DM_est.PI_topology_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx(DM_base, Limit.limit)
            elif DM_Topo == 1:
                DM_est.PI2_topology_array_input(x_cap_1, x_cap_2, y_cap, leakage_choke)
                DM_est.find_Zx_PI2(DM_base, Limit.limit)
            elif DM_Topo == 2:
                DM_est.CCL_topology_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CCL(DM_base, Limit.limit)
            elif DM_Topo == 3:
                DM_est.LCC_topology_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_LCC(DM_base, Limit.limit)
            elif DM_Topo == 5:
                DM_est.CxLCy_topology_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CxLCy(DM_base, Limit.limit)
            elif DM_Topo == 4:
                DM_est.PI_topology_math_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx(DM_base, Limit.limit)
                DM_est.PI2_topology_math_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_PI2(DM_base, Limit.limit)
                DM_est.CCL_topology_math_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CCL(DM_base, Limit.limit)
                DM_est.LCC_topology_math_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_LCC(DM_base, Limit.limit)
                DM_est.CxLCy_topology_math_array_input(x_cap0, y_cap, leakage_choke)
                DM_est.find_Zx_CxLCy(DM_base, Limit.limit)

            YLX = np.mean(DM_est.needed_Z_x)
            print("ylx")
            print(YLX)
            YXL = np.mean(DM_est.needed_Z_x_CCL)
            print("yxl")
            print(YXL)
            XLYX = np.mean(DM_est.needed_Z_x_PI2)
            print("xlyx")
            print(XLYX)
            LYX = np.mean(DM_est.needed_Z_x_LCC)
            print("lyx")
            print(LYX)
            XLY1 = np.mean(DM_est.needed_Z_x_CxLCy)
            XLY = abs(XLY1)
            print("xly")
            print(XLY)

            if YLX > YXL and YLX > XLYX and YLX > LYX and YLX > XLY:
                label = ttk.Label(options_frame, text="Topology 1 is the best topology for this noise source", foreground="green")
                label.pack()
            elif XLYX > YLX and XLYX > YXL and YLX > LYX and XLYX > XLY:
                label = ttk.Label(options_frame, text="Topology 2 is the best topology for this noise source", foreground="green")
                label.pack()
            elif YXL > YLX and YXL > XLYX and YXL > LYX and YXL > XLY:
                label = ttk.Label(options_frame, text="Topology 3 is the best topology for this noise source", foreground="green")
                label.pack()
            elif LYX > YLX and LYX > XLYX and LYX > YXL and LYX > XLY:
                label = ttk.Label(options_frame, text="Topology 4 is the best topology for this noise source", foreground="green")
                label.pack()
            elif XLY > YLX and XLY > XLYX and XLY > YXL and XLY > LYX:
                label = ttk.Label(options_frame, text="Topology 5 is the best topology for this noise source", foreground="green")
                label.pack()

        needed_impedance = Embedded_Graph(2)
        needed_impedance.plot(0, DM_base.freq, DM_base.measurement, "Baseline")
        needed_impedance.plot(0, Limit.freq, Limit.FCC, "FCC Limit")
        needed_impedance.plot(0, Limit.freq, Limit.limit, "Limit with Margin")
        needed_impedance.prettify(0, 'Noise\n[dB$\mu$V]', 'Differential Mode Noise without filter')
        if DM_Topo is None:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x, "TP1: Suggested X Capacitor Impedance")
            needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif DM_Topo == 1:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_PI2, "TP2: Suggested X Capacitor Impedance")
            needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif DM_Topo == 2:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_CCL, "TP3: Suggested X Capacitor Impedance")
            needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif DM_Topo == 3:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_LCC, "TP4: Suggested X Capacitor Impedance")
            needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif DM_Topo == 5:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_CxLCy, "TP5: Suggested X Capacitor Impedance")
            needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif DM_Topo == 4:
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x, "TP1: Suggested X Capacitor Impedance")
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_PI2, "TP2: Suggested X Capacitor Impedance")
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_CCL, "TP3: Suggested X Capacitor Impedance")
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_LCC, "TP4: Suggested X Capacitor Impedance")
            needed_impedance.plot(1, DM_base.freq, DM_est.needed_Z_x_CxLCy, "TP5: Suggested X Capacitor Impedance")
        needed_impedance.log_scale(1)
        needed_impedance.prettify(1, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        needed_impedance.draw(self.graph_frame)
        return
    
    def show_actual_impedance(self):
        global leakage_choke, DM_est, DM_Topo

        margin = float(self.margin.get())
        if nonideal_toggle_x is None:
            x_cap_0 = float(self.x_cap_0.get()) * self.x_cap_unit.get()
        elif nonideal_toggle_x == 1:
            x_cap_0 = impedance_xcap_actual
            x_cap_01 = impedance_xcap_actual_1

        #x_cap_0 = float(self.x_cap_0.get()) * self.x_cap_unit.get()

        Limit = emi.Noise_Limit(DM_base.freq)
        Limit.add_margin(margin)

        actual_impedance = Embedded_Graph(2)

        if DM_est == None:
            DM_est = emi.Differential_Mode_Estimate()
            DM_est.add_noise(DM_Zs)
        elif DM_Topo is None:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x, "TP1: Suggested X Capacitor Impedance")
        elif DM_Topo == 1:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x_PI2, "TP2: Suggested X Capacitor Impedance")
        elif DM_Topo == 2:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x_CCL, "TP3: Suggested X Capacitor Impedance")
        elif DM_Topo == 3:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x_LCC, "TP4: Suggested X Capacitor Impedance")
        elif DM_Topo == 5:
            actual_impedance.plot(0, DM_base.freq, DM_est.needed_Z_x_CxLCy, "TP5: Suggested X Capacitor Impedance")

        if nonideal_toggle_x is None:
            if DM_Topo is None:
                DM_est.PI_topology(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 1:
                DM_est.PI2_topology_one_x(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 2:
                DM_est.CCL_topology(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 3:
                DM_est.LCC_topology(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 5:
                DM_est.CxLCy_topology(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
        elif nonideal_toggle_x == 1:
            if DM_Topo is None:
                DM_est.PI_topology_array_input(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 1:
                DM_est.PI2_topology_array_input(x_cap_0, x_cap_01, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 2:
                DM_est.CCL_topology_array_input(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 3:
                DM_est.LCC_topology_array_input(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
            elif DM_Topo == 5:
                DM_est.CxLCy_topology_array_input(x_cap_0, y_cap, leakage_choke)
                DM_est.calculate_noise(DM_base)
        

        #looking for two values in PI2, thats why theres an error
        if nonideal_toggle_x is None:
            actual_impedance.plot(0, DM_est.freq, np.abs(DM_est.Z_x_cap), "Actual X Capacitor Impedance")
            actual_impedance.log_scale(0)
            actual_impedance.prettify(0, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')
        elif nonideal_toggle_x == 1:
            actual_impedance.plot(0, DM_est.freq, np.abs(x_cap_0), "Actual X1 Capacitor Impedance")
            if DM_Topo == 1:
                actual_impedance.plot(0, DM_est.freq, np.abs(x_cap_01), "Actual X2 Capacitor Impedance")
            actual_impedance.log_scale(0)
            actual_impedance.prettify(0, 'Impedance[$\Omega$]', 'Impedance Vs Freq, Log Scale')

        actual_impedance.plot(1, DM_est.freq, DM_est.mean_estimate, "Mean Noise Estimate")
        actual_impedance.plot(1, DM_est.freq, DM_est.worst_estimate, "Worst Noise Estimate")
        actual_impedance.plot(1, Limit.freq, Limit.FCC, "FCC Limit")
        actual_impedance.plot(1, Limit.freq, Limit.limit, "Limit with Margin")
        actual_impedance.log_scale_x(1)
        actual_impedance.prettify(1, 'Noise\n[dB$\mu$V]', 'Differential Mode Noise with filter')
        
        actual_impedance.draw(self.graph_frame)

        worst_est_max = np.max(DM_est.worst_estimate)
        limit_fcc_max = np.max(Limit.FCC)
        if worst_est_max > limit_fcc_max:
            messagebox.showwarning("FCC limit exceeded", "Try a new X capacitor or topology")
            return
    
    def gen_template(self):
        choke_model.open_template()

    def gen_template_cap(self):
        Capacitor_model.open_template()

    def select_file(self):
        global leakage_choke, CM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])
        
        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        # load measurements into choke
        leakage_choke = choke_model.Choke(file_path)
        leakage_choke.piecewise_linear(CM_base.freq)

    def select_file_cap(self):
        global impedance_xcap, DM_base
        file_path = filedialog.askopenfilename(filetypes=[(".csv","*.csv")])

        if file_path == "":
            messagebox.showerror("No File Selected", "Please select a valid data file")
            return
        
        impedance_xcap = Capacitor_model.Cap(file_path)
        impedance_xcap.piecewise_linear(DM_base.freq)

    def topology_select_CyLCx(self):
        global DM_Topo
        DM_Topo = None

    def topology_select_CxLCyCx(self):
        options_frame = ttk.Frame(self.interface_frame)
        options_frame.pack()
        global DM_Topo
        DM_Topo = 1

    def topology_select_CyCxL(self):
        global DM_Topo
        DM_Topo = 2
    
    def topology_select_LCyCx(self):
        global DM_Topo
        DM_Topo = 3

    def topology_select_CxLCy(self):
        global DM_Topo
        DM_Topo = 5

    def DM_topology_compare(self):
        global DM_Topo
        DM_Topo = 4

    """def DM_topology_actual_capacitor_impedance(self):
        global Imported_Z_cap
        Imported_Z_cap = 1"""

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

def button_color(parent: ttk.Frame, TXT: str, FNC, clr):
        row_frame = ttk.Frame(parent)
        button = tk.Button(row_frame, text=TXT, command=FNC, activebackground=clr)
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