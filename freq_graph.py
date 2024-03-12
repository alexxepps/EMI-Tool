## Name: freq_graph.py    Author: Abel Keeley
# Handles the busy work of drawing graphs.

from matplotlib import pyplot as plt
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import numpy as np
import math

# Simplifies the creation and display of common graphs with frequency as X-axis.
class Embedded_Graph():
    # number of subplots in this graph needs to specified
    def __init__(self, num_subplots: int) -> None:
        self.subplots = num_subplots
        self.fig, self.axs = plt.subplots(self.subplots)
        if num_subplots < 2:
            self.axs = [self.axs]
        self.unit_dict = {}
        pass

    def plot(self, subplot: int, x_data: np.ndarray, y_data: np.ndarray, label: str):
        # select units and store
        self.unit_dict[subplot] = self.select_unit(x_data)
        unit, denominator = self.unit_dict[subplot]
        # establish initial zoom level
        self.zoom_min = x_data.min()/denominator
        self.zoom_max = x_data.max()/denominator
        # plot the y data vs the scaled x data
        self.axs[subplot].plot(x_data/denominator, y_data, label = label)

    # sets a specific subplot to log scaling
    def log_scale(self, subplot: int):
        self.axs[subplot].set_yscale("log")
        self.axs[subplot].set_xscale("log")
        
    # selects frequency unit - MHz, KHz, or Hz, with appropriate divisor
    def select_unit(self, x_data: np.ndarray) -> tuple:
        delta = abs(x_data.max() - x_data.min())
        if delta >= 2 * 10 ** 6:
            return ("MHz", 10 ** 6)
        if delta >= 2 * 10 ** 3:
            return ("KHz", 10 ** 3)
        return ("Hz", 1)
    
    # makes a given subplot look nice, with a specified y-axis label and title
    def prettify(self, subplot: int, ylabel: str, title: str):
        # find correct unit
        unit, denominator = self.unit_dict[subplot]

        # add axis labels
        self.axs[subplot].set_xlabel('Frequency [' + unit + ']')
        self.axs[subplot].set_ylabel(ylabel, rotation = 0, horizontalalignment = 'right')

        # Suppress the warning about missing artists in the legend
        logger = logging.getLogger("matplotlib.legend")
        logger.setLevel(logging.ERROR)

        # create legend and add title
        self.axs[subplot].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.axs[subplot].set_title(title)
        plt.tight_layout(pad=1.5)

    # draws this graph onto the specified graph frame
    def draw(self, graph_frame: ttk.Frame):
        self.master = graph_frame
        # Clear the graph frame before drawing a new graph
        for widget in self.master.winfo_children():
            widget.destroy()

        # set appropraite axis limits based on zoom
        for i in range(self.subplots):
            self.axs[i].set_xlim([self.zoom_min, self.zoom_max])
            # resize y limits after zoom...
            self.axs[i].set_ylim(self.update_y_lims(self.axs[i]))

        #draw on frame
        canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        #close fig
        plt.close(self.fig)

        self.add_zoom_widget()

    # adds interface for user to control graph zooming
    def add_zoom_widget(self):
        zoom_frame = ttk.Frame(self.master)
        # find correct unit
        unit, denominator = self.unit_dict[0]

        # setup GUI elements
        label = ttk.Label(zoom_frame, text=unit)
        self.x_min_entry = ttk.Entry(zoom_frame)
        self.x_min_entry.insert(0, self.zoom_min)
        self.x_max_entry = ttk.Entry(zoom_frame)
        self.x_max_entry.insert(0, self.zoom_max)

        self.x_min_entry.grid(column = 1, row = 1, sticky="w")
        label = ttk.Label(zoom_frame, text=unit+"\t")
        label.grid(column = 2, row = 1, sticky="w")
        self.x_max_entry.grid(column = 3, row = 1, sticky="e")
        label = ttk.Label(zoom_frame, text=unit)
        label.grid(column = 4, row = 1, sticky="e")
        button = ttk.Button(zoom_frame, text="Zoom", command=self.zoom)
        button.grid(column=5, row=1, sticky="e")
        button = ttk.Button(zoom_frame, text="Reset", command=self.reset)
        button.grid(column=6, row=1, sticky="e")

        zoom_frame.pack()

    # get zoom bounds from GUI elements, then trigger a redraw to display new zoom level
    def zoom(self):
        self.zoom_min = float(self.x_min_entry.get())
        self.zoom_max = float(self.x_max_entry.get())
        self.draw(self.master)

    # reset zoom bounds to the axis bounds, then triggers a redraw
    def reset(self):
        self.zoom_min = min(self.axs[0].lines[0].get_xdata())
        self.zoom_max = max(self.axs[0].lines[0].get_xdata())
        self.draw(self.master)

    # handles updating the y axis limits, so data is easy to see and isn't too close to the upper or lower bound
    def update_y_lims(self, axes):
        y_min = None
        y_max = None
        # cycle through all the lines (could be multiple per subplot...)
        for line in axes.get_lines():
            x_data = line.get_xdata()
            x_range = np.logical_and(x_data >= self.zoom_min, x_data <= self.zoom_max)
            y_data = line.get_ydata()
            y_data_in_range = np.trim_zeros(np.where(x_range, y_data, np.zeros_like(y_data)))
            # dynamically determine padding between extrema and bounds
            round_amount = math.ceil((max(y_data_in_range) - min(y_data_in_range)) / 10)
            if round_amount == 0:
                round_amount = 2
            if y_min == None:
                y_min = math.floor(min(y_data_in_range)/round_amount) * round_amount
            else:
                y_min = np.minimum(y_min, math.floor(min(y_data_in_range)/round_amount) * round_amount)
            if y_max == None:
                y_max = math.ceil(max(y_data_in_range)/round_amount) * round_amount
            else:
                y_max = np.maximum(y_max, math.ceil(max(y_data_in_range)/round_amount) * round_amount)
        return [y_min, y_max]