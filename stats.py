import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
from config import *

import pylab
from collections import deque

class Statistics():
    def __init__(self):
        self.fig = pylab.figure(figsize=[4,2.5])
        self.fig.set_facecolor('#161d1f')
        self.ax = self.fig.gca()
        self.loss = deque([])
        self.accuracy = deque([])

    def rotateQueue(self):
        if len(self.loss) > STAT_LIM:
            self.loss.popleft()
        if len(self.accuracy) > STAT_LIM:
            self.accuracy.popleft()

    def plotLoss(self):
        self.rotateQueue()
        self.ax.clear()
        
        self.ax.set_xlim(0, STAT_LIM)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.patch.set_alpha(0.0)

        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        self.ax.set_title('Loss [MSE]', color='white')
        
        self.ax.plot(self.loss, color = 'white')

        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        size = canvas.get_width_height()
        raw_data = renderer.tostring_rgb()
        
        return raw_data, size

    def plotAccuracy(self):
        self.rotateQueue()
        self.ax.clear()
        
        self.ax.set_xlim(0, STAT_LIM)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.patch.set_alpha(0.0)

        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        self.ax.set_title('Accuracy [%]', color='white')
        
        self.ax.plot(self.accuracy, color = 'white')

        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        size = canvas.get_width_height()
        raw_data = renderer.tostring_rgb()
        
        return raw_data, size
