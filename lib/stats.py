import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
from lib.config import STAT_LIM
import pylab
from collections import deque

class Statistics():
    """
    Generate neural network metrics plots, convert them as bitstring and 
    pass them to the pygame display to be rendered.

    Attributes:
        fig (pylab.figure): define the figure
        ax (pyab.figure.gca): plot axes
        loss (dequeue): collection of the loss values
        accuracy (dequeue): collection of the accuracy values

    """
    def __init__(self):
        self.fig = pylab.figure(figsize=[4,2.5])
        self.fig.set_facecolor('#161d1f')
        self.ax = self.fig.gca()
        self.loss = deque([])
        self.accuracy = deque([])

    def rotateQueue(self):
        """
        Update the loss and accuracy lists discarding older values and 
        appending the new ones.

        """
        if len(self.loss) > STAT_LIM:
            self.loss.popleft()
        if len(self.accuracy) > STAT_LIM:
            self.accuracy.popleft()

    def plotLoss(self):
        """
        Plot the loss values, convert the plot into bit strings and pass them
        to the pygame display

        """
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
        """
        Plot the accuracy values, convert the plot into bit strings and pass 
        them to the pygame display

        """
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
