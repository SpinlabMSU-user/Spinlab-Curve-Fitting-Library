# -*- coding: utf-8 -*-
"""
Spinlab Curve Fitting Template

File:          SpinlabCF_template.py
Author:        Steve Fromm
Last Modified: 2017-08-04

This template demonstrates usage of the SpinlabCF curve fitting library.  This
library makes use of the packages scipy.optimize, numpy and matplotlib.  It
handles loading of data files, preparing and performing curve fitting, and also
has the ability to plot the data and the results.  A set of standard models is
provided, and custom models can be created as needed.

To use the SpinlabCF library, place a copy of the file SpinlabCF.py in the same
as your curve fitting script.

* Template model class below is taken/modified from Dustin Frisbie's ELOG General #130
"""

# Import the library
import SpinlabCF as cf
import numpy as np

###############################
#    CONFIGURATION OPTIONS    #
###############################

# Input file options
dataFile = 'path_to_data_file.txt' # Must be a text file
delimeter = '\t' # The delimeter used to separate columns in your file
absoluteSigma = True # True if uncertainies are absolute values, False for relative
headers = False # Set this to True if you data file contains a row of column headers
# These next values are the columns that represent your x,dx,y,dy columns
# If headers = False, these are the column numbers (starting at 0)
# If headers = True, set these as the string name of the column
xDataCol = 0
dxDataCol = 1
yDataCol = 2
dyDataCol = 3

# Output files
# Plots can be saved to any standard image file
dataPlotFile = 'path_to_save_dataPlotFile.png'
fitPlotFile = 'path_to_save_fitPlotFile.png'
residualPlotFile = 'path_to_save_residualPlotFile.png'

"""Plot Configuration Options
Each plot has its own PlotOptions configuration object, each of these options
and their default values are shown below.  Options that are not directly
configured are set to default.  Examples are shown after the descriptions; add
options in the same style of option=vale to the PlotOptions object.

Available options:
            title - string, title of the plot
                    (default: 'X vs. Y')
            pt_fmt - string, pyplot format code, used for data points
                    (default: 'rx')
            line_fmt - string, pyplot format code, used for lines
                    (default: 'b-')
            xmin,xmax,ymin,ymax - float, value of the bounds of the plot
                    (default: autoscale)
            xlabel,ylabel - string, label for each axis
                    (default: 'X' and 'Y')
            capsize - int, size of the error bar caps
                    (default: 5)
            fontfamily - string, name of sans-serif font to use
                    (default: 'Palatino Linotype')
            figsize - float tuple - (x,y) size in inches of the figure
                    (default: (8,6))
            dpi - int, dpi to calcuate figure size in
                    (default: 80)
            fontsize - int, point size for fonts used
                    (defualt: 12)
            powerlimits - int tuple, lower and upper limits of exponents before
                          scientific notation is used
                    (default: (-1,4))"""

# Data Plot - shows only data points with error bars
dataPlotOptions = cf.PlotOptions(title = 'Data Plot Title',
                                 xlabel = 'X Data',
                                 ylabel = 'Y Data')

# Fit Plot - shows data points with error bars and the fit line
fitPlotOptions = cf.PlotOptions(title = 'Fit Plot Title',
                                 xlabel = 'X Values',
                                 ylabel = 'Y Values')

# Residual Plot - shows residuals with error bars and a zero reference line
residualPlotOptions = cf.PlotOptions(title = 'Residual Plot Title',
                                 xlabel = 'X Values',
                                 ylabel = 'Y Residuals')

"""Model Options
Choose an existing SpinlabCF model or write your own based on the following
template:"""
#class CustomModel(cf.Model): # Needs to be derived from the cf.Model class
#    """This class will implement your custom model, associate parameters with
#    text names, and generate text for your plots"""
#    def __init__(self):
#        """Change these variables to set up your model
#        ***ALL THREE REQUIRED***"""
#        self.numParams = 0             # The number of parameters in your model
#                                       # e.g. self.numParams = 2
#                                       
#        self.paramNames = []           # The names the parameters (same order as in Function below)
#                                       # e.g. self.paramNames = ['m','b']
#                                       
#        self.name = 'Custom Model'     # The name of your model
#                                       # e.g. self.name = 'Linear Model'
#        
#    def Function(self,x,"""PARAMETERS HERE"""):
#        """This is the function that defines your model; include your parameters
#        in place of m,b.  For example, if your model is has parameters m,b:
#            def Function(self,x,m,b)
#        ***REQUIRED***"""
#        return """EQUATION HERE""" # Return a y-value for a given x-value and parameters, e.g. return m*x+b
#    
#    def Text(self,vals):
#        """Return a text version of your model
#        Parameters:
#            vals - dict, keys are the parameter names as strings; the fit object
#                   will pass in these values"""
#        """Parameters"""
#        """Parameters go here""" = self.paramNames      # e.g. m,b = self.paramNames
#        if vals:
#            return 'Formatted equation with fit values' # e.g. return 'y = ({:1.2e})x + {:1.2e}'.format(vals[m],vals[b])
#        else:
#            return 'Text version of equation with parameter names' # e.g. 'y = mx + b'
#    
#    def InitialGuess(self,data):
#        """Generate an initial guess of the parameters for the model
#        This function is only used if you do not provide initial guesses at your
#        parameters."""
#        param1 = """Initial guess for param1"""    # e.g. m = slope of line between two points
#        param2 = """Initiall guess for param2"""   # e.g. b = y-intercept of line between two points of slope m
#        return param1,param2 # return all your parameters in the same order as listed in Function

model = cf.Fit.Linear # or for a custom model like above: model = CustomModel()
"""Currently available models are:
        Fit.Linear    <-- y(x) = m*x + b
        Fit.Quadratic <-- y(x) = a*x^2 + b*x + c
        Fit.Cubic     <-- y(x) = a*x^3 + b*x^2 + c*x + d
        Fit.Gaussian  <-- y = a*exp(-((x-b)/c)^2)
        Fit.OhmIV     <-- I(V) = V/R + b
        Fit.OhmIR     <-- I(R) = V/R + b
"""

# Fit Options
initialParams = [1,2]   # Initial guess of parameters (number of values needs to match number of parameters)
                        # If you want to use the Model's best guess:
                        #    initialParams = None
                        # For a Fit.Linear:
                        #     initialParams = [m,b]
                        # For a Fit.Quadratic:
                        #     initialParams = [a,b,c]
                        # The same follows for all models; must be in same order as Model.paramNames

"""bounds - (optional) float array of two tuples.  Values can be set in
                     two ways:
                         bounds = [low, high] to set the same bounds for each parameter
                         bounds = [(low1,low2,low3),(high1,high2,high3)] to individually set parameter bounds
                     default:
                         bounds = [-inf,inf]"""
bounds = [-np.inf,np.inf]   # Bounds to search for parameter values in

"""method - (optional) - string, algorithm used for the minimization
                     Available options:
                         'lm' - (default)Levenberf-Marquardt algorithm - does not use bounds
                         'trf' - Trust Region Reflective algorithm - requires bounds
                         'dogbox' - dogleg algorithm - requires bounds
                         see scipy.optimize.least_squares docs for more info"""
method = 'lm'

######################
#   Curve-Fitting    #
######################

# Load the data file with options configured above
cols = {'X':xDataCol,'dX':dxDataCol,'Y':yDataCol,'dY':dyDataCol}
data = cf.DataSet(dataFile,delimeter,absoluteSigma,headers,cols)

# Perform the fit
fit = cf.Fit(data,model,initialParams,bounds,method)

# Display the three plots
fit.DisplayData(dataPlotOptions,dataPlotFile)
fit.DisplayFit(fitPlotOptions,fitPlotFile)
fit.DisplayResiduals(residualPlotOptions,residualPlotFile)

# Display the results and goodness of fit
print(fit.Results())