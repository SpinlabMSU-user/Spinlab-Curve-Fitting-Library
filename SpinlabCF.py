# -*- coding: utf-8 -*-
"""
Spinlab Curve Fitting Library

File: SpinlabCF.py
Author: Steve Fromm
Last Modified: 2017-09-05

This library provides an easy to use interface to perform non-linear function
fitting to a provided data set.  The underlying curve-fitting algorithm is
from the scipy.optimize package.  Provided data sets should be formatted as
a 4-column, tab separated file, with the columns being [X, dX, Y, dY].
"""

import scipy.optimize as sopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import copy

# Custom exceptions
class DataSetException(Exception):
    pass

class ModelException(Exception):
    pass

class CurveFitException(Exception):
    pass

#class StandardDataSet(object):
#    """This class provides a simple container for a data set loaded from a file"""
#    def __init__(self,fileName,delim='\t',absoluteSigma=True):
#        """Load a data set from a file
#        Parameters:
#            fileName - string, path of the file to load
#            delim - (optional) string, delimiter used in the file
#            absolulteSigma - (optional) bool, True if sigma is absolute, False
#                             if sigma is a percent"""
#        # Load the file
#        self.X,self.dX,self.Y,self.dY = \
#            np.loadtxt(fileName,delimiter=delim,unpack=True)
#            
#        # If sigma is in percent, calculate absolute sigma
#        if not absoluteSigma:
#            self.dX = np.array([x*dx for x,dx in zip(self.X,self.dX)])
#            self.dY = np.array([y*dy for y,dy in zip(self.Y,self.dY)])
#            
#        self.N = len(self.X)
#            
#    def __repr__(self):
#        """Return a readadble table of the data set"""
#        lines = '{:^12s}{:^12s}{:^12s}{:^12s}\n'.format('X','dX','Y','dY')
#        for x,dx,y,dy in zip(self.X,self.dX,self.Y,self.dY):
#            lines += '{:^12.3e}{:^12.3e}{:^12.3e}{:^12.3e}\n'.format(x,dx,y,dy)
#        return lines
#    
#    def __str__(self):
#        """String form of the table for the data set"""
#        return self.__repr__()
    
class DataSet(object):
    """This class provides a simple container for a data set loaded from a file"""
    def __init__(self,fileName,delim='\t',absoluteSigma=True,headers=False,cols=None):
        """Load a data set from a file
        Parameters:
            fileName - string, path of the file to load
            delim - (optional) string, delimiter used in the file
            absolulteSigma - (optional) bool, True if sigma is absolute, False
                             if sigma is a percent
            headers - (optional) bool, True if file contians a header row
            cols - (optional) dict, Determines which columns are X,dX,Y,dY.
                   Must have keys of 'X', 'dX', 'Y', 'dY'"""
        # Store for later
        self.headers = headers
        
        # Determine number of columns
        self.numCols = len(open(fileName,'r').readline().split(delim))
        
        if self.headers:
            # Get the header names
            with open(fileName,'r') as fp:
                colNames = fp.readline().rstrip().split(delim)
                colNames = [s.strip() for s in colNames]
            
            # Associate the header names with a column number
            self.columns = dict()
            for i,colName in enumerate(colNames):
                if colName in self.columns.keys():
                    raise DataSetException('Duplicate column name '+colName)
                self.columns[colName] = i
            
            # Skip header row when loading data
            skip = 1
            
            # If columns were not specified, set X,dX,Y,dY to first four columns
            if not cols:
                cols = {'X':colNames[0],'dX':colNames[1],'Y':colNames[2],'dY':colNames[3]}
        else:
            # This lets columns be used generically
            self.columns = [i for i in range(self.numCols)]
            skip = 0
            
            # default values
            if not cols:
                cols = {'X':0,'dX':1,'Y':2,'dY':3}
            
        # Load the data file
        self.data = np.loadtxt(fileName,delimiter=delim,skiprows=skip)
        
        # Set initial X,dX,Y,dY
        self.colIDs = cols
        self.SetColumns(**cols)
            
        # If sigma is in percent, calculate absolute sigma
        if not absoluteSigma:
            self.dX = np.array([x*dx for x,dx in zip(self.X,self.dX)])
            self.dY = np.array([y*dy for y,dy in zip(self.Y,self.dY)])
            
        self.N = len(self.X)
        
    def SetColumns(self,**kwargs):
        """Set which columns represent X, dX, Y, dY
        Takes key/value pairs in the form of:
            X = column#/name,
            dX = column#/name,
            Y = column#/name,
            dY = column#/name"""
        if 'X' in kwargs.keys():
            self.X = copy.copy(self.data[:,self.columns[kwargs['X']]])
            self.colIDs['X'] = kwargs['X']
        
        if 'dX' in kwargs.keys():
            self.dX = copy.copy(self.data[:,self.columns[kwargs['dX']]])
            self.colIDs['dX'] = kwargs['dX']
        
        if 'Y' in kwargs.keys():
            self.Y = copy.copy(self.data[:,self.columns[kwargs['Y']]])
            self.colIDs['Y'] = kwargs['Y']
        
        if 'dY' in kwargs.keys():
            self.dY = copy.copy(self.data[:,self.columns[kwargs['dY']]])
            self.colIDs['dY'] = kwargs['dY']
            
    def __getitem__(self,key):
        """Makes DataSet object callable by index of column"""
        return self.data[:,self.columns[key]]
            
    def __repr__(self):
        """Return a readadble table of the data set"""
        line = ''
        
        if self.headers:
            padding = max(len(s) for s in self.columns.keys()) + 4
        else:
            padding = 12
            
        for col in self.columns:
            val = ''
            if self.headers:
                val = col
            else:
                val = ' '
            line += '{:^' + str(padding) +'s}'
            if col in self.colIDs.values():
                k = list(self.colIDs.keys())
                v = list(self.colIDs.values())
                val += '(' + k[v.index(col)] + ')'
                
            line = line.format(val)
                
        lines = line + '\n'
                
            
        for row in self.data:
            lines += str(str('{:^'+ str(padding)+'.3e}')*self.numCols+'\n').format(*row)
        return lines
    
    def __str__(self):
        """String form of the table for the data set"""
        return self.__repr__()
    
def LinSolve(X,Y,deg):
    """Linearizes and solves n-degree polynomials based off of evenly spaced
    data points in a given data set.  Used to help generate intial guesses
    at model parameters.  Data must have deg+1 points.
    Parameters:
        X - iterable, X values
        Y - iterable, Y values
        deg - int, degree of polynomial
    Returns:
        numpy array of coefficients in order of highest to lowest degree terms"""
    n = deg + 1
    N = len(X)
    if N < n:
        raise DataSetException('Data set has insufficient points for this model.')
        
    # Create vectors of points to use
    x = np.zeros(n)   
    y = np.matrix(np.zeros((n,1)))
    
    # Create array of indices of X/Y to use
    idx = np.zeros(n,dtype=int)
    # Use first and last points
    idx[0] = 0
    idx[n-1] = N - 1

    step = N // n
    if n > 2:
        # Calculate inner indices
        for i in range(1,n-1):
            idx[i] = i*step
    
    # Grab the x/y values at a given index
    for i in range(n):
        ind = idx[i]
        x[i] = X[ind]
        y[i,0] = Y[ind]
    
    # Build the matrix of x values (x**0, x**1,...x**deg) rows    
    mat = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            mat[i,j] = x[i]**(j)
    
    # Solve system        
    guesses = (np.linalg.inv(mat)*y).A1
    
    # return parameters in reverse order (highest degree coefficient first)
    return guesses[::-1]

class Model(object):
    """This class is the base model function for the fitting algorithm"""
    def __init__(self):
        """Override these member variables in sub-classes
        ***ALL THREE REQUIRED***"""
        self.numParams = 0
        self.paramNames = []
        self.name = 'Undefined Model'
        
    def __call__(self,x,*args):
        """Allows object to be called as function"""
        return self.Function(x,*args)
        
    def Function(self,x,*args):
        """Must override this function in any derived Model class
        ***REQUIRED***"""
        raise ModelException('All defined models must override __call__()')
    
    def Text(self,vals):
        """Override this in subclass if you want to format the model as
        a string
        Parameters:
            vals - dict, keys are the parameter names as strings; the fit object
                   will pass in these values"""
        return 'Model is not available in text format'
    
    def InitialGuess(self,data):
        """Generate an initial guess of the parameters for the model"""
        return None
    
class UserModel(Model):
    """Custom model based on a user provided function"""
    def __init__(self,name,func,params,text,initGuess=None,formatText=None):
        """Generate a custom model based on a provided funtion
        Parameters:
            name - string, name of the model
            func - function to fit to of form f(x,p0,..pn)
            params - string list, name of the parameters in same order as the
                     provided function, e.g. ['m','b'] for f(x,m,b)
            text - string, text form of the equation, e.g. 'y=m*x+b'
            initGuess - (optional) function to provide initial guess for fitting
                        algorithm, of form guess(data) where data is a DataSet
            formatText - (optional) string, format string with identifiers as
                         param names, e.g. 'y={m:1.2f}x+{b:1.2f}"""
        super().__init__()
        self.name = name
        self.numParams = len(params)
        self.paramNames = params
        self.func = func
        self.initGuess = initGuess
        self.text = text
        self.formatText = formatText
        
    def Function(self,x,*args):
        """Returns user provided function evaluated at a given x"""
        return self.func(x,*args)
    
    def __call__(self,x,*args):
        """Allows object to be called as function"""
        return self.func(x,*args)
    
    def InitialGuess(self,data):
        """Generates an initial guess to use in the fitting algorithm (if provided)"""
        if not self.initGuess:
            # Return 1's if not provided
            return (1 for i in range(self.numParams))
        
        return self.initGuess(data)
    
    def Text(self,vals=None):
        """Return text form of model"""
        if not self.formatText or not vals:
            return self.text
        else:
            return self.formatText.format(**{n:vals[n] for n in self.paramNames})
        
class LinearModel(Model):
    """Basic linear mode"""
    def __init__(self):
        """Create a linear model"""
        super().__init__()
        self.name = 'linear'
        self.numParams = 2
        self.paramNames = ['m','b']
        
    def Function(self,x,m,b):
        """Calculate a y value at the given x"""
        return m*x+b
    
    def Text(self,vals=None):
        """This provides the text version of the equation"""
        m,b = self.paramNames
        if vals:
            return 'y = ({:1.2e})x + {:1.2e}'.format(vals[m],vals[b])
        else:
            return 'y = mx + b'
        
    def InitialGuess(self,data):
        """Use evenly spaced points to solve linear system"""
        return LinSolve(data.X,data.Y,1).tolist()
    
class OhmIVModel(Model):
    """Ohm's Law as I(V) = V/R + b"""
    def __init__(self):
        """Create the model"""
        super().__init__()
        self.name = 'ohmIV'
        self.numParams = 2
        self.paramNames = ['R','b']
        
    def Function(self,x,R,b):
        """Calculate a y value at the given x"""
        return x/R+b
    
    def Text(self,vals=None):
        """This provides the text version of the equation"""
        R,b = self.paramNames
        if vals:
            return 'I = V/{:1.2e} + {:1.2e}'.format(vals[R],vals[b])
        else:
            return 'I = V/R + b'
        
    def InitialGuess(self,data):
        """Use evenly spaced points to solve linear system"""
        guesses = LinSolve(data.X,data.Y,1)
        R = 1/guesses[0]
        b = guesses[1]
        return (R,b)
    
class OhmIRModel(Model):
    """Ohm's Law as I(R) = V/R + b"""
    def __init__(self):
        """Create the model"""
        super().__init__()
        self.name = 'ohmIR'
        self.numParams = 2
        self.paramNames = ['V','b']
        
    def Function(self,x,V,b):
        """Calculate a y value at the given x"""
        return V/x+b
    
    def Text(self,vals=None):
        """This provides the text version of the equation"""
        V,b = self.paramNames
        if vals:
            return 'I = {:1.2e}/R + {:1.2e}'.format(vals[V],vals[b])
        else:
            return 'I = V/R + b'
        
    def InitialGuess(self,data):
        """Use evenly spaced points to solve linear system"""
        x = 1/data.X
        guesses = LinSolve(x,data.Y,1)
        V = guesses[0]
        b = guesses[1]
        return (V,b)
    
class QuadraticModel(Model):
    """Basic Quadratic Fit Model"""
    def __init__(self):
        super().__init__()
        self.name = 'quadratic'
        self.numParams = 3
        self.paramNames = ['a','b','c']
        
    def Function(self,x,a,b,c):
        """Calculate y for a given x"""
        return a*x**2 + b*x + c
    
    def Text(self,vals=None):
        """This provides the text version of the equation"""
        a,b,c = self.paramNames
        if vals:
            return 'y = ({:1.2e})x^2 + ({:1.2e})x + {:1.2e}'.format(vals[a],vals[b],vals[c])
        else:
            return 'y = ax^2 + bx + c'
        
    def InitialGuess(self,data):
        """Use evenly spaced points to solve linear system"""
        return LinSolve(data.X,data.Y,2).tolist()
        
class CubicModel(Model):
    """Basic Cubic Fit Model"""
    def __init__(self):
        """Create the model"""
        super().__init__()
        self.name = 'cubic'
        self.numParams = 4
        self.paramNames = ['a','b','c','d']
        
    def Function(self,x,a,b,c,d):
        """Calculate y for a given x"""
        return a*x**3 + b*x**2 + c*x + d
    
    def Text(self,vals=None):
        """Text form of the model"""
        a,b,c,d = self.paramNames
        if vals:
            return '({:1.2e})x^3 + ({:1.2e})x^2 + ({:1.2e})x + {:1.2e}'.format(vals[a],vals[b],vals[c],vals[d])
        else:
            return 'ax^3 + bx^2 + cx + d'
        
    def InitialGuess(self,data):
        """Use evenly spaced points to solve linear system"""
        return LinSolve(data.X,data.Y,3).tolist()
        
class GaussianModel(Model):
    """Basic Gaussian Fit Model"""
    def __init__(self):
        """Create the model"""
        super().__init__()
        self.name = 'gaussian'
        self.numParams = 3
        self.paramNames = ['a','b','c']
        
    def Function(self,x,a,b,c):
        """Calculate y for a given x"""
        return a*np.exp(-(((x-b)/c)**2))
    
    def Text(self,vals=None):
        """Text version of the equation"""
        if vals:
            a,b,c = self.paramNames
            a,b,c = '{:1.2e},{:1.2e},{:1.2e}'.format(vals[a],vals[b],vals[c]).split(',')
            return '$y = ('+a+')*e^{-\left(\\frac{x - ('+b+')}{'+c+'}\\right)^2}$'
        else:
            return 'y = a*exp(-((x-b)/c)^2)'
    
    def InitialGuess(self,data):
        """a from heigh, b from x coord of height, solve for c"""
        a = max(data.Y)
        idx = np.where(data.Y == a)[0][0]
        pt = int(np.abs((idx - np.where(data.Y == min(data.Y))[0][0])/2))
        b = data.X[idx]
        c = 1/(np.sqrt((-1*np.log(data.Y[pt]/a)))/(data.X[pt]-b))
        
        if b > 0:
            c *= -1
        
        return a,b,c
    
class ExpDecayModel(Model):
    """Basic Exponential Decay Fit Model"""
    def __init__(self):
        """Create the model"""
        super().__init__()
        self.name = 'exp decay'
        self.numParams = 2
        self.paramNames = ['a','b']
        
    def Function(self,x,a,b):
        return a*np.exp(-b*x)
    
    def Text(self,vals=None):
        return 'a*exp(-b*x)'
    
    def InitialGuess(self,data):
        return (1,1)
    
class PlotOptions(object):
    """Holds information on how to display plots"""
    def __init__(self,**kwargs):
        """Available options:
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
        keys = kwargs.keys()
        self.title = 'X v.s Y' if 'title' not in keys else kwargs['title']
        self.pt_fmt = 'ro' if 'pt_fmt' not in keys else kwargs['pt_fmt']
        self.line_fmt = 'b-' if 'line_fmt' not in keys else kwargs['line_fmt']
        self.xmax = None if 'xmax' not in keys else kwargs['xmax']
        self.xmin = None if 'xmin' not in keys else kwargs['xmin']
        self.ymax = None if 'ymax' not in keys else kwargs['ymax']
        self.ymin = None if 'ymin' not in keys else kwargs['ymin']
        self.xlabel = 'X' if 'xlabel' not in keys else kwargs['xlabel']
        self.ylabel = 'Y' if 'ylabel' not in keys else kwargs['ylabel']
        self.capsize = 5 if 'capsize' not in keys else kwargs['capsize']
        self.fontfamily = 'Palatino Linotype' if 'fontfamily' not in keys else kwargs['fontfamily']
        self.figsize = (8,6) if 'figsize' not in keys else kwargs['figsize']
        self.dpi = 80 if 'dpi' not in keys else kwargs['dpi']
        self.fontsize = 12 if 'fontsize' not in keys else kwargs['fontsize']
        self.powerlimits = (-1,4) if 'powerlimits' not in keys else kwargs['powerlimits']
    
class Fit(object):
    """This class produces and displays the curve fit to the provided data set"""
    
    # Provided Models, assignable as Fit.ModelName
    Linear = LinearModel()
    Quadratic = QuadraticModel()
    Cubic = CubicModel()
    OhmIV = OhmIVModel()
    OhmIR = OhmIRModel()
    Gaussian = GaussianModel()
    ExpDecay = ExpDecayModel()
    
    def __init__(self,data:DataSet,model:Model,initialParams=[],bounds=[],method='lm',weighted=True,eps=1e-6):
        """Create the fit for this data set
        Parameters:
            data - DataSet, previously loaded data set
            model - A function defining the fit and parameters
            initialParams - (optional) float array, initial guess at parameters
            bounds - (optional) float array of two tuples.  Values can be set in
                     two ways:
                         bounds = [low, high] to set the same bounds for each parameter
                         bounds = [(low1,low2,low3),(high1,high2,high3)] to individually set parameter bounds
                     default:
                         bounds = [-inf,inf]
            method - (optional) - string, algorithm used for the minimization
                     Available options:
                         'lm' - (default)Levenberf-Marquardt algorithm - does not use bounds
                         'trf' - Trust Region Reflective algorithm - requires bounds
                         'dogbox' - dogleg algorithm - requires bounds
                         see scipy.optimize.least_squares docs for more info
            eps - (optional) float, precent size of an 'infinitesimal' step"""
            
        self.data = data
        self.eps = eps
        self.model = model
        
        self.numFigs = 1
        
        self.bounds = bounds
        self.method = method
        
        args = dict()
        args['absolute_sigma'] = True
        args['method'] = method
        
        if bounds:
            args['bounds'] = bounds
            
        if (method == 'trf' or method == 'dogbox') and not bounds:
            raise CurveFitException('Bounds are reuqired for trf and dogbox algorithms')
        
        # Check if initial params are given, otherwise run without x errors
        # to obtain initial guesses
        if not initialParams:
            #initialParams,a = sopt.curve_fit(self.model.Function,self.data.X,self.data.Y)
            initialParams = model.InitialGuess(self.data)
            if not initialParams:
                raise ModelException('You must implement the InitialGuess method of your model')
            
        # Calculate intiial sigma total and weights
        dXprop = self.PropDX(initialParams)
        #dXprop = np.array([(self.model.Function(x+(self.eps*x),*initialParams)-self.model.Function(x-(self.eps*x),*initialParams))/(2*(self.eps*x))*dx \
        #                 for x,dx in zip(self.data.X,self.data.dX)])
        
        # Calculate and store total uncertainty/weights
        sigma2 = dXprop**2+data.dY**2
        self.sigma = np.sqrt(sigma2)
        self.weight = 1/sigma2

        # This is the main curve fitting algorithm
        # popt returns an array of the fit parameters in the order of the
        # provided argument list
        # pcov is the covariance matrix of the fit
        # We converted our uncertainties to absolute values so absolute_sigma
        # is set to True
        if weighted:
            args['p0'] = initialParams
            args['sigma'] = self.sigma

        popt,pcov = sopt.curve_fit(self.model.Function,self.data.X,self.data.Y,maxfev=20000,**args)
        
        # Calculate and store the uncertainties in the parameters
        self.errors = np.sqrt(np.diag(pcov))
        self.params = popt
        
        # Associate the parameters and uncertainties with the provided names
        self.parameters = {n:p for n,p in zip(model.paramNames,popt)}
        self.uncertainties = {'d'+n:u for n,u in zip(model.paramNames,self.errors)}
        
        # Calculate residuals and degrees of freedom
        self.expVals = np.array([model(x,*self.params) for x in data.X])
        self.residuals = np.array([y-yi for y,yi in zip(data.Y,self.expVals)])
        self.dof = data.N - model.numParams
        
        # Calculate chi-squared: sum[(residual/uncertainty)^2]
        self.chi2 = sum([(r**2)*w for r,w in zip(self.residuals,self.weight)])
        
    def ParamText(self,p):
        """Formats the result of the provided parameter p:
            p = value +/- uncertainty"""
        return p + ' = ' + str(self.parameters[p]) + ' +/- ' \
                 + str(self.uncertainties['d'+p])
        
    def PropDX(self,params):
        """Propogate x-uncertainties to y-uncertainties for current paramters.
        Takes numerical derivative at each point and calculates the x uncertainty
        contribution to the total uncertainty as:
            sigma_y(x,sigma_x) = abs(dy/dx)*sigma_x"""
        return np.array([(self.model.Function(x+(self.eps*x),*params)-self.model.Function(x-(self.eps*x),*params))/(2*(self.eps*x))*dx \
                         for x,dx in zip(self.data.X,self.data.dX)])
    
    def SetPlotProps(self,po):
        """Configure plot properties"""
        rcParams.update({'figure.autolayout':True})
        rcParams.update({'font.sans-serif':po.fontfamily})
        rcParams.update({'font.family':'sans-serif'})
        rcParams.update({'figure.figsize':(8,6)})
        rcParams.update({'figure.dpi':po.dpi})
        rcParams.update({'font.size':po.fontsize})
        
    def ForceSciNotation(self,po):
        plt.gca().yaxis.get_major_formatter().set_powerlimits(po.powerlimits)
        plt.gca().xaxis.get_major_formatter().set_powerlimits(po.powerlimits)
        
    def DisplayData(self,po=PlotOptions(),saveFile=None):
        """Displays the plot of the data points
        Parameters:
            po - (optional) PlotOptions, configure how the plot is displayed
            saveFile - (optional) string, name of the image file to save the plot as"""
        # Make plots look nice
        self.SetPlotProps(po)
        
        # Create a new unique figure for this plot
        plt.figure(self.numFigs)
        self.numFigs += 1
        
        # Plot the points with errorbars; returns 3 lines: the points, y-error
        # bars and x-error bars; we only need the points line, a and b are dummy
        # variables so the interpreter doesn't squawk at us
        line1,a,b = plt.errorbar(self.data.X,self.data.Y,xerr=self.data.dX,yerr=self.data.dY,
                                 fmt=po.pt_fmt,fillstyle='none',capsize=po.capsize,)
        line1.set_label('Data')
        
        # Set the plot bounds if provided
        if po.xmin:
            plt.xlim(xmin=po.xmin)
        if po.xmax:
            plt.xlim(xmax=po.xmax)
        if po.ymin:
            plt.ylim(ymin=po.ymin)
        if po.ymin:
            plt.ylim(ymax=po.ymax)
            
        # Insert the text labels on the plot
        plt.title(po.title)
        plt.xlabel(po.xlabel)
        plt.ylabel(po.ylabel)
        plt.legend()
        
        self.ForceSciNotation(po)
        
        # Save the figure to a file if requested
        if saveFile:
            plt.savefig(saveFile)
        
        # Flush the image buffer to the screen
        plt.show()
    
    def DisplayFit(self,po=PlotOptions(),saveFile=None,showData=True):
        """Displays the plot of the data points and the fit line
        Parameters:
            po - (optional) PlotOptions, configure how the plot is displayed
            saveFile - (optional) string, name of the image file to save the plot as
            showData - (optional) bool, determine if data points should overlay fit line"""
        # Make plots look nice
        self.SetPlotProps(po)
        
        # Create a new unique figure for this plot
        plt.figure(self.numFigs)
        self.numFigs += 1
        
        # Plot the points with errorbars; returns 3 lines: the points, y-error
        # bars and x-error bars; we only need the points line, a and b are dummy
        # variables so the interpreter doesn't squawk at us
        if showData:
            line1,a,b = plt.errorbar(self.data.X,self.data.Y,xerr=self.data.dX,yerr=self.data.dY,
                                     fmt=po.pt_fmt,fillstyle='none',capsize=po.capsize,)
            line1.set_label('Data')
        
        # Create a smooth line of the fit and plot it;
        xf = np.linspace(min(self.data.X),max(self.data.X),1000)
        yf = self.model(xf,*self.params)
        line2, = plt.plot(xf,yf,po.line_fmt)
        line2.set_label(self.model.Text(self.parameters))
        
        # Set the plot bounds if provided
        if po.xmin:
            plt.xlim(xmin=po.xmin)
        if po.xmax:
            plt.xlim(xmax=po.xmax)
        if po.ymin:
            plt.ylim(ymin=po.ymin)
        if po.ymin:
            plt.ylim(ymax=po.ymax)
            
        # Insert the text labels on the plot
        plt.title(po.title)
        plt.xlabel(po.xlabel)
        plt.ylabel(po.ylabel)
        plt.legend()
        
        self.ForceSciNotation(po)
        
        # Save the figure to a file if requested
        if saveFile:
            plt.savefig(saveFile)
        
        # Flush the image buffer to the screen
        plt.show()
            
    def DisplayResiduals(self,po=PlotOptions(),saveFile=None):
        """Displays the plot of the residuals and a zero reference line
        Parameters:
            po - (optional) PlotOptions, configure how the plot is displayed
            saveFile - (optional) string, name of the image file to save the plot as"""
        # Make plots look nice
        self.SetPlotProps(po)
        
        # Create a new unique figure for this plot
        plt.figure(self.numFigs)
        self.numFigs += 1
        
        # Plot the points with errorbars; returns 3 lines: the points, y-error
        # bars and x-error bars; we only need the points line, a and b are dummy
        # variables so the interpreter doesn't squawk at us
        line,a,b = plt.errorbar(self.data.X,self.residuals,yerr=self.data.dY,
                     fmt=po.pt_fmt,fillstyle='none',capsize=po.capsize)
        
        # Create and plot the zero-reference line
        x1 = min(self.data.X)
        x2 = max(self.data.X)
        padding = (x2 - x1)*0.05
        xf = np.array([x1-padding,x2+padding])
        yf = np.array([0,0])
        plt.plot(xf,yf,po.line_fmt)
        
        # Set the plot bounds if provided
        if po.xmin:
            plt.xlim(xmin=po.xmin)
        if po.xmax:
            plt.xlim(xmax=po.xmax)
        if po.ymin:
            plt.ylim(ymin=po.ymin)
        if po.ymin:
            plt.ylim(ymax=po.ymax)
            
        # Insert the text labels on the plot
        plt.title(po.title)
        plt.xlabel(po.xlabel)
        plt.ylabel(po.ylabel)
        line.set_label('$y - y_{fit}$')
        plt.legend()
        
        self.ForceSciNotation(po)
        
        # Save the figure to a file if requested
        if saveFile:
            plt.savefig(saveFile)
        
        # Flush the image buffer to the screen
        plt.show()
            
    def Results(self):
        """Generates a formatted string containing the results of the fit"""
        out = ''
        out += 'Model: {}\n'.format(self.model.Text())
        for k,v in self.parameters.items():
            out += '{}: {:1.6e} +/- {:1.6e}\n'.format(k,v,self.uncertainties['d'+k])
            
        out += 'SSR: {:1.6e}\n'.format(sum(self.residuals**2))
        out += 'Chi2: {:1.6e}\n'.format(self.chi2)
            
        return out
    
def SetPlotProps(po):
    """Configure plot properties"""
    rcParams.update({'figure.autolayout':True})
    rcParams.update({'font.sans-serif':po.fontfamily})
    rcParams.update({'font.family':'sans-serif'})
    rcParams.update({'figure.figsize':(8,6)})
    rcParams.update({'figure.dpi':po.dpi})
    rcParams.update({'font.size':po.fontsize})
    plt.gca().tick_params(direction='out')
    
def ForceSciNotation(po):
    plt.gca().yaxis.get_major_formatter().set_powerlimits(po.powerlimits)
    plt.gca().xaxis.get_major_formatter().set_powerlimits(po.powerlimits)
    
def DisplayData(data,po=PlotOptions(),saveFile=None):
    """Displays the plot of the data points
    Parameters:
        po - (optional) PlotOptions, configure how the plot is displayed
        saveFile - (optional) string, name of the image file to save the plot as"""
    # Make plots look nice
    SetPlotProps(po)
    
    # Create a new unique figure for this plot
    plt.figure()
    
    # Plot the points with errorbars; returns 3 lines: the points, y-error
    # bars and x-error bars; we only need the points line, a and b are dummy
    # variables so the interpreter doesn't squawk at us
    line1,a,b = plt.errorbar(data.X,data.Y,xerr=data.dX,yerr=data.dY,
                             fmt=po.pt_fmt,fillstyle='none',capsize=po.capsize,)
    line1.set_label('Data')
    
    # Set the plot bounds if provided
    if po.xmin:
        plt.xlim(xmin=po.xmin)
    if po.xmax:
        plt.xlim(xmax=po.xmax)
    if po.ymin:
        plt.ylim(ymin=po.ymin)
    if po.ymin:
        plt.ylim(ymax=po.ymax)
        
    # Insert the text labels on the plot
    plt.title(po.title)
    plt.xlabel(po.xlabel)
    plt.ylabel(po.ylabel)
    plt.legend()
    
    ForceSciNotation(po)
    
    # Save the figure to a file if requested
    if saveFile:
        plt.savefig(saveFile)
    
    # Flush the image buffer to the screen
    plt.show()
