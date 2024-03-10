import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from ipywidgets import interact
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, binned_statistic
from matplotlib.lines import Line2D

sns.set_context('poster', font_scale=1.1)

!wget https://raw.githubusercontent.com/SciX-UNSW/D2D/main/ExoMol_Data.csv
!wget https://raw.githubusercontent.com/SciX-UNSW/D2D/main/transit-spectrum-W39b-G395H-10pix_weighted-average.nc
ExoMol_data = pd.read_csv("ExoMol_Data.csv")
data_check = xr.open_dataset('transit-spectrum-W39b-G395H-10pix_weighted-average.nc')

# Fucntion to add width to spectral lines from comp chem data
def spectrawidth(freq, intensity, peakwidth = 20, scalingfactor = 1):
    minfreq = min(freq)- min(freq)*0.1
    maxfreq = max(freq) + max(freq)*0.1
    npts = int((maxfreq-minfreq)/1.0) #one datapoint per wavnumber
    x = np.linspace(minfreq,maxfreq,npts)
    y = np.zeros(npts)
    for i in range(len(freq)):
        #This add each peak sequentially based on the intensity of the peak and the
        y = y + np.exp(-(2.77/(2*peakwidth)**2)*(x-freq[i]*scalingfactor)**2)*intensity[i]
    return x, y

# Function to normalise vibrational intensities
def norminten(tbnorm):
    norm = []
    upp = max(tbnorm)
    for entry in tbnorm:
        norm.append(entry/upp)
    return norm

# Function to change from cm-1 to um
def tick_function(X):
    V = 1.0000000e4/(X)
    return ["%.2f" % z for z in V]

# Function to plot the JWST WASP-39b spectrum
def JWST_frequecies_intensities():
    # Reading weighted average spectral data

    file_strings = [drivedir+'transit-spectrum-W39b-G395H-10pix_weighted-average.nc']
    spec_names = ['average']

    data_check = xr.open_dataset(drivedir+'transit-spectrum-W39b-G395H-10pix_weighted-average.nc')

    nbins = len(data_check['central_wavelength'].values[5:])
    nspec = len(file_strings)
    gauss_x = np.linspace(-0.001, 0.001, 1000)

    # Creating arrays of 1x339 dimensions
    # -----------------------------------

    # Wavelengths (+ error)
    w_array = np.empty([nspec,nbins])
    we_array = np.empty([nspec,nbins])

    # Transition depth (+ error)
    d_array = np.empty([nspec,nbins])
    de_array = np.empty([nspec,nbins])

    dr_array = np.empty([nspec,nbins])
    dr_mean = np.empty([nspec])
    dr_std = np.empty([nspec])

    pdf_norm = np.empty([nspec,len(gauss_x)])

    align_d_array = np.empty([nspec,nbins])

    for i in range(0, nspec):
        data = xr.open_dataset(file_strings[i])
        data_keys = list(data.keys())

        # Collecting data (wavelength, wavelength error, and transit depth) from weighted average spectrum file
        w = data['central_wavelength'].values[5:]
        we = data['bin_half_width'].values[5:]
        d = data['transit_depth'].values[5:]

        # Collecting errors from transit depth
        if (data_keys[1]=='transit_depth_error_pos'):
            de = data['transit_depth_error_pos'].values[5:]
        else:
            de = data['transit_depth_error'].values[5:]

        # Moving data from the weighted average spectrum file to their corresponding arrays created before
        data_sort = np.argsort(w)
        w_array[i,:] = w[data_sort]
        d_array[i,:] = d[data_sort]
        de_array[i,:] = de[data_sort]
        we_array[i,:] = we[data_sort]

    weights_de = 1./de_array**2
    average_d, sum_of_weights = np.average(d_array, weights=weights_de, axis=0, returned=True)
    median_d = np.median(d_array, axis=0)

    weighted_average_de = np.sqrt(1./ np.sum(weights_de, axis=0))
    average_de = np.mean(de_array, axis=0)
    median_de = np.median(de_array, axis=0)

    std_array = np.std(d_array, axis=0)

    for i in range(0, nspec):

        dr_array[i,:] = d_array[i,:] - average_d

        dr_mean[i] = np.mean(dr_array[i,:])
        dr_std[i] = np.std(dr_array[i,:])

        pdf_norm[i,:] = norm.pdf(gauss_x, loc=dr_mean[i], scale=dr_std[i])


    # aligned average
    for i in range(0, nspec):
        align_d_array[i,:] = d_array[i,:] - dr_mean[i]

    align_average_d = np.average(align_d_array, weights=weights_de, axis=0)

    data_wavelength = w_array[0,:]
    data_wavelength_err = we_array[0,:]

    JWSTData = pd.DataFrame()
    JWSTData['Wavelength'] = data_wavelength
    JWSTData['Wavelength_Err'] = data_wavelength_err

    JWSTData['Wavenumber'] = tick_function(data_wavelength)
    JWSTData['Wavenumber'] = pd.to_numeric(JWSTData['Wavenumber'])

    JWSTData['TransDepth'] = average_d
    JWSTData['TransDepth_Err'] = average_de

    scale = 1.e2

    JWSTData['TransDepth'] = JWSTData['TransDepth']*scale
    JWSTData['TransDepth_Err'] = JWSTData['TransDepth_Err']*scale

    freq = np.array(JWSTData['Wavenumber'])
    intens = np.array(JWSTData['TransDepth'])#/np.nanmax(JWSTData[['TransDepth']].values)
    err = np.array(JWSTData['TransDepth_Err'])
    return freq, intens, err

# Fucntion to convert from cm-1 to um
def wavenumbers2microns(x):
    return 10000/x

# Function to convert from um to cm-1
def microns2wavenumbers(x):
    return 10000/x

# Fucntion for plotting compchem vibrational spectra from different molecules (interactive)
def MoleculesSpectra(Molecule):

    # Dictionary to display the chemical formulas with the right formatting.
    mol_names = {'CO':'CO : Carbon Monoxide',
                 'PH3':'PH$_3$ : Phosphine',
                 'H2O':'H$_2$O : Water',
                 'CO2':'CO$_2$ : Carbon Dioxide',
                 'NH3':'NH$_3$ : Ammonia',
                 'CH4':'CH$_4$ : Methane',
                 'SO2':'SO$_2$ : Sulfur Dioxide'}

    # Figure formatting.
    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(111)

    # Storing the relevant molecular data and processing for plotting.
    mol_data = QC_vibfreq.query("500 <= Frequency <= 4000 and Formula == @Molecule & Kind == 'FundScaled'").reset_index()
    wfreq, wintens = spectrawidth(mol_data['Frequency'], mol_data['Intensity'],peakwidth = 20)
    ax.plot(wfreq,wintens, linestyle = '-', label = mol_names[Molecule], color = 'rebeccapurple')

    # Plot legend.
    legend = plt.legend(frameon = False, loc = 'best')

    # Plot axes formating.
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb. units)")
    plt.xlim(500,4000)
    plt.ylim(0,max(wintens)+30)

    plt.show()


# Fucntion for plotting compchem spectra from different molecules simultaneously (interactive)
def MoleculesSpectra2(Molecule1, Molecule2):

    # Dictionary to display the chemical formulas with the right formatting.
    emol_names = {'CO':'CO : Carbon monoxide',
                 'PH3':'PH$_3$ : Phosphine',
                 'H2O':'H$_2$O : Water',
                 'CO2':'CO$_2$ : Carbon Dioxide',
                 'NH3':'NH$_3$ : Ammonia',
                 'CH4':'CH$_4$ : Methane',
                 'SO2':'SO$_2$ : Sulfur Dioxide'}

    # Figure formatting.
    fig, ax = plt.subplots(2,1, figsize=(13,12))

    # Appending molecules to a list.
    molecules = []
    molecules.append(Molecule1)
    molecules.append(Molecule2)

    # ----

    # Plot 1: showcasing spectral signals from different molecules.
    for mol in molecules:

        # Distinguishing between the spectra for different molecules.
        mol_data0 = QC_vibfreq.query("500 <= Frequency <= 4000 and Formula == @mol & Kind == 'FundScaled'").reset_index()
        wfreq1, wintens1 = spectrawidth(mol_data0['Frequency'], mol_data0['Intensity'], peakwidth = 20)
        ax[0].plot(wfreq1,wintens1, linestyle = '-', label = emol_names[mol])

    # Plot 1 legend.
    legend = ax[0].legend(frameon = False, loc = 'best')

    # ----

    # Storing spectral data for all molecules without distinguishing between them.
    mol_data = QC_vibfreq.query("500 <= Frequency <= 4000 and Formula == @Molecule1 & Kind == 'FundScaled'").reset_index().drop('index',1)
    mol_data2 = QC_vibfreq.query("500 <= Frequency <= 4000 and Formula == @Molecule2 & Kind == 'FundScaled'").reset_index().drop('index',1)
    mol_data = mol_data.append(mol_data2, ignore_index = True)

    # Plot 2: plotting combined spectral signals.
    wfreq, wintens = spectrawidth(mol_data['Frequency'], mol_data['Intensity'], peakwidth = 20)
    ax[1].plot(wfreq,wintens, linestyle = '-', label = 'Total Spectrum',color = 'rebeccapurple')

    legend1 = ax[1].legend(frameon = False, loc = 'best')

    ax[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    ax[1].set_ylabel("Intensity (arb. units)")
    ax[1].set_xlim(500,4000)
    ax[1].set_ylim(0,max(wintens)+30)

    # Plot 1 axes formatting.
    ax[0].set_ylabel("Intensity (arb. units)")
    ax[0].set_xlim(500,4000)
    ax[0].set_ylim(0,max(wintens)+30)

    plt.show()


# Function for comparing compchem spectra with ExoMol line lists (interactive)
def SpectraComparison(Molecule):

    # Dictionary to display the chemical formulas with the right formatting.
    mol_names = {'CO':'CO : Carbon Monoxide',
                 'PH3':'PH$_3$ : Phosphine',
                 'H2O':'H$_2$O : Water',
                 'CO2':'CO$_2$ : Carbon Dioxide',
                 'NH3':'NH$_3$ : Ammonia',
                 'CH4':'CH$_4$ : Methane',
                 'SO2':'SO$_2$ : Sulfur Dioxide'}

    # Figure formatting.
    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(111)

    # Storing the relevant molecular data and processing for plotting.
    mol_data = ExoMol_data.query("Molecule == @Molecule").reset_index()
    mol_data['Norm_Ints'] = np.array(mol_data['Intensity'])/np.nanmax(mol_data['Intensity'])
    ax.stem(mol_data['Frequency'],mol_data['Norm_Ints'], markerfmt=' ', basefmt=' ',
            label = 'High-resolution data', linefmt = 'goldenrod')

    # Plot legend.
    #legend = plt.legend(frameon = False, loc = 'best')

    # Plot axes formating.
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (arb. units)")
    plt.xlim(500,4000)


    ax2 = ax.twinx()

    # Storing the relevant molecular data and processing for plotting.
    mol_data_qc = QC_vibfreq.query("500 <= Frequency <= 4000 and Formula == @Molecule & Kind == 'FundScaled'").reset_index()
    wfreq, wintens = spectrawidth(mol_data_qc['Frequency'], mol_data_qc['Intensity'], peakwidth = 20)
    ax2.plot(wfreq,wintens, linestyle = '-', label = 'Mid-resolution data', color = 'rebeccapurple')

        # ----------
    # Plot legend settings
    custom_lines = [Line2D([0], [0], color='goldenrod', lw=4),
                    Line2D([0], [0], color='rebeccapurple', lw=4)]

    # Put a legend to the right of the current axis
    ax.legend(custom_lines,['High-resolution data','Mid-resolution data'],
                            frameon = False, loc = 'best', title = mol_names[Molecule])

    plt.show()


# Function for finding molecules present in the JWST WASP-39b spectrum
import warnings
warnings.filterwarnings("ignore")
columns = [None,'H2O','NH3','CH4','SO2','CO2','CO']

def updatePlot(Molecule1, Molecule2, Molecule3, Molecule4):

    # Setting figure's aesthetics and labels
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    fig.subplots_adjust(wspace=0.1)
    fig.subplots_adjust(hspace=0.1)
    ax.set_ylabel('Transit Depth (%)')
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", rotation = 'horizontal', horizontalalignment = 'center')

    # Double-x-axis figure settings
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_ylim(0,1)

    secax = ax.secondary_xaxis('top', functions=(microns2wavenumbers,wavenumbers2microns))
    secax.set_xlabel(r'Wavelength ($\mu$m)')


    # ----------
    # Plotting JWST WASP-39b spectrum
    freq, intens, err = JWST_frequecies_intensities()
    g = ax.errorbar(freq, intens, yerr = err,color='black', linestyle='none', marker='s', markersize=4, elinewidth=1.0, alpha = 1)


    # ----------
    # Collecting ExoMol line lists in 'empty_dict' suitable for plotting
    columns = ['H2O','NH3','CH4','SO2','CO2','CO','PH3']
    colour_list = ['blue','g','grey','r','purple','y','orange']

    empty_dict = {}

    for mol,colour in zip(columns,colour_list):
        if mol != None:
            temp_data = ExoMol_data.query("Molecule == @mol")
            temp_data['Norm_Ints'] = np.array(temp_data['Intensity'])/np.nanmax(temp_data['Intensity'])
            freq, ints = temp_data['Frequency'], temp_data['Norm_Ints']

            empty_dict[mol] = list(freq),list(ints),colour


    # ----------
    # Plotting ExoMol line lists data alongside JWST WASP-39b spectrum
    molecule_list = [Molecule1, Molecule2, Molecule3, Molecule4]
    for molecule in molecule_list:
        if molecule != None:
            markers,stems,base = ax2.stem(empty_dict[molecule][0],
                                          empty_dict[molecule][1],
                                          markerfmt=' ', basefmt=' ',
                                          linefmt=empty_dict[molecule][-1])

            plt.setp(stems, 'linewidth', 1, alpha = 0.7)

    ax.set_xlim(1900,3600)


    # ----------
    # Plot legend settings
    custom_lines = [Line2D([], [], color='black', linewidth=2.0, marker='s',
                    markeredgecolor='black',markersize=8),
                    Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0],[0], color='grey',lw=4),
                    Line2D([0],[0], color='r',lw=4),
                    Line2D([0],[0], color='purple',lw=4),
                    Line2D([0],[0], color='y',lw=4),
		    Line2D([0],[0], color='orange',lw=4)]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(custom_lines,['WASP-39b Spectrum',
                            r'Water - H$_2$O',r'Ammonia - NH$_3$','Methane - CH$_4$',
                            'Sulphur Dioxide - SO$_2$','Carbon Dioxide - CO$_2$', 'Carbon Monoxide - CO',
			    'Phosphine - PH$_{3}$'],
                            loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

    return
