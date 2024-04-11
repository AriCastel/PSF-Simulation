"""_summary_
PSF Simulator by Ari the Lynx
release 1.0.20240409
"""

import phaseSimFunctions as phasesim 
import numpy as np
import matplotlib.pyplot as plt
import aotools
import seaborn as sns 
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Generate an aberrated OTF
noiseFactor = 0.1
sigma = 3
otf_shape = (256, 256)  # Define shape of the OTF
aberrated_otf = phasesim.poliphase.generate_aberrated_otf(otf_shape, sigma, noiseFactor, type='ideal')
logging.info("Original OTF created")

#Generates a Phase Mask 
mask_size = (256,256)
mask_shift = (40,0)
maskOrder = 3
maskFact = 1
#phaseMask = phasesim.poliphase.generate_phase_mask(maskOrder, mask_size, otf_shape, mask_shift)
phaseMask = phasesim.poliphase.generate_curtain(mask_size,150)
logging.info("Phase Mask Created")

#Generates a Curtain Binary phaseMask

#applies the PhaseMask to the OTF
phaseShiftedOTF = phasesim.matriarch.exponent_phase_mask(aberrated_otf, phaseMask,maskFact)
logging.info("Phase Mask applied")

# Generate PSF from aberrated OTF
psf = phasesim.poliphase.psf_from_otf(phaseShiftedOTF)
logging.info("PSF Obtained")
print('values, Max_Mean')
print(psf.max())
print(np.mean(psf))

#ZOOM IN THE psf    
zoomWidth = 256 # pixels
psizef = psf.shape[0]
zoomiePSF = psf[(int(psizef/2))-(int(zoomWidth/2)):(int(psizef/2))+(int(zoomWidth/2)), (int(psizef/2))-(int(zoomWidth/2)):(int(psizef/2))+(int(zoomWidth/2))]

# Plot the results
logging.info("Plotting Results")
plt.figure(figsize=(10, 5))
# Plot OTF
plt.subplot(1, 2, 1)
plt.imshow(np.abs(phaseShiftedOTF), cmap='rocket')
plt.title('Aberrated OTF')
plt.colorbar()
# Plot PSF
plt.subplot(1, 2, 2)
plt.imshow(zoomiePSF ,cmap='rocket', vmin=0)
plt.title('Point Spread Function (PSF)')
plt.colorbar()

plt.tight_layout()
plt.show()

#Plots the Azimuth Average of the PSF
aziAverage = aotools.azimuthal_average(zoomiePSF)
plt.plot(aziAverage, linestyle = 'dotted')
plt.xlabel('Radious')
plt.ylabel('Average')
plt.title('Azimuthal Average')
plt.grid(True)
plt.show()
