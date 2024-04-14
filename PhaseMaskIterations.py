import phaseSimFunctions as phasesim 
import numpy as np
import matplotlib.pyplot as plt
import aotools
import seaborn as sns 
import logging 
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Declare important lists were results will be saved
means = []
maximums = []
pointImages = []

# Generate an aberrated OTF
noiseFactor = 0.1
sigma = 2
otf_shape = (256, 256)  # Define shape of the OTF
aberrated_otf = phasesim.poliphase.generate_aberrated_otf(otf_shape, sigma, noiseFactor, type='idea')
logging.info("Original OTF created")
mask_size = (256,256)
maskFact =1

#Loop Code
start = 0
end = otf_shape[0]
num_points = 20  # Number of points in the interval
interval = np.linspace(start, end, num_points)

#additional variables
psizef = mask_size[0]
zoomWidth = 256

logging.info("Entering the simulation Loop")
for i in tqdm(range(len(interval))):
    phaseMask = phasesim.poliphase.generate_curtain(mask_size,int(interval[i]))
    phaseShiftedOTF = phasesim.matriarch.exponent_phase_mask(aberrated_otf, phaseMask,maskFact)
    psf = phasesim.poliphase.psf_from_otf(phaseShiftedOTF)
    #calculates the measurable variables
    maximums.append(psf.max())
    means.append(np.mean(psf))
    #zooms in the PSF
    psizef = psf.shape[0]
    zoomedPSF = psf [(int(psizef/2))-(int(zoomWidth/2)):(int(psizef/2))+(int(zoomWidth/2)), (int(psizef/2))-(int(zoomWidth/2)):(int(psizef/2))+(int(zoomWidth/2))]
    pointImages.append(zoomedPSF)

logging.info("Simulation Loop Sucessful") 
    
#Animates the PSF change
logging.info("Animating the psf") 
# Create a figure and axis for plotting
fig, ax = plt.subplots()

# Define a function to update the plot for each frame of the animation
def update(frame):
    ax.clear()
    ax.imshow(pointImages[frame], cmap='rocket', vmax=max(maximums))  # Adjust colormap as needed
    ax.set_title(f'Fourier plane occlusion: {frame}%')
    ax.set_axis_off()  # Turn off axis


# Create the animation
ani = FuncAnimation(fig, update, frames=len(pointImages), interval=41,)


plt.show()
# If you want to save the animation to a file
ani.save('PSF.GIF', writer='ffmpeg')

    
#Plot the Mean and Maximum values of the PSF    
logging.info("Plotting Results")
plt.figure(figsize=(10, 5))
# Plot maximums
plt.subplot(1, 2, 1)
plt.plot(interval/mask_size[0], maximums/max(maximums), color='#240046', linewidth=2, linestyle='-')
plt.scatter(interval/mask_size[0], maximums/max(maximums), marker='d', color='#B40424')
plt.title('PSF Maximum Intensities',fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Fourier plane occlusion', fontsize=18)
plt.ylabel('Maximum PSF intensity', fontsize=18)

# Plot averages
plt.subplot(1, 2, 2)
plt.plot(interval/mask_size[0], means/min(means), color='#240046', linewidth=2, linestyle='-')
plt.scatter(interval/mask_size[0], means/min(means),  marker='d', color='#B40424')
plt.title('PSF Mean Intensity', fontsize=20)
plt.grid(True, color='#E1E2EF')  # Add grid lines
plt.xlabel('Fourier plane occlusion', fontsize=18)
plt.ylabel('Mean PSF Intesity', fontsize=18)
plt.show()
