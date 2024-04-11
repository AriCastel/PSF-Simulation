"""_summary_
PSF Simulator Functions by Ari the Lynx
release 1.0.20240409
"""

import logging
import numpy as np
import aotools
import seaborn as sns 
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class poliphase:

    def generate_curtain(shape, split_column):
        matrix = np.zeros(shape)  # Create a matrix of zeros
        matrix[:, :split_column] = 1     # Set columns before split_column to 1
        curtainGraph = sns.heatmap(matrix, linewidth=0, square=True, xticklabels = 25, yticklabels= 25, cmap='rocket')
        plt.title('Curtain Phase Mask')
        plt.show()
        return matrix

    def generate_phase_mask(order, shape, pupilShape, shift=(0,0), fact=1):
        """
        Generate a phase mask Using a Zernike Polinomial.
        
        Parameters:
            order = int, order of the zernike Polynomial to Use
            shape: tuple, shape of the Phase mask.
            pupilShape: tuple, shape of the pupil plane.
            shift: tupe, introduces a shift to the phase mask (I.E. DM/SLM aren't correctly centered)
            type: wheter the psf is a Gaussian function or an ideal aperture
            
        Returns:
            ShiftedMask: ndarray, Final Phase Mask.
        """
        zernike = aotools.zernikeArray(order+1, shape[0])
        polynomial = zernike[order]
        sizedMask = matriarch.phase_sizing(pupilShape[0], polynomial)
        shiftedMask = matriarch.shift_image(sizedMask, shift[0], shift[1])
        
        shiftGraph = sns.heatmap(shiftedMask, linewidth=0, square=True, xticklabels = 25, yticklabels= 25, cmap='rocket')
        plt.title('Off-center defocus mask')
        plt.show()
            
        return shiftedMask
        
    def generate_aberrated_otf(shape, sigma=1.0, aberration_factor=0.1, type = 'gaussian'):
        """
        Generate an aberrated Optical Transfer Function (OTF).
        
        Parameters:
            shape: tuple, shape of the OTF.
            sigma: float, standard deviation of the Gaussian distribution.
            aberration_factor: float, scaling factor for aberration strength.
            type: wheter the psf is a Gaussian function or an ideal aperture
            
        Returns:
            aberrated_otf: ndarray, aberrated OTF.
        """
        x = np.linspace(-5, 5, shape[0])
        y = np.linspace(-5, 5, shape[1])
        xx, yy = np.meshgrid(x, y)
        
        if type == 'ideal':
            zernike = aotools.zernikeArray(1, shape[0])
            otf = zernike[0]
        else:
            # Generate a Gaussian distribution as a base OTF
            otf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Add Noise to the OTF
        aberration = aberration_factor * np.random.rand(*shape)
        aberrated_otf = otf + aberration
        
        #trim OTF to fit the aperture
        Polynomial = aotools.zernikeArray(1, shape[0])
        aberrated_otf = aberrated_otf*Polynomial[0]
        logging.debug(aberrated_otf.shape)
        
        return aberrated_otf

    def psf_from_otf(otf):
        """
        Generate the Point Spread Function (PSF) from the Optical Transfer Function (OTF).
        
        Parameters:
            otf: ndarray, Optical Transfer Function.
            
        Returns:
            psf: ndarray, Point Spread Function.
        """
        padFact = 10
        
        padded_otf = np.pad(otf, [(0, padFact*(otf.shape[0])), (0, padFact*(otf.shape[1]))], mode='constant')
        psf = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(padded_otf)))
        
        
        psf = np.abs(psf)
        #psf /= np.sum(psf)  # Normalize PSF
        
        #scales Up the PSF for the sake of Graphing
        #psf = psf*padFact*25
        
        return psf


class matriarch:
    
    def exponent_phase_mask(OTF, phase_shift, maxS):
        """
        Applies a phase Shift to an specified OTF Matrix.
    
        Parameters:
            OTF: Matrix we want to apply the phase shift to.
            phase_shift: Matrix containing the phase shift for each OTF term.
            maxS: additional scaling factor for the phase mask
        
        Returns:
            shiftedOTF: ndarray, shifted OTF.
        """
        logging.debug('Phase Mask application entered')

        #"Normalizes" the phase mask in respect to pi, such that the phase mask goes from -pi to pi
        maxmat = phase_shift.max()
        alpha = (np.pi)/maxmat
        phase_shift = phase_shift*alpha
        
        
        try:
            shifted = OTF* np.exp(1j*phase_shift*maxS)
            logging.debug('Phase Mask application done')
            return shifted 
            
        except Exception as e:
            logging.error('Unable to apply phase shift')
            logging.exception('An error occurred: %s', e)
            return OTF
        
    def frame_image (frame, image, center_point):
        """
        places the values of the image matrix inside the frame matrix with 
        center_point as the center point.
        Stable only where frame can contain image

        Parameters:
            frame (numpy.ndarray): Larger matrix to be modified.
            image (numpy.ndarray): Smaller matrix whose values will be placed into the larger matrix.
            center_point (tuple): Coordinates (row, column) specifying the center point.

        Returns:
            numpy.ndarray: Modified frame matrix.
        """
        smaller_rows, smaller_cols = image.shape
        center_row, center_col = center_point
        start_row = max(center_row - smaller_rows // 2, 0)
        end_row = min(start_row + smaller_rows, frame.shape[0])
        start_col = max(center_col - smaller_cols // 2, 0)
        end_col = min(start_col + smaller_cols, frame.shape[1])

        frame[start_row:end_row, start_col:end_col] = image[:end_row-start_row, :end_col-start_col]

        return frame
    
    def shift_image(I, dx, dy):
        """_Shifts the center of the I image by dx and dy using the numpy roll function_

        Raises: I image array; dx, dy shift integers 
            ValueError: _description_

        Returns: Shifted array
            _type_: _description_
        """        
        I = np.roll(I, dy, axis=0)
        I = np.roll(I, dx, axis=1)
        if dy>0:
            I[:dy, :] = 0
        elif dy<0:
            I[dy:, :] = 0
        if dx>0:
            I[:, :dx] = 0
        elif dx<0:
            I[:, dx:] = 0
        return I
    
    def phase_sizing(pupilSize, zeries):
        zize = len(zeries)
        
        if zize < pupilSize: 

            #Takes the 'Order' mode of the Zernike Polinomials
            preMask = zeries * aotools.circle((pupilSize/2), zize) 
            #fits the phase mask into an array of 'SLMSize' size
            phaseMask = np.zeros((pupilSize,pupilSize))
            phaseMask[(int(pupilSize/2))-(int(zize/2)):(int(pupilSize/2))+(int(zize/2)), (int(pupilSize/2))-(int(zize/2)):(int(pupilSize/2))+(int(zize/2))] = preMask
            return phaseMask

        elif zize > pupilSize:

            phaseMask = zeries
            phaseMask = phaseMask[(int(zize/2))-(int(pupilSize/2)):(int(zize/2))+(int(pupilSize/2)), (int(zize/2))-(int(pupilSize/2)):(int(zize/2))+(int(pupilSize/2))]
            return phaseMask
        
        elif zize == pupilSize:
            return zeries 