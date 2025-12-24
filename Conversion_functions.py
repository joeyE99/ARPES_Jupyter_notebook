import numpy as np

import lmfit as lmfit

import scipy as scipy
from scipy.ndimage import map_coordinates

from joblib import Parallel, delayed

me = (0.511*1e6)/(299792458 *1e10)**2  # eV s^2 Angstrom^-2
hbar = 6.582*1e-16   # eV s

########################################################################################################################

def find_value_index(matrix, y):
    # Ensure inputs are numpy arrays
    matrix = np.asarray(matrix)
    y = np.atleast_1d(y)
    
    # matrix shape: (N,)
    # y shape: (K, 1) to allow broadcasting
    y_reshaped = y[:, np.newaxis]
    
    diff = np.abs(matrix - y_reshaped)
    indices = diff.argmin(axis=1)
    
    return indices

########################################################################################################################

def Fermi_level_fit_ABfd(data1_angle_sum, fit_range, eb, hv, T, center=[0,-0.02,0.02]):
    index_range = find_value_index(eb, fit_range) # The data range in index of binding energy for fitting
    index_range = np.sort(index_range)  

    def affine_broadened_fd(
            x, center=0, T=30, conv_width=0.02, const_bkg=1, lin_bkg=0, offset=0
        ):
        """Fermi function convoled with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            fd_center: center of the step
            fd_width: width of the step
            conv_width: The convolution width
            const_bkg: constant background
            lin_bkg: linear background slope
            offset: constant background
        """
        dx = center-x
        x_scaling = np.abs(x[1] - x[0])
        fermi = 1 / (np.exp(np.clip(dx / (8.617e-5 * T), -700, 700)) + 1)
        return (
            scipy.ndimage.gaussian_filter((const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling)
            + offset
        )

    mod = lmfit.Model(affine_broadened_fd)

    # Initial guess, need manualy change for each data fitting
    params = mod.make_params()

    print('Fitting in process ...')

    x = eb[index_range[0]:index_range[1]]

    def fit_single_edge(y_data, x, params_template, mod):
        # Copy params to avoid conflicts between threads
        local_params = params_template.copy()
    
        # Set your dynamic parameters
        local_params['center'].set(min=center[1], max=center[2], value=center[0])
        local_params['T'].set(vary=True, min=max(0, T-5), max=T+5, value=T)
        local_params['conv_width'].set(vary=True, min=0)
        local_params['const_bkg'].set(min=y_data.max()/10, value = y_data.max())
        local_params['lin_bkg'].set(max=0, value=-100  )
        local_params['offset'].set(min=0, value = y_data.min())

        out = mod.fit(y_data, local_params, x=x, method='nelder')
    
        return (out.params['center'].value, out.params['T'].value, 
                out.params['conv_width'].value, out.params['const_bkg'].value, 
                out.params['lin_bkg'].value, out.params['offset'].value, out.rsquared)

    # Run in parallel
    fermi_fit_results = Parallel(n_jobs=-1)(
        delayed(fit_single_edge)(data1_angle_sum[index_range[0]:index_range[1], i], x, params, mod) 
        for i in range(len(hv))
    )
    return np.array(fermi_fit_results)

########################################################################################################################

def Fermi_level_fit_fd(data1_angle_sum, fit_range, eb, hv, T):
    index_range = find_value_index(eb, fit_range) # The data range in index of binding energy for fitting
    index_range = np.sort(index_range)  

    def Fermi_Dirac(x, amplitude, center, T, const_bkg, lin_bkg): 
        return amplitude * 1 / (np.exp( -(x-center)/(8.617e-5 * T) ) + 1) + const_bkg + lin_bkg * np.heaviside(x-center, 0.5) * (x-center)
    
    '''
    def Fermi_Dirac(x, amplitude, center, T, const_bkg, lin_bkg):
        test = np.zeros(len(x))
        for i in range(len(x)):
            test[i] = np.minimum( 1, np.maximum((x[i]-center)/0.0001,0) )

        return amplitude * 1 / (np.exp( -(x-center)/(8.617e-5 * T) ) + 1) + const_bkg + lin_bkg * test * (x-center)
    '''

    mod = lmfit.Model(Fermi_Dirac)

    # Initial guess, need manualy change for each data fitting
    params = mod.make_params()

    print('Fitting in process ...')

    x = eb[index_range[0]:index_range[1]]

    def fit_single_edge(y_data, x, params_template, mod):
        # Copy params to avoid conflicts between threads
        local_params = params_template.copy()
    
        # Set your dynamic parameters
        local_params['amplitude'].set(min=y_data.max()/10, value=y_data.max()/2)
        local_params['center'].set(min=-0.05, max=0.05, value=0)
        local_params['T'].set(vary=True, min=max(0, T-5), max=T+5, value=T)
        local_params['const_bkg'].set(min=0, value = y_data.max())
        local_params['lin_bkg'].set(value = 1  )

        out = mod.fit(y_data, local_params, x=x, method='nelder')
    
        return (out.params['center'].value, out.params['T'].value, 
                out.params['amplitude'].value, out.params['const_bkg'].value, out.params['lin_bkg'].value, 
                out.rsquared)

    # Run in parallel
    fermi_fit_results = Parallel(n_jobs=-1)(
        delayed(fit_single_edge)(data1_angle_sum[index_range[0]:index_range[1], i], x, params, mod) 
        for i in range(len(hv))
    )
    return np.array(fermi_fit_results)

########################################################################################################################

def convert_angle_to_kp(data, angle_axis, eb_axis, hv_axis, work_function, k_res_factor=1.0):
    """
    Converts 3D ARPES data from Angle to Momentum space.
    
    Data Structure:
    Axis 0: Emission Angle
    Axis 1: Binding Energy
    Axis 2: Photon Energy
    
    Parameters:
    -----------
    data : 3D array
        Shape (N_angle, N_Eb, N_hv)
    angle_axis : 1D array
        Emission angles in degrees.
    eb_axis : 1D array
        Binding Energy in eV.
    hv_axis : 1D array
        Photon Energies in eV.
    work_function : float
        Work function (eV) to calculate Kinetic Energy.
    k_res_factor : float
        Scales the resolution of the new k-axis relative to the angle axis.
        
    Returns:
    --------
    k_axis : 1D array
        The new linear momentum axis.
    new_data : 3D array
        Shape (N_k, N_Eb, N_hv)
    """
    
    # --- 1. SETUP CONSTANTS & BOUNDS ---
    C = 1/hbar * np.sqrt(2*me)
    angles_rad = np.deg2rad(angle_axis)
    
    # Calculate global k-range to ensure all data fits in the new matrix.
    # Max k occurs at: Max Photon Energy + Min Binding Energy (Max Kinetic Energy)
    max_hv = np.max(hv_axis)
    min_eb = np.min(eb_axis)
    max_E_kin = max_hv - work_function - min_eb
    
    
    # Maximum physical k covered by the detector
    max_k_val = C * np.sqrt(max_E_kin) * np.sin(np.max(angles_rad))
    min_k_val = min( 0, C * np.sqrt(max_E_kin) * np.sin(np.min(angles_rad)) )
    
    # Define new k-axis resolution
    n_k = int(len(angle_axis) * k_res_factor)
    k_axis = np.linspace(min_k_val, max_k_val, n_k).flatten()
    
    # Prepare Output Container (N_k, N_Eb, N_hv)
    n_eb = len(eb_axis)
    n_hv = len(hv_axis)
    new_data = np.zeros((n_k, n_eb, n_hv))
    
    # Pre-calculate angle axis parameters for index conversion
    
    for i_hv, hv in enumerate(hv_axis):
        
        current_slice = data[:, :, i_hv]
        
        # Calculate Kinetic Energy for this slice
        # Shape: (1, N_Eb) for broadcasting
        E_kin = (hv - work_function - eb_axis)[None, :]
        
        # Generate Target Coordinate Grid (Momentum, Eb)
        # We want to fill a matrix of shape (N_k, N_Eb)
        # k_vals shape: (N_k, 1)
        k_vals = k_axis[:, None]
        
        # D. Inverse Physics: Calculate which Angle corresponds to (k, Eb)
        # Result shape: (N_k, N_Eb)
        with np.errstate(divide='ignore', invalid='ignore'):
            sin_theta_grid = k_vals / (C * np.sqrt(E_kin + 1e-9))
            
        # Filter physically impossible k-values (outside the light cone)
        valid_mask = np.abs(sin_theta_grid) <= 1.0
        sin_theta_grid[~valid_mask] = 0 # Dummy value for arcsin
        
        theta_grid = np.arcsin(sin_theta_grid)
        
        # Map Physical Coordinates -> Array Indices
        angle_indices = np.interp(theta_grid.ravel(), angles_rad, np.arange(len(angle_axis)), left=-1, right=-1)
        
        # Broadcast eb to shape (N_k, N_Eb)
        eb_indices = np.arange(n_eb)[None, :] * np.ones((n_k, 1))
        
        # map_coordinates expects a (2, N_points) array of coordinates
        coords = np.array([angle_indices.ravel(), eb_indices.ravel()])
        
        interpolated_flat = map_coordinates(
            current_slice, 
            coords, 
            order=3,            # Cubic Spline for high accuracy and avoid oscillations
            mode='constant',    # Fill out-of-bounds with 0
            cval=0.0,
            prefilter=True
        )
        
        # Reshape back to (N_k, N_Eb)
        interpolated_slice = interpolated_flat.reshape(n_k, n_eb)
        
        # Re-apply validity mask (kill signal where sin(theta) > 1)
        interpolated_slice[~valid_mask] = 0.0
        new_data[:, :, i_hv] = interpolated_slice

    return k_axis, new_data

########################################################################################################################

def convert_hv_to_kz(data, k_para_axis, eb_axis, hv_axis, V0, work_function, kz_res_factor=1.0):
    """
    Converts ARPES data from (k_para, Eb, hv) to (k_para, Eb, kz).
    
    This performs a fully 3D-consistent 'k-warping' correction, accounting 
    for the dependence of kz on both Binding Energy and Parallel Momentum.
    
    Parameters:
    -----------
    data : 3D array
        Input data with shape (N_k_para, N_Eb, N_hv).
        (Output from the previous conversion function).
    k_para_axis : 1D array
        Parallel momentum axis (Axis 0).
    eb_axis : 1D array
        Binding Energy axis (Axis 1).
    hv_axis : 1D array
        Photon Energy axis (Axis 2).
    V0 : float
        Inner Potential (eV).
    work_function : float
        Work Function (eV).
    kz_res_factor : float
        Oversampling factor for the new kz axis.
        
    Returns:
    --------
    kz_axis : 1D array
        The new linear kz axis.
    new_data : 3D array
        Transformed data shape (N_k_para, N_Eb, N_kz).
    """
    
    # Constants
    C = 1/hbar * np.sqrt(2*me)
    C_sq = C**2
    
    # --- 1. DEFINE TARGET KZ GRID ---
    # kz range
    k_z_min_est = C * np.sqrt( np.min(hv_axis) - np.max(eb_axis) - work_function + V0 - np.max( np.abs(k_para_axis) )**2/C_sq ) 
    k_z_max_est = C * np.sqrt(np.max(hv_axis) - np.min(eb_axis) - work_function + V0)
    
    # Create the new linear axis
    n_kz = int(len(hv_axis) * kz_res_factor)
    kz_axis = np.linspace(k_z_min_est, k_z_max_est, n_kz)
    
    hv_indices = np.arange(len(hv_axis))
    
    # Shapes:
    # Kp: (N_kp, 1, 1)
    # Eb: (1, N_eb, 1)
    # Kz: (1, 1, N_kz)
    Kp = k_para_axis[:, None, None]
    Eb = eb_axis[None, :, None]
    Kz = kz_axis[None, None, :]
    
    # INVERSE PHYSICS CALCULATION
    required_hv = ((Kp**2 + Kz**2) / C_sq) - V0 + work_function + Eb
    
    # CONVERT HV TO INDICES
    # Flatten for speed
    hv_map_indices = np.interp(required_hv.ravel(), hv_axis, hv_indices, left=-1, right=-1)
    hv_map_indices = hv_map_indices.reshape(required_hv.shape)
    
    idx_0 = np.arange(len(k_para_axis))[:, None, None] * np.ones((1, len(eb_axis), n_kz))
    idx_1 = np.ones((len(k_para_axis), 1, n_kz)) * np.arange(len(eb_axis))[None, :, None]
    
    # Stack into shape (3, N_pixels)
    coords = np.array([idx_0.ravel(), idx_1.ravel(), hv_map_indices.ravel()])
    
    # SPLINE INTERPOLATION
    new_data_flat = map_coordinates(
        data,
        coords,
        order=3,            # Cubic spline
        mode='constant',    # Fill with 0
        cval=0.0,
        prefilter=True
    )
    
    # Reshape to final 3D block
    new_data = new_data_flat.reshape(len(k_para_axis), len(eb_axis), n_kz)
    
    # Mask values where required_hv was out of range (idx == -1)
    # map_coordinates handles -1 by interpolating near edge or filling 0, 
    # but strictly speaking, if hv is outside range, data is invalid.
    mask = (hv_map_indices == -1)
    new_data[mask] = 0
    
    return kz_axis, new_data

########################################################################################################################

def convert_angle_to_k_map(data, eb_axis, angle_axis, tilt_axis, hv, work_function, k_res_factor=1.0):
	"""
	Converts 3D ARPES data from Angle to Momentum space.
	
	Data Structure:
	Axis 0: Binding Energy
	Axis 1: Emission Angle
	Axis 2: Tilt Angle
	
	Parameters:
	-----------
	data : 3D array
		Shape (N_Eb, N_angle, N_tilt)
	eb_axis : 1D array
		Binding Energy in eV.
	angle_axis : 1D array
		Emission angles in degrees.
	tilt_axis : 1D array
		Tilt angles in degrees.
	hv : float
		Photon energies in eV.
	work_function : float
		Work function (eV) to calculate Kinetic Energy.
	k_res_factor : float
		Scales the resolution of the new k-axis relative to the angle axis.
		
	Returns:
	--------
	kx_axis : 1D array
		The new linear momentum axis corresponding to emission angle.
	ky_axis : 1D array
		The new linear momentum axis corresponding to tilt angle.
	new_data : 3D array
		Shape (N_Eb, N_kx, N_ky)
	"""
	
	# --- 1. SETUP CONSTANTS & BOUNDS ---
	C = 1/hbar * np.sqrt(2*me)
	angles_x_rad = np.deg2rad(angle_axis)
	angles_y_rad = np.deg2rad(tilt_axis)
	
	# Maximum & Minimum physical k covered by the detector
	min_eb = np.min(eb_axis)
	max_E_kin = hv - work_function - min_eb
	max_kx_val = C * np.sqrt(max_E_kin) * np.sin(np.max(angles_x_rad))
	max_ky_val = C * np.sqrt(max_E_kin) * np.sin(np.max(angles_y_rad))
	min_kx_val = min( 0, C * np.sqrt(max_E_kin) * np.sin(np.min(angles_x_rad)) )
	min_ky_val = min( 0, C * np.sqrt(max_E_kin) * np.sin(np.min(angles_y_rad)) )

	# Define new k-axis resolution
	n_kx = int(len(angle_axis) * k_res_factor)
	n_ky = int(len(tilt_axis) * k_res_factor)
	kx_axis = np.linspace(min_kx_val, max_kx_val, n_kx).flatten()
	ky_axis = np.linspace(min_ky_val, max_ky_val, n_ky).flatten()
	
	# Prepare Output Container (N_Eb, N_kx, N_ky)
	n_eb = len(eb_axis)
	new_data = np.zeros((n_eb, n_kx, n_ky))
		
	# Shape: (N_Eb, N_kx, N_ky) for broadcasting
	Eb = eb_axis[:, None, None]
	Kx = kx_axis[None, :, None]
	Ky = ky_axis[None, None, :]

	E_kin = (hv - work_function - Eb)
	
	# Inverse Physics: Calculate which Tilt Angle corresponds to (Eb, ky)
	# Result shape: (N_Eb, N_kx, N_ky)
	with np.errstate(divide='ignore', invalid='ignore'):
		sin_phi_grid = Ky / (C * np.sqrt(E_kin + 1e-9))
		sin_phi_grid = sin_phi_grid * np.ones((1,n_kx,1)) 
	# Filter physically impossible k-values (outside the light cone)
	valid_mask_phi = np.abs(sin_phi_grid) <= 1.0
	sin_phi_grid[~valid_mask_phi] = 0 # Dummy value for arcsin
	phi_grid = np.arcsin(sin_phi_grid)
	
	# Inverse Physics: Calculate which Emission Angle corresponds to (Eb, kx, ky)
	# Result shape: (N_Eb, N_kx, N_ky)
	with np.errstate(divide='ignore', invalid='ignore'):
		sin_theta_grid = Kx / (C * np.sqrt(E_kin + 1e-9) * np.cos(phi_grid))
	valid_mask_theta = np.abs(sin_theta_grid) <= 1.0
	sin_theta_grid[~valid_mask_theta] = 0 # Dummy value for arcsin
	theta_grid = np.arcsin(sin_theta_grid)

	# Map Physical Coordinates -> Array Indices
    # Flatten for Speed 
	angle_indices = np.interp(theta_grid.ravel(), angles_x_rad, np.arange(len(angle_axis)), left=-1, right=-1)
	tilt_indices = np.interp(phi_grid.ravel(), angles_y_rad, np.arange(len(tilt_axis)), left=-1, right=-1)

	# Broadcast eb to shape (N_Eb, , N_ky)
	eb_indices = np.arange(n_eb)[:, None, None] * np.ones((1, n_kx, n_ky))
	
	# map_coordinates expects a (3, N_points) array of coordinates
	coords = np.array([eb_indices.ravel(), angle_indices.ravel(), tilt_indices.ravel()])
	
	interpolated_flat = map_coordinates(
		data, 
		coords, 
		order=3,            # Cubic Spline for high accuracy and avoid oscillations
		mode='constant',    # Fill out-of-bounds with 0
		cval=0.0,
		prefilter=True
	)

	# Reshape back to (N_Eb, N_kx, N_ky)
	new_data = interpolated_flat.reshape(n_eb, n_kx, n_ky)
		
	# Re-apply validity mask (kill signal where sin(theta) > 1)
	new_data[~valid_mask_phi] = 0.0
	new_data[~valid_mask_theta] = 0.0
	
	return kx_axis, ky_axis, new_data