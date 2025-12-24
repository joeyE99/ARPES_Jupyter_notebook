import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from cmcrameri import cm
import matplotlib.colors as colors
cmp = cm.devon
cmp = cmp.reversed()

import ipywidgets as widgets

import scipy as scipy

import os

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
    
    # Calculate absolute difference: (K, N)
    # Each row corresponds to a value in 'y'
    # Each column corresponds to an element in 'matrix'
    diff = np.abs(matrix - y_reshaped)
    
    # Find the index of the minimum difference for each row
    indices = diff.argmin(axis=1)
    
    return indices

########################################################################################################################

def curve_interactive(data, x, y):
	
	fig, ax = plt.subplots(1, 1, figsize=(4, 2))
	
	control_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=0, description='y Index', continuous_update=True )

	curve_data = ax.plot( x, data[:, control_index.value], zorder=1, label='Data', marker='+' ) 
	ax.set_xlim(np.min(x), np.max(x))
	title_text = ax.set_title('y = ' + str(round(y[control_index.value],2)) )
	
	def update_plot(change=None):
		curve_data[0].set_ydata( data[:, control_index.value])
		ax.relim()
		ax.autoscale(axis='y')
		title_text.set_text('y = ' + str(round(y[control_index.value],2)) )
		
		fig.canvas.draw_idle()
	

	control_index.observe(update_plot, names='value')

	output_widget = widgets.Output()
	ui = widgets.VBox([control_index, output_widget])

	display(ui)

	fig.canvas.header_visible = False
	fig.canvas.toolbar_visible = 'fade-in-fade-out'
	#fig.canvas.footer_visible = False
	
	return fig, ax
	
########################################################################################################################

def slice_3D(data, axes, sample_name, output_filepath, figsize=(4, 4), cmap=cmp):
	if output_filepath and not os.path.exists(output_filepath):
		os.makedirs(output_filepath, exist_ok=True)

	# Use constrained_layout for better axis label handling
	fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
	
	# 2. State & Constants
	axis_map = {'X': 0, 'Y': 1, 'Z': 2}
	axis_labels_list = ['X', 'Y', 'Z']
	
	state = {
		'mesh': None,
		'v_min_arr': None,
		'v_max_arr': None,
		'current_axis_idx': 0,
		'previous_axis_idx': 1,
		'other_axes': (1, 2), # (horizontal_idx, vertical_idx)
		'axvline': None, # Vertical line
		'axhline': None, # Horizontal line
		'bar': None # Measured bar
	}

	# 3. Widgets
	w_axis = widgets.RadioButtons(
		options=['X', 'Y', 'Z'], value='X', 
		description='Slice Axis', 
		layout=widgets.Layout(width='100px')
	)
	
	w_index = widgets.IntSlider(description='Slice Index', continuous_update=True)
	w_gamma = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description=r'$\gamma$')

	w_inv_y = widgets.Checkbox(value=False, description='Invert Y', layout=widgets.Layout(width='30%'), indent=False)
	w_swap = widgets.Checkbox(value=False, description='Swap Axes', layout=widgets.Layout(width='30%'), indent=False)
	
	# Bar/Crosshair controls
	w_bar_len = widgets.FloatSlider(description='Bar Len', continuous_update=True)
	w_bar_pos = widgets.FloatSlider(description='Bar Pos', continuous_update=True)
	
	w_show_bar = widgets.Checkbox(value=False, description='Show Bar', layout=widgets.Layout(width='30%'), indent=False)
	w_save = widgets.Button(description="Save Image", icon="save", layout=widgets.Layout(width='40%'), indent=False)

	# 4. Helper Functions
	def update_state(slice_ax_idx):
		remaining = tuple(i for i in range(3) if i != slice_ax_idx)
		state['current_axis_idx'] = slice_ax_idx
		state['other_axes'] = remaining
		state['v_min_arr'] = np.nanmin(data, axis=remaining)
		state['v_max_arr'] = np.nanmax(data, axis=remaining)

	def get_current_coords():
		ax_h_idx, ax_v_idx = state['other_axes']
		
		if w_swap.value:
			h_idx, v_idx = ax_v_idx, ax_h_idx
		else:
			h_idx, v_idx = ax_h_idx, ax_v_idx

		return h_idx, v_idx

	def update_slider_ranges():
		h_idx, v_idx = get_current_coords()
		
		if not (state['current_axis_idx'] == state['previous_axis_idx']):
			# Update Slice Index Slider
			slice_dim = state['current_axis_idx']
			max_idx = len(axes[slice_dim]) - 1
		
			w_index.unobserve(draw_plot_wrapper, names='value')
			w_index.max = max_idx
			w_index.value = np.abs(axes[slice_dim] - 0).argmin()
			w_index.observe(draw_plot_wrapper, names='value')
			state['previous_axis_idx'] = state['current_axis_idx']

		# Update Bar/Crosshair Sliders (Associated with the Horizontal Plot Axis)
		# The bar moves along the X-axis of the plot
		x_axis_data = axes[h_idx]
		x_min, x_max = np.min(x_axis_data), np.max(x_axis_data)
		x_range = x_max - x_min
		
		w_bar_len.unobserve(draw_plot_wrapper, names='value')
		w_bar_pos.unobserve(draw_plot_wrapper, names='value')

		w_bar_len.min = 0
		w_bar_len.max = x_range
		w_bar_len.step = x_range / 250.0

		w_bar_pos.min = min(w_bar_pos.min, x_min)
		w_bar_pos.max = max(w_bar_pos.max, x_max)
		w_bar_pos.min = x_min
		w_bar_pos.max = x_max
		w_bar_pos.step = x_range / 500.0
		
		# Restore observers
		w_bar_len.observe(draw_plot_wrapper, names='value')
		w_bar_pos.observe(draw_plot_wrapper, names='value')

	def draw_plot(force_clear=False):
		idx = w_index.value
		slice_dim = state['current_axis_idx']
		
		slice_data = np.take(data, idx, axis=slice_dim)
		h_idx, v_idx = get_current_coords()
		if w_swap.value:
			plot_data = slice_data
		else:
			plot_data = slice_data.T
			
		xl = axis_labels_list[h_idx]
		yl = axis_labels_list[v_idx]
		
		# Normalization
		vmin = state['v_min_arr'][idx]
		vmax = state['v_max_arr'][idx]
		norm = colors.PowerNorm(gamma=w_gamma.value, vmin=vmin, vmax=vmax)

		if force_clear or state['mesh'] is None:
			ax.clear()
			state['mesh'] = ax.pcolormesh(axes[h_idx], axes[v_idx], plot_data, 
										  cmap=cmap, norm=norm, shading='auto', zorder=1)
			
			ax.set_xlabel(xl)
			ax.set_ylabel(yl)
			ax.set_xlim(np.min(axes[h_idx]), np.max(axes[h_idx]))
			ax.set_ylim(np.min(axes[v_idx]), np.max(axes[v_idx]))
			
			state['axvline'] = ax.axvline(w_bar_pos.value, color='red', ls='--', lw=1, zorder=2)
			state['axhline'] = ax.axhline(0, color='red', ls='--', lw=1, zorder=2)
			
			# Measurement Bar
			half_len = w_bar_len.value / 2
			state['bar'], = ax.plot(
				[w_bar_pos.value - half_len, w_bar_pos.value + half_len],
				[0, 0], color='red', lw=3, zorder=3 )
			
			if w_inv_y.value:
				ax.invert_yaxis()
				
		else:
			state['mesh'].set_array(plot_data.ravel())
			state['mesh'].set_norm(norm)
			if w_show_bar.value:
				pos = w_bar_pos.value
				length = w_bar_len.value
				state['axvline'].set_xdata([pos])
				state['bar'].set_xdata([pos - length/2, pos + length/2])

		# Update Title
		coord_val = axes[slice_dim][idx]
		ax.set_title(f"{axis_labels_list[slice_dim]} Slice: {coord_val:.2f} (idx: {idx})")
		
		state['axvline'].set_visible(w_show_bar.value)
		state['axhline'].set_visible(w_show_bar.value)
		state['bar'].set_visible(w_show_bar.value)

		fig.canvas.draw_idle()

	# Callbacks
	def on_axis_change(change):
		new_axis = change['new']
		new_idx = axis_map[new_axis]
		update_state(new_idx)
		update_slider_ranges()
		draw_plot(force_clear=True)

	def on_config_change(change):
		if (change['owner'] == w_swap):
			update_slider_ranges()
		force = (change['owner'] == w_swap) or (change['owner'] == w_inv_y)
		draw_plot(force_clear=force)

	def draw_plot_wrapper(change):
		draw_plot(force_clear=False)

	def on_save(b):
		idx = w_index.value
		axis_name = w_axis.value
		slice_dim = axis_map[axis_name]
		val = axes[slice_dim][idx]
		
		# Handle sample name
		s_name = sample_name
		if isinstance(sample_name, (list, tuple)):
			s_name = sample_name[1] if len(sample_name) > 1 else sample_name[0]
			
		fname = f"{s_name}_{axis_name}_{val:.2f}.png"
		path = os.path.join(output_filepath, fname)
		fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
		print(f"Saved: {path}")

	# Linkage
	# Heavy updates
	w_axis.observe(on_axis_change, names='value')
	w_swap.observe(on_config_change, names='value')
	w_show_bar.observe(on_config_change, names='value')

	w_inv_y.observe(on_config_change, names='value')
	
	# Fast updates
	w_index.observe(draw_plot_wrapper, names='value')
	w_gamma.observe(draw_plot_wrapper, names='value')
	w_bar_len.observe(draw_plot_wrapper, names='value')
	w_bar_pos.observe(draw_plot_wrapper, names='value')
	
	w_save.on_click(on_save)

	# Initialization
	update_state(0) # Start with X
	update_slider_ranges()
	
	# Layout assembly
	box_layout = widgets.Layout(display='flex', flex_flow='row', align_items='center', width='100%')
	
	row1 = widgets.HBox([w_axis, w_index, w_gamma], layout=box_layout)
	row2 = widgets.HBox([w_swap, w_inv_y, w_show_bar, w_save], layout=box_layout)
	row3 = widgets.HBox([w_bar_pos, w_bar_len], layout=box_layout)
	
	ui = widgets.VBox([row1, row2, row3])

	controls_row1 = widgets.HBox([w_inv_y, w_swap, w_show_bar, w_save], 
								 layout=widgets.Layout(width='80%', justify_content='space-between'))
	controls_row2 = widgets.HBox([w_index, w_gamma], 
								 layout=widgets.Layout(width='100%', justify_content='space-between'))
	controls_row3 = widgets.HBox([w_bar_pos, w_bar_len], 
								 layout=widgets.Layout(width='100%', justify_content='space-between'))
	
	right_panel = widgets.VBox([controls_row1, controls_row2, controls_row3], layout=widgets.Layout(width='60%'))
	ui = widgets.HBox([w_axis, right_panel], layout=widgets.Layout(width='75%'))
	
	draw_plot(force_clear=True)
	
	display(ui)
	
	fig.canvas.header_visible = False
	fig.canvas.toolbar_visible = 'fade-in-fade-out'
	#fig.canvas.footer_visible = False

	return fig, ax
	
########################################################################################################################
	 
def EDC_angle_sum(data, eb, z, eb_range, sample_name, output_filepath):

	fig, ax = plt.subplots(1, 1, figsize=(4,2), layout='constrained')
	
	eb_index_range = find_value_index(eb, eb_range)
	
	x = eb[eb_index_range[1]:eb_index_range[0]]

	control_z_index = widgets.IntSlider( min=0, max=len(z)-1, step=1, value=0, description='Z Index', continuous_update=True )
	if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )

	button = widgets.Button(description="Savefig")

	curve_data = ax.plot( x, data[eb_index_range[1]:eb_index_range[0],control_z_index.value], zorder=1, label='Data', marker='+' ) 
	vline = ax.axvline(x=0, linestyle='--', linewidth=1, zorder=0, color='orange')
	ax.set_xlabel("Binding Energy (eV)")
	ax.set_yticks([])
	ax.set_xlim(np.min(x), np.max(x))
	title_text = ax.set_title("Z = "+str(round(z[control_z_index.value],2)))

	def update_plot(change=None):
		control_index = control_z_index.value
		curve_data[0].set_ydata(data[eb_index_range[1]:eb_index_range[0],control_index])
		ax.relim()
		ax.autoscale(axis='y')
		title_text.set_text("Z = "+str(round(z[control_index],2)))
		fig.canvas.draw_idle() 

	def update_vline(change):
		vline.set_visible(change['new'])
		fig.canvas.draw_idle() 

	def on_button_clicked(b):
		fig.savefig(output_filepath + sample_name[1]+'_EDC_'+str(z[control_z_index.value])+'.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
	button.on_click(on_button_clicked)

	control_z_index.observe(update_plot, names='value')
	if_axvline.observe(update_vline, names='value')

	output_widget = widgets.Output()
	ui_controls = widgets.HBox([control_z_index, if_axvline, button])
	ui = widgets.VBox([ui_controls, output_widget])
	
	display(ui)
	
	fig.canvas.header_visible = False
	#fig.canvas.toolbar_visible = False
	#fig.canvas.footer_visible = False

	return fig, ax
	
########################################################################################################################  
	
def Affine_broadened_Fermi_edge_fit_plot(
		data, eb, hv, fit_range, fermi_edge_center, fermi_edge_T, fermi_edge_conv_width, fermi_edge_const_bkg, fermi_edge_lin_bkg, fermi_edge_offset, sample_name, output_filepath
	):
	
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize=(10,3), layout='constrained')
	
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
		fermi = 1 / (np.exp(dx / (8.617e-5 * T) ) + 1)
		return (
			scipy.ndimage.gaussian_filter((const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling)
			+ offset
		)
	index_range = find_value_index(eb, fit_range) # The data range in index of binding energy for fitting
	index_range = np.sort(index_range)  
	x = eb[index_range[0]:index_range[1]]
	x_fit = np.linspace(x[0], x[-1], num=100, endpoint=True)

	control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
	
	if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )

	button = widgets.Button(description="Savefig")

	curve_data = ax1.plot( x, data[index_range[0]:index_range[1],control_hv_index.value], zorder=0, label='Data', marker='+' )
	curve_fit = ax1.plot( x_fit,  affine_broadened_fd(
								x_fit,
								fermi_edge_center[control_hv_index.value], fermi_edge_T[control_hv_index.value], fermi_edge_conv_width[control_hv_index.value], 
								fermi_edge_const_bkg[control_hv_index.value], fermi_edge_lin_bkg[control_hv_index.value], fermi_edge_offset[control_hv_index.value]
								), zorder=1, label='Fit')  
	ax1.set_xlabel("Binding Energy (eV)")
	#ax1.set_yticks([])
	ax1.set_xlim(np.min(x), np.max(x))
	vline = ax1.axvline(0, linestyle='--', color='orange', lw=1)
	vline_fit = ax1.axvline(fermi_edge_center[control_hv_index.value], linestyle='--', color='red', lw=1, label='Fit Edge')
	ax1.legend(loc=2)
	title_text1 = ax1.set_title("hv = "+str(round(hv[control_hv_index.value],2))+" eV")
	
	ax2.plot(hv, fermi_edge_center, zorder=0)
	scatter = ax2.scatter(hv[control_hv_index.value], fermi_edge_center[control_hv_index.value], color='r', zorder=1)
	ax2.set_xlabel("hv (eV)")
	ax2.set_ylabel("Offset (eV)")
	title_text2 = ax2.set_title('Offset = '+str(round(fermi_edge_center[control_hv_index.value],4))+ ' eV' )
	
	ax3.plot(hv, fermi_edge_T, zorder=0)
	scatter2 = ax3.scatter(hv[control_hv_index.value], fermi_edge_T[control_hv_index.value], color='r', zorder=1)
	ax3.set_xlabel("hv (eV)")
	ax3.set_ylabel(r"$T$ (K)")
	title_text3 = ax3.set_title(r'$T$ = '+str(round(fermi_edge_T[control_hv_index.value],2))+ ' K' )

	def update_plot(change=None):
		control_index = control_hv_index.value
		curve_data[0].set_ydata(data[index_range[0]:index_range[1],control_index])
		curve_fit[0].set_ydata(affine_broadened_fd(
								x_fit,
								fermi_edge_center[control_index], fermi_edge_T[control_index], fermi_edge_conv_width[control_index], 
								fermi_edge_const_bkg[control_index], fermi_edge_lin_bkg[control_index], fermi_edge_offset[control_index]
								))
		vline_fit.set_xdata([fermi_edge_center[control_index]])
		ax1.relim()
		ax1.autoscale(axis='y')
		title_text1.set_text("hv = "+str(round(hv[control_index],2))+" eV")
		
		scatter.set_offsets([hv[control_index], fermi_edge_center[control_index]])
		title_text2.set_text('Offset = '+str(round(fermi_edge_center[control_index],4))+ ' eV' )
		
		scatter2.set_offsets([hv[control_index], fermi_edge_T[control_index]])
		title_text3.set_text(r'$T$ = '+str(round(fermi_edge_T[control_index],2))+ ' K' )
  
		fig.canvas.draw_idle()

	def update_axvline(change):
		vline.set_visible(change['new'])
		vline_fit.set_visible(change['new'])
		fig.canvas.draw_idle()
 
	def on_button_clicked(b):
		fig.savefig(output_filepath + sample_name[1]+'_Fermi_edge_fit_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
	button.on_click(on_button_clicked)

	control_hv_index.observe(update_plot, names='value')
	if_axvline.observe(update_axvline, names='value')

	output_widget = widgets.Output()
	ui_controls = widgets.HBox([control_hv_index, if_axvline, button])
	ui = widgets.VBox([ui_controls, output_widget])

	display(ui)

	fig.canvas.header_visible = False
	#fig.canvas.toolbar_visible = False
	#fig.canvas.footer_visible = False
	
	return fig, ax1, ax2, ax3

########################################################################################################################

def Fermi_edge_fit_plot(data, eb, hv, fit_range, fermi_edge_T, fermi_edge_center, fermi_edge_amplitude, fermi_edge_const_bkg, fermi_edge_lin_bkg, sample_name, output_filepath):
	
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize=(10,3), layout='constrained')
	
	def Fermi_Dirac(x, amplitude, center, T, const_bkg, lin_bkg): 
		return amplitude * 1 / (np.exp( -(x-center)/(8.617e-5 * T) ) + 1) + const_bkg + lin_bkg * np.heaviside(x-center, 0.5) * (x-center)
		
	index_range = find_value_index(eb, fit_range) # The data range in index of binding energy for fitting
	index_range = np.sort(index_range)  
	x = eb[index_range[0]:index_range[1]]
	x_fit = np.linspace(x[0], x[-1], num=100, endpoint=True)

	control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
	if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )
	button = widgets.Button(description="Savefig")

	curve_data = ax1.plot( x, data[index_range[0]:index_range[1],control_hv_index.value], zorder=0, label='Data', marker='+' )
	curve_fit = ax1.plot( x_fit,  Fermi_Dirac(
								x_fit,
								fermi_edge_amplitude[control_hv_index.value], fermi_edge_center[control_hv_index.value], fermi_edge_T[control_hv_index.value], 
								fermi_edge_const_bkg[control_hv_index.value], fermi_edge_lin_bkg[control_hv_index.value]
								), zorder=1, label='Fit')  
	ax1.set_xlabel("Binding Energy (eV)")
	#ax1.set_yticks([])
	ax1.set_xlim(np.min(x), np.max(x))
	vline = ax1.axvline(0, linestyle='--', color='orange')
	ax1.legend(loc=2)
	title_text1 = ax1.set_title("hv = "+str(round(hv[control_hv_index.value],2))+" eV")
	
	ax2.plot(hv, fermi_edge_center, zorder=0)
	scatter = ax2.scatter(hv[control_hv_index.value], fermi_edge_center[control_hv_index.value], color='r', zorder=1)
	ax2.set_xlabel("hv (eV)")
	ax2.set_ylabel("Offset (eV)")
	title_text2 = ax2.set_title('Offset = '+str(round(fermi_edge_center[control_hv_index.value],4))+ ' eV' )
	
	ax3.plot(hv, fermi_edge_T, zorder=0)
	scatter2 = ax3.scatter(hv[control_hv_index.value], fermi_edge_T[control_hv_index.value], color='r', zorder=1)
	ax3.set_xlabel("hv (eV)")
	ax3.set_ylabel(r"$T$ (K)")
	title_text3 = ax3.set_title(r'$T$ = '+str(round(fermi_edge_T[control_hv_index.value],2))+ ' K' )

	def update_plot(change=None):
		control_index = control_hv_index.value
		curve_data[0].set_ydata(data[index_range[0]:index_range[1],control_index])
		curve_fit[0].set_ydata(Fermi_Dirac(
								x_fit,
								fermi_edge_amplitude[control_hv_index.value], fermi_edge_center[control_hv_index.value], fermi_edge_T[control_hv_index.value], 
								fermi_edge_const_bkg[control_hv_index.value], fermi_edge_lin_bkg[control_hv_index.value]
								))  
		ax1.relim()
		ax1.autoscale(axis='y')
		title_text1.set_text("hv = "+str(round(hv[control_index],2))+" eV")
		
		scatter.set_offsets([hv[control_index], fermi_edge_center[control_index]])
		title_text2.set_text('Offset = '+str(round(fermi_edge_center[control_index],4))+ ' eV' )
		
		scatter2.set_offsets([hv[control_index], fermi_edge_T[control_index]])
		title_text3.set_text(r'$T$ = '+str(round(fermi_edge_T[control_index],2))+ ' K' )
  
		fig.canvas.draw_idle()

	def update_axvline(change):
		vline.set_visible(change['new'])
		fig.canvas.draw_idle()
 
	def on_button_clicked(b):
		fig.savefig(output_filepath + sample_name[1]+'_Fermi_edge_fit_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
	button.on_click(on_button_clicked)

	control_hv_index.observe(update_plot, names='value')
	if_axvline.observe(update_axvline, names='value')

	output_widget = widgets.Output()
	ui_controls = widgets.HBox([control_hv_index, if_axvline, button])
	ui = widgets.VBox([ui_controls, output_widget])

	display(ui)

	fig.canvas.header_visible = False
	#fig.canvas.toolbar_visible = False
	#fig.canvas.footer_visible = False
	
	return fig, ax1, ax2, ax3
		
########################################################################################################################

def kp_align(data, x, y, z, sample_name, output_filepath, x_name='Y', y_name='Eb', z_name='hv'):
	fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(8,4),
	layout='constrained')

	y_index_0 = np.abs(y - 0).argmin()
	v_min1, v_max1 = np.nanmin(data, axis=(0,1)), np.nanmax(data, axis=(0,1))
	v_min2, v_max2 = np.nanmin(data, axis=(0,2)), np.nanmax(data, axis=(0,2))

	control_index = widgets.IntSlider( min=0, max=len(z)-1, step=1, value=0, description=z_name + ' Index', continuous_update=True )
	control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
	if_axhline = widgets.Checkbox( value=True, description='Fermi Edge', layout=widgets.Layout(width='10%'), indent=False )
	
	control_bar_len = widgets.FloatSlider( min=0, max=max(x), step=0.1, value=5, description=r'Bar len', continuous_update=True )
	control_bar_center = widgets.FloatSlider( min=-5, max=5, step=0.1, value=0, description=r'Offset', continuous_update=True )
	
	control_Eb_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=y_index_0, description=y_name + ' Index', continuous_update=True )

	button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False )
	
	mesh1 = ax1.pcolormesh( x, y, np.transpose(data[:,:,control_index.value]), cmap=cmp, zorder=0 )
	mesh2 = ax2.pcolormesh( z, x, data[:,control_Eb_index.value,:], cmap=cmp, zorder=0 )

	hline1 = ax1.axhline(y=y[control_Eb_index.value], linestyle='--', linewidth=1, zorder=1, color='r')
	bar = ax1.plot([control_bar_center.value-control_bar_len.value/2.0, control_bar_center.value + control_bar_len.value/2.0],[0, 0], color='r')
	
	center_line1 = ax1.axvline(control_bar_center.value, color='r', linestyle='--')
	center_line2 = ax2.axhline(control_bar_center.value, color='r', linestyle='--', zorder=1)
	
	hv_line = ax2.axvline(z[control_index.value], color='r', linestyle='--', zorder=1)
	
	ax1.set_ylabel('Binding Energy (eV)')

	if z_name == 'hv':
		title_text1 = ax1.set_title( "hv = "+str(round(z[control_index.value],2)) + ' eV' )
	elif z_name == 'kz':
		title_text1 = ax1.set_title( "kz = "+str(round(z[control_index.value],2)) + r' $\AA^{-1}$' ) 
	else:
		print( 'z_name error' )
		raise

	title_text2 = ax2.set_title( "Eb = "+str(round(y[control_Eb_index.value],2)) + ' eV' )
	
	if x_name == 'Y':
		ax1.set_xlabel('Y (Deg)')
	elif x_name == 'kp':
		ax1.set_xlabel(r'$k_\parallel$ ($\AA^{-1}$)')
	else:
		print( 'x_name error' )
		raise
	ax1.invert_yaxis()
	ax1.set_ylim(top=-0.15)

	def update_mesh1(change=None):
		idx = control_index.value
		gamma = control_gamma.value

		mesh1.set_array( np.transpose(data[:,:,idx]) )
		mesh1.set_norm( colors.PowerNorm(gamma=gamma, vmin=v_min1[idx], vmax=v_max1[idx]) )
		hv_line.set_xdata([z[idx]])
		
		if z_name == 'hv':
			title_text1.set_text( "hv = "+str(round(z[idx],2)) + ' eV' )
		elif z_name == 'kz':
			title_text1.set_text( "kz = "+str(round(z[idx],2)) + r' $\AA^{-1}$' ) 
		else:
			print( 'z_name error' )
			raise
		
		fig.canvas.draw_idle()

	def update_mesh2(change=None):
		idx = control_Eb_index.value
		gamma = control_gamma.value

		mesh2.set_array( data[:,idx,:] )
		mesh2.set_norm( colors.PowerNorm(gamma=gamma, vmin=v_min2[idx], vmax=v_max2[idx]) )

		title_text2.set_text( "Eb = "+str(round(y[idx],2)) + ' eV' )

		hline1.set_ydata([y[idx]])
		
		fig.canvas.draw_idle()

	def update_gamma(change=None):
		idx = control_index.value
		idx2 = control_Eb_index.value
		gamma = control_gamma.value

		mesh1.set_norm( colors.PowerNorm(gamma=gamma, vmin=v_min1[idx], vmax=v_max1[idx]) )
		mesh2.set_norm( colors.PowerNorm(gamma=gamma, vmin=v_min2[idx2], vmax=v_max2[idx2]) )
  
		fig.canvas.draw_idle()

	def update_axhline(change):
		hline1.set_visible(change['new'])
		fig.canvas.draw_idle()

	def update_bar(change=None):
		bar[0].set_xdata([control_bar_center.value-control_bar_len.value/2.0, control_bar_center.value + control_bar_len.value/2.0])
		center_line1.set_xdata([control_bar_center.value])
		center_line2.set_ydata([control_bar_center.value])
		fig.canvas.draw_idle()

	def on_button_clicked(b):
		if z_name == 'hv':
			fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_Eb_hv_'+str(z[control_index.value])+'eV.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
		elif z_name == 'kz':
			fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_Eb_kz_'+str(round(z[control_index],2))+'A^-1.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
		else:
			print( 'z_name error' )
			raise
	button.on_click(on_button_clicked)
	
	control_index.observe(update_mesh1, names='value')
	control_Eb_index.observe(update_mesh2, names='value')
	control_gamma.observe(update_gamma, names='value')
	if_axhline.observe(update_axhline, names='value')
	control_bar_len.observe(update_bar, names='value')
	control_bar_center.observe(update_bar, names='value')

	output_widget = widgets.Output()
	ui1 = widgets.HBox( [control_index, control_gamma, if_axhline, button] )
	ui2 = widgets.HBox( [control_Eb_index, control_bar_len, control_bar_center] )
	ui = widgets.VBox( [ui1, ui2, output_widget] )
	
	display(ui)
	
	fig.canvas.header_visible = False
	#fig.canvas.toolbar_visible = False
	#fig.canvas.footer_visible = False
	
	return fig, ax1, ax2

########################################################################################################################  
	   
def Tune_V0(data, x, y, hv, BZ_len, x_name, z_name, sample_name, output_filepath, E_work):
	fig, ax = plt.subplots(1, 1, figsize=(8,4), layout='constrained')

	y_index_0 = np.abs(y - 0).argmin()
	v_min, v_max = np.nanmin(data, axis=(0,2)), np.nanmax(data, axis=(0,2))

	control_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=y_index_0, description='Eb Index', continuous_update=True )
	control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
	control_V0 = widgets.FloatSlider( min=max(-30, float(E_work - np.min(hv)) ), max=30, step=0.5, value=0, description=r'V0', continuous_update=True )

	button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False)
	
	z = 1/hbar * np.sqrt( 2*me* (hv - E_work + control_V0.value) ) 
	
	mesh = ax.pcolormesh( z, x, data[:,control_index.value,:], cmap=cmp, zorder=0 )
	
	hv_cal = hbar**2 * np.arange( 0, BZ_len * 30, BZ_len )**2 / (2*me) + E_work - control_V0.value
	hv_cal = [str(round(x,2)) for x in hv_cal]
	
	x_ticks = np.arange( 0, BZ_len * 30, BZ_len )
	x_ticks = [str(round(x,2)) for x in x_ticks]
	
	new_labels = [ '\n\n'.join(x) for x in zip( x_ticks, hv_cal  ) ]
	
	ax.set_xticks(np.arange( 0, BZ_len * 30, BZ_len ), new_labels)
	title_text = ax.set_title(f"Binding Energy = {y[control_index.value]:.2f} eV")
	
	ax.grid(True)
	ax.set_xlim([min(z)-0.5*BZ_len, max(z)+0.5*BZ_len])
	
	
	if x_name == 'Y':
		ax.set_ylabel('Y (Deg)')
	elif x_name == 'kp':
		ax.set_ylabel(r'$k_\parallel$ ($\AA^{-1}$)')
	else:
		print( 'y_name error' )
		raise
	
	if z_name == 'hv':
		ax.set_xlabel('hv (eV)')
	elif z_name == 'kz':
		ax.set_xlabel(r'$k_z$ ($\AA^{-1}$)')
	else:
		print( 'z_name error' )
		raise

		
	def update_mesh(change=None):  
		nonlocal mesh
		current_slice = data[:, control_index.value, :]
		mesh.set_array(current_slice.flatten())
	
		mesh.set_norm(colors.PowerNorm(gamma=control_gamma.value, 
								   vmin=v_min[control_index.value], 
								   vmax=v_max[control_index.value]))
	
		title_text.set_text(f"Binding Energy = {y[control_index.value]:.2f} eV")
	
		fig.canvas.draw_idle()

	def update_gamma(change=None):
		nonlocal mesh
		mesh.set_norm(colors.PowerNorm(gamma=control_gamma.value, 
								   vmin=v_min[control_index.value], 
								   vmax=v_max[control_index.value]))
		fig.canvas.draw_idle()

	def update_V0(change=None):
		nonlocal mesh
		z = 1/hbar * np.sqrt( 2*me* (hv - E_work + control_V0.value) ) 

		mesh.remove()
		mesh = ax.pcolormesh( z, x, data[:,control_index.value,:], cmap=cmp, zorder=0 )

		hv_cal = hbar**2 * np.arange( 0, BZ_len * 30, BZ_len )**2 / (2*me) + E_work - control_V0.value
		new_labels = [f"{xt}\n\n{hv:.2f}" for xt, hv in zip(x_ticks, hv_cal)]
		ax.set_xticks(np.arange( 0, BZ_len * 30, BZ_len ), new_labels)
		
		ax.set_xlim([min(z)-0.5*BZ_len, max(z)+0.5*BZ_len])
	
		fig.canvas.draw_idle()
	
	def on_button_clicked(b):
		fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_'+z_name+'_Eb_'+str(round(y[control_index.value],2))+'eV.png', format='png', transparent=True,
					pad_inches = 0, bbox_inches='tight', dpi=300)
	button.on_click(on_button_clicked)    

	control_index.observe(update_mesh, names='value')
	control_gamma.observe(update_gamma, names='value')
	control_V0.observe(update_V0, names='value')

	output_widget = widgets.Output()
	ui_controls = widgets.HBox([control_V0, control_index, control_gamma, button] )
	ui = widgets.VBox([ui_controls, output_widget])
	
	display(ui)
	
	fig.canvas.header_visible = False
	#fig.canvas.toolbar_visible = False
	#fig.canvas.footer_visible = False
	
	return fig, ax   