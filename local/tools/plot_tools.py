#!/usr/bin/python
"""
tools for generating interactive plots of the map space (also with spectrograms)
"""

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
from bokeh.io import show
from colorcet import glasbey_light
from holoviews.streams import Selection1D
hv.extension('bokeh')

def gen_markers(labels):
	"""Generate markers and sizes.
	"""

	stylemap = {'1080Lines': 'circle', '1400Ripples': 'hex', 'Air_Compressor': 'diamond', 'Blip': 'square', 'Chirp': 'inverted_triangle', 
				'Extremely_Loud': 'triangle', 'Koi_Fish': 'circle', 'Light_Modulation': 'hex', 'Low_Frequency_Burst': 'diamond', 
				'Low_Frequency_Lines': 'square', 'No_Glitch': 'inverted_triangle', 'None_of_the_Above': 'triangle', 'Paired_Doves': 'circle', 
				'Power_Line': 'hex', 'Repeating_Blips': 'diamond', 'Scattered_Light': 'square', 'Scratchy': 'inverted_triangle', 
				'Tomte': 'triangle', 'Unlabeled': 'circle', 'Wandering_Line': 'hex', 'Whistle': 'diamond', 'H_detector': 'hex', 
				'L_detector': 'square'}

	sizemap = {'1080Lines': 8, '1400Ripples': 8, 'Air_Compressor': 11, 'Blip': 8, 'Chirp': 9, 
			   'Extremely_Loud': 9, 'Koi_Fish': 8, 'Light_Modulation': 8, 'Low_Frequency_Burst': 11, 
			   'Low_Frequency_Lines': 8, 'No_Glitch': 9, 'None_of_the_Above': 9, 'Paired_Doves': 8, 
			   'Power_Line': 8, 'Repeating_Blips': 11, 'Scattered_Light': 8, 'Scratchy': 9, 
			   'Tomte': 9, 'Unlabeled': 8, 'Wandering_Line': 8, 'Whistle': 11, 'H_detector': 8, 'L_detector': 8}

	markers = []
	sizes = []
	for label in (labels):
		if label not in stylemap.keys():
			markers.append(stylemap['Unlabeled'])
			sizes.append(sizemap['Unlabeled'])
			continue

		markers.append(stylemap[label])
		sizes.append(sizemap[label])

	return markers, sizes

def gen_cmap(labels):
	color_dict = {'1080Lines': '#d60000', '1400Ripples': '#018700', 'Air_Compressor': '#b500ff', 'Blip': '#05acc6', 'Chirp': '#97ff00', 
				  'Extremely_Loud': '#ffa52f', 'Koi_Fish': '#ff8ec8', 'Light_Modulation': '#79525e', 'Low_Frequency_Burst': '#00fdcf', 
				  'Low_Frequency_Lines': '#afa5ff', 'No_Glitch': '#93ac83', 'None_of_the_Above': '#9a6900', 'Paired_Doves': '#366962', 
				  'Power_Line': '#d3008c', 'Repeating_Blips': '#fdf490', 'Scattered_Light': '#c86e66', 'Scratchy': '#9ee2ff', 
				  'Tomte': '#00c846', 'Wandering_Line': '#a877ac', 'Whistle': '#b8ba01', 'Unlabeled': '#f4bfb1', 
				  'H_detector': '#ff28fd', 'L_detector': '#f2cdff', 'BH_Merger': '#f4bfb1'} # 'H_detector': '#ff28fd', '1080Lines': '#d60000'
	
	cmap = []
	i = 0
	for label in pd.unique(labels): #labels: #np.unique(labels): #color_dict.keys():
		if label in color_dict.keys():
			cmap.append(color_dict[label])
		elif label == 'noise':
			cmap.append(color_dict[list(color_dict.keys())[-1]])
		else:
			cmap.append(color_dict[list(color_dict.keys())[i]])
			i += 1 % len(color_dict.keys())
	
	return cmap

def gen_colors(labels):
	color_dict = {'1080Lines': '#d60000', '1400Ripples': '#018700', 'Air_Compressor': '#b500ff', 'Blip': '#05acc6', 'Chirp': '#97ff00', 
				  'Extremely_Loud': '#ffa52f', 'Koi_Fish': '#ff8ec8', 'Light_Modulation': '#79525e', 'Low_Frequency_Burst': '#00fdcf', 
				  'Low_Frequency_Lines': '#afa5ff', 'No_Glitch': '#93ac83', 'None_of_the_Above': '#9a6900', 'Paired_Doves': '#366962', 
				  'Power_Line': '#d3008c', 'Repeating_Blips': '#fdf490', 'Scattered_Light': '#c86e66', 'Scratchy': '#9ee2ff', 
				  'Tomte': '#00c846', 'Wandering_Line': '#a877ac', 'Whistle': '#b8ba01', 'Unlabeled': '#f4bfb1', 
				  'H_detector': '#ff28fd', 'L_detector': '#f2cdff', 'BH_Merger': '#f4bfb1'} # 'H_detector': '#ff28fd', '1080Lines': '#d60000'

	colors = []
	for label in labels:
		if label not in color_dict.keys():
			colors.append(color_dict['Unlabeled'])
			continue

		colors.append(color_dict[label])

	return colors

def set_active_tool(plot, element):
	"""Set default active inspection to None, that way hovertool will be not active by default.
	"""

	plot.state.toolbar.active_inspect = None
	return

def interactive_plot(features, y, times, index=None, title='', alpha=0.8, xlabel='dimension 1', ylabel='dimension 2'):
	"""Create interactive points plot, of the embedded features.
	"""

	df = pd.DataFrame(features, columns=['feature_1', 'feature_2'])
	df['times'] = times
	df['y'] = y
	markers, sizes = gen_markers(y)
	df['marker'] = markers
	df['size'] = sizes

	if index is None:
		df['index'] = np.arange(len(times))
	else:
		df['index'] = index

	cmap = gen_cmap(y)

	points = hv.Points(df)
	points.opts(width=800, height=550, labelled=[])

	tooltips = [('Label', '@y'), ('Time', '@times{f}'), ('Index', '@index')]
	hover = HoverTool(tooltips=tooltips)

	points.opts(tools=[hover, 'tap', 'box_select'], size='size', color='y', 
				cmap=cmap, line_color='black', padding=0.1, 
				marker='marker', alpha=alpha, show_grid=True, 
				title=title, xlabel=xlabel, ylabel=ylabel, 
				nonselection_alpha=0.2, nonselection_line_color='gray', 
				legend_position='left', 
				hooks=[set_active_tool])
	return points

def interactive_with_image(x, y, features, times, index=None, title=''):
	"""Create layout of interactive points plot and corresponding q-transform image.
	"""

	def selected_index(index):
		"""Create image corresponding to selected image in point plot.
		"""

		if index:
			selected = hv.RGB(x[index[0]]).opts(width=500, height=500)
			label = '%s, %f, %d, %d selected' % (y[index[0]], times[index[0]], index[0], len(index))
		else:
			selected = hv.RGB(x[index]).opts(width=500, height=500)
			label = 'No selection'
		return selected.relabel(label).opts(labelled=[])

	points = interactive_plot(features, y, times, index, title=title)

	selection = Selection1D(source=points)

	xtick_pos = np.linspace(-0.5, 0.5, 11)
	xtick_label = np.linspace(0, 2, 11)
	xticks = []
	for a, b in zip(xtick_pos, xtick_label):
		xticks.append((a, "{:.1f}".format(b)))
	ytick_pos = np.linspace(-0.5, 0.5, 9)
	ytick_label = np.logspace(4, 11, 8, base=2)
	yticks = [(ytick_pos[0], '')]
	for a, b in zip(ytick_pos[1:], ytick_label):
		yticks.append((a, str(int(b))))

	dmap = hv.DynamicMap(selected_index, streams=[selection]).opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), 
																   xticks=xticks, yticks=yticks, 
																   xlabel='time [s]', ylabel='frequency [Hz]')
	layout = (points.opts(toolbar='above') + dmap.opts(toolbar=None)).opts(merge_tools=False)
	return layout

def get_random_label_examples(x, features, y, times, num_examples=1):
	"""Get random labeled examples.
	"""

	x_examples = []
	features_examples = []
	y_examples = []
	times_examples = []

	idxs = np.unique(y, return_index=True)[1]
	labels_unordered = [y[idx] for idx in sorted(idxs)]

	for label in (labels_unordered):
		idx = [j for j, val in enumerate(y) if val == label]
		if not idx:
			continue

		idx = np.random.choice(idx, size=num_examples, replace=False)#, random_state=0)
		for k in idx:
			x_examples.append(x[k])
			features_examples.append(features[k])
			y_examples.append(y[k])
			times_examples.append(times[k])


	x_examples = np.asarray(x_examples)
	features_examples = np.asarray(features_examples)
	y_examples = np.asarray(y_examples)
	times_examples = np.asarray(times_examples)

	return x_examples, features_examples, y_examples, times_examples

def plot_both_dets(tomap, index=None, plot_ex=True, title='', alpha=0.8, xlabel='dimension 1', ylabel='dimension 2'):
	"""Create plot with images from both detectors.
	"""

	if plot_ex:
		features = np.append(tomap['ex']['umap'], tomap['H']['umap'], axis=0)
		y = np.append(tomap['ex']['y'], tomap['H']['y'], axis=0)
		times = np.append(tomap['ex']['times'], tomap['H']['times'], axis=0)
	else:
		features = tomap['H']['umap']
		y = tomap['H']['y']
		times = tomap['H']['times']

	# features = np.append(tomap['ex']['umap'], tomap['H']['umap'], axis=0)
	features = np.append(features, tomap['L']['umap'], axis=0)

	# y = np.append(tomap['ex']['y'], tomap['H']['y'], axis=0)
	y = np.append(y, tomap['L']['y'], axis=0)

	# times = np.append(tomap['ex']['times'], tomap['H']['times'], axis=0)
	times = np.append(times, tomap['L']['times'], axis=0)

	if index is not None:
		index = np.append(index, index, axis=0)

	width = 400
	height = 400

	def H_selected_index(index):
		"""Create image of H detector corresponding to selected point.
		"""

		x_img = tomap['ex']['x']
		y_img = tomap['ex']['y']
		times_img = tomap['ex']['times']

		if index:
			if plot_ex:
				ex_len = tomap['ex']['x'].shape[0]
			else:
				ex_len = 0

			# ex_len = tomap['ex']['x'].shape[0]
			H_len = tomap['H']['x'].shape[0]

			if index[0] < ex_len:
				selected = hv.RGB(x_img[index[0]]).opts(width=width, height=height)
				label = '%s, %f, %d selected' % (y_img[index[0]], times_img[index[0]], len(index))
			else:
				x_img = tomap['H']['x']
				y_img = tomap['H']['y']
				times_img = tomap['H']['times']

				if index[0] < ex_len + H_len:
					selected = hv.RGB(x_img[index[0] - ex_len]).opts(width=width, height=height)
					label = 'H1 (H1): %s, %f, %d selected' % (y_img[index[0] - ex_len], times_img[index[0] - ex_len], len(index))
				else:
					selected = hv.RGB(x_img[index[0] - (ex_len + H_len)]).opts(width=width, height=height)
					label = 'H1 (L1): %s, %f, %d selected' % (y_img[index[0] - (ex_len + H_len)], times_img[index[0] - (ex_len + H_len)], len(index))
		else:
			selected = hv.RGB(x_img[index]).opts(width=width, height=height)
			label = 'No selection'
		return selected.relabel(label).opts(labelled=[])

	def L_selected_index(index):
		"""Create image of L detector corresponding to selected point.
		"""

		x_img = tomap['ex']['x']
		y_img = tomap['ex']['y']
		times_img = tomap['ex']['times']

		if index:
			if plot_ex:
				ex_len = tomap['ex']['x'].shape[0]
			else:
				ex_len = 0

			# ex_len = tomap['ex']['x'].shape[0]
			H_len = tomap['H']['x'].shape[0]

			if index[0] < ex_len:
				selected = hv.RGB(x_img[index]).opts(width=width, height=height)
				label = 'No selection'
			else:
				x_img = tomap['L']['x']
				y_img = tomap['L']['y']
				times_img = tomap['L']['times']

				if index[0] < ex_len + H_len:
					selected = hv.RGB(x_img[index[0] - ex_len]).opts(width=width, height=height)
					label = 'L1 (H1): %s, %f, %d selected' % (y_img[index[0] - ex_len], times_img[index[0] - ex_len], len(index))
				else:
					selected = hv.RGB(x_img[index[0] - (ex_len + H_len)]).opts(width=width, height=height)
					label = 'L1 (L1): %s, %f, %d selected' % (y_img[index[0] - (ex_len + H_len)], times_img[index[0] - (ex_len + H_len)], len(index))
		else:
			selected = hv.RGB(x_img[index]).opts(width=width, height=height)
			label = 'No selection'
		return selected.relabel(label).opts(labelled=[])

	points = interactive_plot(features, y, times, index, title=title, alpha=alpha, xlabel=xlabel, ylabel=ylabel)

	selection = Selection1D(source=points)

	xtick_pos = np.linspace(-0.5, 0.5, 11)
	xtick_label = np.linspace(0, 2, 11)
	xticks = []
	for a, b in zip(xtick_pos, xtick_label):
		xticks.append((a, "{:.1f}".format(b)))
	# ytick_pos = np.linspace(-0.5, 0.5, 9)
	# ytick_label = np.logspace(4, 11, 8, base=2)
	# yticks = [(ytick_pos[0], '')]
	# for a, b in zip(ytick_pos[1:], ytick_label):
	# 	yticks.append((a, str(int(b))))

	ytick_pos = np.linspace(-0.5, 0.5, tomap['H']['x'].shape[2])
	ytick_label = np.logspace(4, 11, 8, base=2)
	f_ax = np.logspace(np.log10(10), np.log10(2048), tomap['H']['x'].shape[2])
	# pos = [ytick_pos[0]]
	pos = []
	for tick in ytick_label:
		ax_idx = (min(range(len(f_ax)), key=lambda i: abs(f_ax[i] - tick)))
		pos.append(ytick_pos[ax_idx])

	yticks = [(ytick_pos[0], '')]
	for a, b in zip(pos, ytick_label):
		yticks.append((a, str(int(b))))


	H_dmap = hv.DynamicMap(H_selected_index, streams=[selection]).opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
																	   xticks=xticks, yticks=yticks, 
																	   xlabel='time [s]', ylabel='frequency [Hz]')
	L_dmap = hv.DynamicMap(L_selected_index, streams=[selection]).opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
																	   xticks=xticks, yticks=yticks, 
																	   xlabel='time [s]', ylabel='frequency [Hz]')

	layout = (points.opts(toolbar='below') + hv.Empty() 
		+ H_dmap.opts(frame_width=width, frame_height=height) 
		+ L_dmap.opts(frame_width=width, frame_height=height)).opts(merge_tools=False).cols(2)
	return layout

def create_contour_curves(contours, alpha=1):
	"""Create contour curves given contour vertices.
	"""

	curves = None
	cmap = gen_cmap(range(len(contours)))
	i = 0
	for contour in contours:
		if curves is None:
			curves = hv.Curve(contour).opts(alpha=alpha, color=cmap[i])
		else:
			curves = curves * hv.Curve(contour).opts(alpha=alpha, color=cmap[i])
		i += 1
	return curves

def interactive_cmap(features, y, times, c, title=''):
	"""Create interactive points plot, of the embedded features.
	"""

	df = pd.DataFrame(features, columns=['feature_1', 'feature_2'])
	df['times'] = times
	df['y'] = y
	df['c'] = c
	markers, sizes = gen_markers(y)
	# df['marker'] = markers
	# df['size'] = sizes

	points = hv.Points(df)
	points.opts(width=800, height=550, labelled=[])

	tooltips = [('Label', '@y'), ('Time', '@times{f}')]
	hover = HoverTool(tooltips=tooltips)

	points.opts(tools=[hover, 'tap', 'box_select'], size=8, color='c', 
				cmap='Viridis', line_color='black', padding=0.1, 
				alpha=0.8, show_grid=True, 
				title=title, 
				nonselection_alpha=0.2, nonselection_line_color='gray', 
				legend_position='left', 
				hooks=[set_active_tool], colorbar=True)
	return points
