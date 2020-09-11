#!/usr/bin/python
"""
tools for plotting using matplotlib - used to generate figures for the paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_style_df(labels):
	"""Define marker, size and color dictionaries
	"""

	stylemap = {'1080Lines': 'circle', '1400Ripples': 'hex', 'Air_Compressor': 'diamond', 'Blip': 'square', 'Chirp': 'inverted_triangle', 
				'Extremely_Loud': 'triangle', 'Koi_Fish': 'circle', 'Light_Modulation': 'hex', 'Low_Frequency_Burst': 'diamond', 
				'Low_Frequency_Lines': 'square', 'No_Glitch': 'inverted_triangle', 'None_of_the_Above': 'triangle', 'Paired_Doves': 'circle', 
				'Power_Line': 'hex', 'Repeating_Blips': 'diamond', 'Scattered_Light': 'square', 'Scratchy': 'inverted_triangle', 
				'Tomte': 'triangle', 'Unlabeled': 'circle', 'Wandering_Line': 'hex', 'Whistle': 'diamond', 'H_detector': 'hex', 
				'L_detector': 'square'}

	colormap = {'1080Lines': '#d60000', '1400Ripples': '#018700', 'Air_Compressor': '#b500ff', 'Blip': '#05acc6', 'Chirp': '#97ff00', 
				'Extremely_Loud': '#ffa52f', 'Koi_Fish': '#ff8ec8', 'Light_Modulation': '#79525e', 'Low_Frequency_Burst': '#00fdcf', 
				'Low_Frequency_Lines': '#afa5ff', 'No_Glitch': '#93ac83', 'None_of_the_Above': '#9a6900', 'Paired_Doves': '#366962', 
				'Power_Line': '#d3008c', 'Repeating_Blips': '#fdf490', 'Scattered_Light': '#c86e66', 'Scratchy': '#9ee2ff', 
				'Tomte': '#00c846', 'Wandering_Line': '#a877ac', 'Whistle': '#b8ba01', 'Unlabeled': '#f4bfb1', 
				'H_detector': '#ff28fd', 'L_detector': '#f2cdff', 'BH_Merger': '#f4bfb1'}

	marker_dict = {'circle': 'o', 'diamond': 'd', 'hex': 'H', 'inverted_triangle': 'v', 'square': 's', 'triangle': '^'}
	size_dict = {'circle': 64, 'diamond': 88, 'hex': 64, 'inverted_triangle': 72, 'square': 64, 'triangle': 72}

	style_df = pd.DataFrame.from_dict(stylemap, orient='index', columns=['bokeh_marker'])
	mpl_marker = [marker_dict[marker] for marker in style_df['bokeh_marker']]
	mpl_size = [size_dict[marker] for marker in style_df['bokeh_marker']]
	colors = [colormap[idx] for idx in style_df.index]
	style_df['mpl_marker'] = mpl_marker
	style_df['size'] = mpl_size
	style_df['color'] = colors

	return style_df

def gen_mpl_map(umap, y, x_range=(-15.5, 14.5), y_range=(-17, 13), ax_size_x=9, ax_size_y=9):
	"""Generate matplotlib scatter plot
	"""

	df = pd.DataFrame(umap, columns=['umap_1', 'umap_2'])
	df['y'] = y
	style_df = gen_style_df(y)

	ax_size = 9
	
	fig, ax = plt.subplots(figsize=(ax_size_x, ax_size_y))
	for label in np.unique(df['y']):
		idx = df['y'] == label
		ax.scatter(df['umap_1'][idx], df['umap_2'][idx], c=style_df['color'][label], 
				   s=style_df['size'][label], marker=style_df['mpl_marker'][label], 
				   alpha=0.8, edgecolor='black', label=label)

	plt.xticks([])
	plt.yticks([])

	plt.xlim(x_range)
	plt.ylim(y_range)

	plt.legend(bbox_to_anchor=(0, 1), prop={'size': 15})

	return fig

def gen_mpl_map_heatmap(umap, y, score_grid, vmin, vmax, x_range=(-15.5, 14.5), y_range=(-17, 13), ax_size_x=9, ax_size_y=9, ratios=[1, 1]):
	"""Generate matplotlib figure with both scatter plot and heatmap
	"""

	df = pd.DataFrame(umap, columns=['umap_1', 'umap_2'])
	df['y'] = y
	style_df = gen_style_df(y)

	ax_size = 9
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ax_size_x, ax_size_y), gridspec_kw={'width_ratios': ratios})
	for label in np.unique(df['y']):
		idx = df['y'] == label
		ax1.scatter(df['umap_1'][idx], df['umap_2'][idx], c=style_df['color'][label], 
					s=style_df['size'][label], marker=style_df['mpl_marker'][label], 
					alpha=0.8, edgecolor='black', label=label)

	ax2.imshow(score_grid, vmin=vmin, vmax=vmax, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])

	return fig, (ax1, ax2)

