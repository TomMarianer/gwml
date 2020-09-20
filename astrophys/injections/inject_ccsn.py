#!/usr/bin/python3

import git
from os import listdir
from os.path import isfile, join, dirname, realpath

def get_git_root(path):
	"""Get git root path
	"""
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root

file_path = dirname(realpath(__file__))
git_path = get_git_root(file_path)

import sys
sys.path.append(git_path + '/astrophys/tools')
from tools_gs_par import *
from params import *

sys.path.append(git_path + '/shared')
from inject_tools import *

local = True
qtrans = True
qsplit = True
save = True

segment_list = get_segment_list('BOTH')
detector = 'H'
files = get_files(detector)

params_path = Path(git_path + '/shared/injection_params')

inj_df = pd.read_csv(join(params_path, 'soft_inj_time.csv'), usecols=['H']) #usecols=[detector])
# inj_df = pd.read_csv(join(params_path, 'soft_inj_time_no_constraint.csv'), usecols=['H']) #usecols=[detector])

sky_loc = pd.read_csv(join(params_path, 'sky_loc_csv.csv'), usecols=['ra', 'dec'])

start_time = time.time()

### get chunks that contain the injection times

times_par = []
inj_times = []
chunks = []

# for i, t_inj in enumerate(inj_df[detector][:15]): # only take the first 15 for the initial constrained injections
for i, t_inj in enumerate(inj_df['H'][:15]): # only use H injection times, calculate actual delay according to sky_loc (in inject_tools)
	segment = find_segment(t_inj, segment_list)
	chunk_list = get_chunks(segment[0], segment[1], Tc=Tc, To=To)
	chunk = find_segment(t_inj, chunk_list)
	chunks.append(find_segment(t_inj, chunk_list))
	inj_times.append(t_inj)
	times_par.append((chunk[0], chunk[1], t_inj, sky_loc['ra'][i], sky_loc['dec'][i], sky_loc['pol'][i]))

# pool = mp.Pool(mp.cpu_count() - 1)
pool = mp.Pool(15)

inj_type = 'ccsn'
inj_params = None

ccsn_paper = 'abdikamalov'
# ccsn_paper = 'andersen'
# ccsn_paper = 'radice'
wfs_path = Path(git_path + '/shared/ccsn_wfs/' + ccsn_paper)
ccsn_files = [f for f in sorted(listdir(wfs_path)) if isfile(join(wfs_path, f))]

h_rss = []

for ccsn_file in ccsn_files[40:41]:
	# for D_kpc in [0.01, 0.1, 0.2, 0.5, 1, 3, 5, 7, 10]:
	# for D_kpc in [0.01, 0.1, 0.2, 0.5, 1, 3, 5, 7]:
	for D_kpc in [3, 5, 7, 10]:
		results = pool.starmap(load_inject_condition_ccsn, [(t[0], t[1], t[2], t[3], t[4], t[5], ccsn_paper, ccsn_file, 
							   D_kpc, local, Tc, To, fw, 'tukey', detector, qtrans, qsplit, dT) for t in times_par])

		x = []
		times = []
		for result in results:
			x.append(result[0])
			times.append(result[1])

		x = np.asarray(x)
		times = np.asarray(times)

		data_path = Path('/storage/fast/users/tommaria/data/conditioned_data/16KHZ/' + detector + '1/injected/ccsn/' + ccsn_paper)
		if not exists(data_path):
			makedirs(data_path)

		if ccsn_paper == 'abdikamalov':
			fname = 'injected-' + ccsn_paper + '-' + '.'.join(ccsn_file.split('.')[2:]) + '-D-' + str(D_kpc) + '.hdf5'

		elif ccsn_paper == 'andersen':
			fname = 'injected-' + ccsn_paper + '-' + '-'.join(ccsn_file.split('.')[0].split('_')[1:]) + '-D-' + str(D_kpc) + '.hdf5'

		elif ccsn_paper == 'radice':
			fname = 'injected-' + ccsn_paper + '-' + ccsn_file.split('.')[0] + '-D-' + str(D_kpc) + '.hdf5'

		with h5py.File(join(data_path, fname), 'w') as f:
			f.create_dataset('x', data=x)
			f.create_dataset('times', data=times)

		print(fname)
		print(x.shape)
		print(times.shape)

pool.close()
pool.join()

print(detector)
print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))



