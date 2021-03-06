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
detector = 'L'
files = get_files(detector)

params_path = Path(git_path + '/shared/injection_params')

inj_df = pd.read_csv(join(params_path, 'soft_inj_time.csv'), usecols=[detector])
# inj_df = pd.read_csv(join(params_path, 'soft_inj_time_no_constraint.csv'), usecols=[detector])

start_time = time.time()

### get chunks that contain the injection times

times_par = []
inj_times = []
chunks = []

for t_inj in inj_df[detector][:15]: # only take the first 15 for the initial constrained injections
	segment = find_segment(t_inj, segment_list)
	chunk_list = get_chunks(segment[0], segment[1], Tc=Tc, To=To)
	chunk = find_segment(t_inj, chunk_list)
	chunks.append(find_segment(t_inj, chunk_list))
	inj_times.append(t_inj)
	times_par.append((chunk[0], chunk[1], t_inj))

# pool = mp.Pool(mp.cpu_count() - 1)
pool = mp.Pool(15)

D_kpc = 10
D = D_kpc * 3.086e+21 # cm

inj_type = 'ccsn'
inj_params = None

ccsn_paper = 'abdikamalov'
wfs_path = Path(git_path + '/shared/ccsn_wfs/' + ccsn_paper)
ccsn_files = [f for f in sorted(listdir(wfs_path)) if isfile(join(wfs_path, f))]

h_rss = []

for ccsn_file in ccsn_files[20:40]:
	data = [i.strip().split() for i in open(join(wfs_path, ccsn_file)).readlines()]
	sim_times = np.asarray([float(dat[0]) for dat in data])
	hp = np.asarray([float(dat[1]) for dat in data]) / D

	dt = sim_times[1] - sim_times[0]
	hp = TimeSeries(hp, t0=sim_times[0], dt=dt)

	hp = hp.resample(rate=fw, ftype = 'iir', n=20) # downsample to working frequency fw
	hp = hp.highpass(frequency=11, filtfilt=True) # filter out frequencies below 20Hz
	window = scisig.tukey(M=len(hp), alpha=0.08, sym=True)
	hp = hp * window

	hp = hp.pad(int((fw * Tc - len(hp)) / 2))

	h_rss.append(np.sqrt(hp.dot(hp) * hp.dt).value)

	results = pool.starmap(load_inject_condition, [(t[0], t[1], t[2], inj_type, inj_params, local, Tc, To, fw, 
						   'tukey', detector, qtrans, qsplit, dT, hp) for t in times_par])

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

	fname = 'injected-' + ccsn_paper + '-' + '.'.join(ccsn_file.split('.')[2:] + ['hdf5'])
	with h5py.File(join(data_path, fname), 'w') as f:
		f.create_dataset('x', data=x)
		f.create_dataset('times', data=times)

	print(fname)
	print(x.shape)
	print(times.shape)

pool.close()
pool.join()

h_rss_df = pd.DataFrame({'h_rss': h_rss})
h_rss_df.to_csv(join(params_path, ccsn_paper + '_h_rss_csv.csv'))

print(detector)
print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))



