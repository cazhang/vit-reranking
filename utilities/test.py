import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

def test_einsum():
	anchor = torch.randn((2, 10))
	fb = torch.randn((100,2,20))
	N,_,R=fb.size()
	sim1 = torch.einsum('cm,ncs->nsm', anchor, fb)
	sim2 = torch.einsum('cm,ncs->nms', anchor, fb)
	sim2_t = sim2.permute(0,2,1)
	assert torch.equal(sim1, sim2_t)==True
	print(sim1.size(), sim2.size())

def test_patch_rect():
	img_size = 224
	grid_size = 4
	patch_scale = img_size // grid_size
	img = np.arange(img_size**2).reshape((img_size,img_size))
	max_ind = 0
	fig, ax = plt.subplots(4, 4, figsize=(10, 10))
	ax=ax.flat
	for max_ind in range(grid_size**2):
		max_src_coord = np.unravel_index(max_ind, (grid_size, grid_size))

		max_src_x, max_src_y = max_src_coord[1] * patch_scale, max_src_coord[0] * patch_scale

		max_src_rect = patches.Rectangle((max_src_x, max_src_y - 1), patch_scale, patch_scale, linewidth=1, edgecolor='r', fill=True)

		ax[max_ind].imshow(img)
		ax[max_ind].add_patch(max_src_rect)

	fig.savefig('test_rect.png')
	plt.close(fig)

def test_logger():
	import logging
	##loging
	# for handler in logging.root.handlers[:]:
	# 	logging.root.removeHandler(handler)

	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(message)s',
						datefmt='%a, %d %b %Y %H:%M:%S',
						filename='log.txt',
						filemode='w')

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	# add the handler to the root logger
	logging.getLogger().addHandler(console)
	logging.info("\nParameters:")

	for i in range(10):
		logging.info(i)

	logging.info("end!")

def test_kwargs(name, title, **kwargs):
	print(f'name is {name}, title is {title}')
	for key, value in kwargs.items():
		print(f'{key} is {value}')
	if 'gender' in kwargs.keys():
		print(kwargs['gender'])

def test_einsum_order(B=1, C=10, H=2, W=2):
	x = torch.randn(B,C,H,W)
	y = torch.randn(B,C,H,W)
	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	x = x.view(B, C, -1, 1)
	y = y.view(B, C, 1, -1)

	x = x.view(B, C, -1)
	y = y.view(B, C, -1)

	sim1 = torch.einsum('bcs, bcm->bsm', x, y).contiguous()
	sim2 = torch.einsum('bcs, bcm->bms', x, y).contiguous()
	print(sim1[0].shape)

	sim = []
	for r in range(H*W):
		for c in range(H*W):
			sim.append((x[:,:,r]*(y[:,:,c])).sum())
	print(sim)

	import ipdb; ipdb.set_trace()

def run_test_kwargs():
	test_kwargs(name='chao', title='dr', gender='male')

def test_faiss(pool_size = 3, k=3):
	import faiss
	faiss_index = faiss.IndexFlatL2(pool_size)

	db = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=np.float32)
	query = np.array([[0,0,0]], dtype=np.float32)
	faiss_index.add(db)
	dists, preds = faiss_index.search(query, k)
	print(dists)
	print(preds)
	
def test_triplets():
	import random
	num_query = 3000
	arr = np.arange(num_query)
	arr = random.choices(arr, k=len(arr))
	# calculate the subcache indices
	subcache_indices = np.array_split(arr, 3)
	
	triplets = {}
	for sub in range(3):
		triplets[sub] = np.random.randint(0, 100, (1000, 5))
	
	tmp_triplets = []
	for sub in range(3):
		tmp_triplets.extend(triplets[sub])
	new_triplets = {}
	arr = random.choices(arr, k=len(arr))
	# calculate the subcache indices
	new_subcache_indices = np.array_split(arr, 3)
	for sub in range(3):
		new_triplets[sub] = [tmp_triplets[i] for i in new_subcache_indices[sub]]
		
	breakpoint()


test_triplets()