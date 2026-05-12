import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from decord import VideoReader, cpu

#%% set path
video_path = "data/scratch/fish_2025-11-27_20-16-33.mp4"

#%% read video
vr = VideoReader(video_path, ctx=cpu(0), num_threads=0)

n_frames = len(vr)
fps = vr.get_avg_fps()

iframe = vr[0]
rows, cols = iframe.shape[:2]

xmin, xmax = 600, 1400
ymin, ymax = 160, 920
h_crop = ymax - ymin
w_crop = xmax - xmin

#%% load frames
print('Loading video frames...')
sample_freq = 100

sample_i = range(0, n_frames, sample_freq)
frames = vr.get_batch(sample_i).asnumpy()

sample_buf = frames[:, ymin:ymax, xmin:xmax, 1] # Slice to (Samples, H_crop, W_crop)
sample_buf = np.moveaxis(sample_buf, 0, -1)     # Move 'Samples' to the last dimension

# Truncate to actual number of loaded frames
sample_n_frames = sample_buf.shape[2]

# Define time arrays
tsample = np.linspace(0, n_frames/fps, sample_n_frames)
t = np.linspace(0, n_frames/fps, n_frames)

# plot minimum pixel value
plt.figure()
plt.imshow(np.max(sample_buf, axis=2), cmap='viridis')
plt.title('Min intensity across samples')
plt.show()

#%% define roi mask
roi_mask = np.zeros((h_crop, w_crop), dtype=bool)
roi_mask[:,150:450] = True
roi_mask[300:,150:750] = True

#%% plot a pixel over time
fig, ax = plt.subplots(2, 2,)
ax = ax.flatten()


#%% through every pixel, fit a polynomial to correct for changing light conditions
degree = 7
coeff = np.zeros((h_crop, w_crop, degree + 1), dtype=float)

locs = [[100,100],[400,500],[559,221],[680,270]]  # [row, col] in cropped image

a = 0
print('Calculating polynomial coefficients...')
for i in tqdm(range(h_crop)):
    for j in range(w_crop):
        y = sample_buf[i, j, :].astype(float)

        # compute mask for 2-sigma filtering
        mu = np.mean(y)
        sigma = np.std(y)
        mask = np.abs(y - mu) <= 2 * sigma

        # fit polynomial using only inliers
        fit = np.polyfit(tsample[mask], y[mask], degree)
        coeff[i, j, :] = fit

        if [i,j] in locs:
            ax[a].plot(tsample, y, 'o-', label='before correction', alpha=0.7)
            yfit = np.polyval(coeff[i, j, :], tsample)
            ax[a].plot(tsample, yfit, 'r-', label='fit')
            ax[a].legend(title=f'({i}, {j})')
            ax[a].set_xlabel('time [s]')
            ax[a].set_ylabel('Intensity')
            a += 1

#%% apply correction only to sampled frames

print('\nApplying correction & detecting movement...')

diff_frames = []
batch_size = 4
t_array = np.array(t)

for start_idx in tqdm(range(0, n_frames, batch_size)):
    end_idx = min(start_idx + batch_size, n_frames)
    indices = list(range(start_idx, end_idx))

    # 1. Batch Fetch (Native Decord to NumPy)
    # Shape: (Batch, Rows, Cols, Channels)
    batch = vr.get_batch(indices).asnumpy()

    # 2. Extract Green Channel + Crop + Convert to float for math
    # Slicing [:, ymin:ymax, xmin:xmax, 1] is extremely fast in NumPy
    gray_batch = batch[:, ymin:ymax, xmin:xmax, 1].astype(np.float32)

    for i, frame_idx in enumerate(indices):
        ti = t_array[frame_idx]

        # 3. Polynomial correction (vectorized across the 2D frame)
        # Using list comprehension with np.sum is efficient for small degrees
        pol_cor = np.sum([coeff[:, :, d] * (ti ** (degree - d))
                          for d in range(degree + 1)], axis=0).astype(np.uint8)

        # 4. Calculate difference
        diff = (pol_cor - gray_batch[i]).astype(np.float32) * roi_mask

        # 5. Fast Fish Detection
        # np.abs(diff) > 15 creates a boolean mask; count_nonzero is the fastest way to sum it
        if np.count_nonzero(np.abs(diff) > 20) > 100:
            # We store the float32 diff to keep precision for downstream analysis
            diff_frames.append(frame_idx)


#%%
diff_frames = np.array(diff_frames)
np.save(video_path.replace('.mp4', '_suspicious_frames.npy'), diff_frames)

#%%