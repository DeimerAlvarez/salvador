"""Continuum normalization (NORM = 1, 2; NORM = 3 under development) and continuum regions."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from astropy.stats import mad_std
from astropy.stats import sigma_clip

from .config import ksize, plt


def norm1(wvl_obs, fl_obs, wvl_syn, fl_syn, continuum_delta=0.005):
    """Re-scale the observed spectrum using the sythetic one."""

    try:
        max_continuum_syn = np.nanmax(fl_syn)
    except ValueError:
        print("Warning: Synthetic spectrum empty. Imposible re-normalize.")
        return fl_obs

    threshold = max_continuum_syn - continuum_delta

    # find conti points on the synthetic and extract the lambda
    continuum_mask_syn = (fl_syn >= threshold)
    wv_continuum_points = wvl_syn[continuum_mask_syn]

    # Check the number of points
    if len(wv_continuum_points) < 10:
        print("Warning: few continuum points (<10) found.")
        return fl_obs

    obs_interpolator = interp1d(wvl_obs, fl_obs, kind='linear', bounds_error=False, fill_value=np.nan)
    fl_obs_at_continuum = obs_interpolator(wv_continuum_points)

    current_continuum_level = np.nanmedian(fl_obs_at_continuum)

    if np.isnan(current_continuum_level) or current_continuum_level == 0:
        print("Warning: Continuum level not valid. Impossible to re-normalize.")
        return fl_obs

    fl_obs_renorm = fl_obs / current_continuum_level
    return fl_obs_renorm


def merge_overlapping_regions(wv_left_raw, wv_right_raw):
    """Merge overlapping wavelength regions [left, right]. Returns the arrays of the merged edges."""
    if len(wv_left_raw) == 0:
        return np.array([]), np.array([])

    # 1. Pairs [start, end] sorted by 'start'
    regions = sorted(zip(wv_left_raw, wv_right_raw))

    merged_left = []
    merged_right = []

    # 2. Start from the first region
    current_left, current_right = regions[0]

    # 3. Itera sulle regioni rimanenti
    for i in range(1, len(regions)):
        next_left, next_right = regions[i]

        # Check for an overlap
        if next_left <= current_right:
            # Overlap: extend the end of the current region
            current_right = max(current_right, next_right)
        else:
            # No overlap: save the current region and start a new one
            merged_left.append(current_left)
            merged_right.append(current_right)
            current_left, current_right = next_left, next_right

    # 4. Add the last region
    merged_left.append(current_left)
    merged_right.append(current_right)

    return np.array(merged_left), np.array(merged_right)


def load_cw_regions(file_path):
    """Reads the cw file: remove the duplciates and merge the overlapping regions"""
    # --- 1. Read the file with pandas ---
    try:
        df = pd.read_csv(file_path, header=None, sep=r'\s+', names=['lambda1', 'lambda2'])

    except FileNotFoundError:
        print(f"Error: file not found {file_path}")
        return np.array([]), np.array([])
    except pd.errors.EmptyDataError:
        print(f"Error: file {file_path} empty.")
        return np.array([]), np.array([])


    # 2. Remove EXACT duplicates
    df_unique = df.drop_duplicates()

    # 3. Sort by lambda1 (needed to merge)
    df_sorted = df_unique.sort_values(by='lambda1').reset_index(drop=True)

    if df_sorted.empty:
        return np.array([]), np.array([])

    # 4. Merge the overlapping regions
    merged_left = []
    merged_right = []

    current_left = df_sorted.loc[0, 'lambda1']
    current_right = df_sorted.loc[0, 'lambda2']

    for i in range(1, len(df_sorted)):
        next_left = df_sorted.loc[i, 'lambda1']
        next_right = df_sorted.loc[i, 'lambda2']

        if next_left <= current_right:
            current_right = max(current_right, next_right)
        else:
            merged_left.append(current_left)
            merged_right.append(current_right)
            current_left = next_left
            current_right = next_right

    # 5. Save the last region
    merged_left.append(current_left)
    merged_right.append(current_right)

    return np.array(merged_left), np.array(merged_right)

def find_clipped_regions(wv_left_raw, wv_right_raw, wvl, fl):
    """Continuum regions cleaned of absorption features: in each region the pixels outside 5 sigma
    (median, MAD) are removed and the groups of consecutive valid pixels ("islands") become the new
    regions. Returns the arrays of their left and right edges."""
    wv_left_new = []
    wv_right_new = []

    # merge overlapping regions
    merged_left, merged_right = merge_overlapping_regions(wv_left_raw, wv_right_raw)

    for i in range(len(wv_left_raw)):
        l = np.where((wvl > wv_left_raw[i]) & (wvl < wv_right_raw[i]))
        wv_tmp, fl_tmp = wvl[l], fl[l]

        if len(wv_tmp) <= 1: # Skip if 0 or 1 points
            continue

        clipped_data = sigma_clip(fl_tmp, sigma_lower=5, sigma_upper=5, cenfunc='median', stdfunc=mad_std)
        mask = ~clipped_data.mask

        if not np.any(mask):
            continue

        # Find the "islands" (groups of consecutive valid points)
        mask_int = mask.astype(int)
        padded_mask = np.pad(mask_int, 1, 'constant')
        diffs = np.diff(padded_mask)

        start_indices = np.where(diffs == 1)[0]
        end_indices = np.where(diffs == -1)[0] - 1

        new_starts_lambda = wv_tmp[start_indices]
        new_ends_lambda = wv_tmp[end_indices]

        wv_left_new.extend(new_starts_lambda)
        wv_right_new.extend(new_ends_lambda)

    return np.array(wv_left_new), np.array(wv_right_new)


def norm2(wv_left_clean, wv_right_clean, wvl, fl, fl_syn=None):
    """Normalization with continuum windows (NORM = 2): the mean flux in each window gives the
    continuum, interpolated over the spectrum (cubic with 4 or more windows, otherwise linear).
    With the synthetic fl_syn, the normalized flux is multiplied by the level of the synthetic in
    the same windows (interpolated in the same way), so that weak lines inside the windows do not
    lower the continuum. Returns the normalized flux (the input flux if there are < 2 windows)."""
    wv_cont = np.array([])
    fl_cont = np.array([])

    if fl_syn is not None:
        fl_cont_syn = np.array([])

    # 1. Collect all the points of all the clean regions
    for i in range(len(wv_left_clean)):
        mask = (wvl >= wv_left_clean[i]) & (wvl <= wv_right_clean[i])

        A, B = np.mean(wvl[mask]), np.mean(fl[mask])

        if not (np.isnan(A) or np.isnan(B)):
            wv_cont = np.append(wv_cont, A)
            fl_cont = np.append(fl_cont, B)

            if fl_syn is not None:
                C = np.mean(fl_syn[mask])
                fl_cont_syn = np.append(fl_cont_syn, C)

    # 2. Sort the points (needed for the interpolation)
    sort_idx = np.argsort(wv_cont)
    wv_cont = wv_cont[sort_idx]
    fl_cont = fl_cont[sort_idx]

    if fl_syn is not None:
        fl_cont_syn = fl_cont_syn[sort_idx]

    # 3. Check that there are enough points
    num_points = len(wv_cont)
    if num_points < 2:
        print("Error: not enough points to perform the normalization.")
        return fl

    # Type of interpolation (cubic with enough points, otherwise linear)
    kind_interp = 'linear' if num_points < 4 else 'cubic'

    # 4. Interpolation of the observed spectrum (always)
    interpolator = interp1d(wv_cont, fl_cont, kind=kind_interp, fill_value='extrapolate')
    continuum = interpolator(wvl)

    # 5. Final continuum, depending on the presence of the synthetic
    if fl_syn is not None:
        interp_syn = interp1d(wv_cont, fl_cont_syn, kind=kind_interp, fill_value='extrapolate')
        continuum_syn = interp_syn(wvl)
        fl_norm = (fl / continuum) * continuum_syn
    else:
        fl_norm = fl / continuum

    return fl_norm

def norm3(wv_obs, fl_obs, wv_syn, fl_syn):  # TO BE IMPROVED (NORM = 3 is still under development)
    # work more on this normalization mode
    """Normalization with a polynomial fit of observed/synthetic (NORM = 3). STILL UNDER DEVELOPMENT:
    salvador stops at start-up if NORM = 3."""
    fl_syn = np.interp(wv_obs, wv_syn, fl_syn)

    plt.plot(wv_obs, fl_obs)
    plt.plot(wv_obs, fl_syn)
    plt.show()

    residuals = fl_obs / fl_syn
    coeffs = np.polyfit(wv_obs, residuals, 5)
    polynomial = np.poly1d(coeffs)

    # Fitted values (the estimated continuum)
    fitted_flux = polynomial(wv_obs)
    fl_norm1 = fl_obs / fitted_flux

    ##
    padded_flux = np.pad(residuals, (ksize//2, ksize//2), mode='reflect')
    smoothed_flux = medfilt(padded_flux, kernel_size=ksize)
    continuum = smoothed_flux[ksize//2 : -ksize//2 +1]

    # normalization
    fl_norm2 = fl_obs / continuum

    plt.plot(residuals, fitted_flux)
    plt.plot(residuals, continuum)
    plt.show()

    plt.plot(wv_obs, fl_norm1, c='b')
    plt.plot(wv_obs, fl_norm2, c='r')
    plt.plot(wv_obs, fl_syn)
    plt.show()


    return fl_norm2
