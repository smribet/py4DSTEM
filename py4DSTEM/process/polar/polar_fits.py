import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from emdfile import tqdmnd


def fit_amorphous_ring(
    im=None,
    center=None,
    radial_range=None,
    coefs=None,
    mask_dp=None,
    show_fit_mask=False,
    fit_all_images=False,
    gaussian_filter_sigma=None,
    maxfev=None,
    robust=False,
    robust_steps=3,
    robust_thresh=1.0,
    verbose=False,
    plot_result=True,
    plot_log_scale=False,
    plot_int_scale=(-3, 3),
    figsize=(8, 8),
    figax=None,
    return_all_coefs=True,
):
    """
    Fit an amorphous halo with a two-sided Gaussian model, plus a background
    Gaussian function.

    Parameters
    --------
    im: np.array
        2D image array to perform fitting on
    datacube: py4DSTEM.DataCube
        datacube to perform the fitting on
    center: np.array
        (x,y) center coordinates for fitting mask. If not specified
        by the user, we will assume the center coordinate is (im.shape-1)/2.
    radial_range: np.array
        (radius_inner, radius_outer) radial range to perform fitting over.
        If not specified by the user, we will assume (im.shape[0]/4,im.shape[0]/2).
    coefs: np.array (optional)
        Array containing initial fitting coefficients for the amorphous fit.
    mask_dp: np.array
        Dark field mask for fitting, in addition to the radial range specified above.
    show_fit_mask: bool
        Set to true to preview the fitting mask and initial guess for the ellipse params
    fit_all_images: bool
        Fit the elliptic parameters to all images
    gaussian_filter_sigma: float
        Standard deviation for Gaussian kernel in pixels.
    maxfev: int
        Max number of fitting evaluations for curve_fit.
    robust: bool
        Set to True to use robust fitting.
    robust_steps: int
        Number of robust fitting steps.
    robust_thresh: float
        Threshold for relative errors for outlier detection. Setting to 1.0 means all points beyond
        one standard deviation of the median error will be excluded from the next fit.
    verbose: bool
        Print fit results
    plot_result: bool
        Plot the result of the fitting
    plot_log_scale: bool
        Plot logarithmic image intensities
    plot_int_scale: tuple of 2 values
        Min and max plotting range in standard deviations of image intensity
    figsize: tuple, list, np.array (optional)
        Figure size for plots
    figax: tuple (fig, ax)
        axes for plotting figure
    return_all_coefs: bool
        Set to True to return the 11 parameter fit, rather than the 5 parameter ellipse

    Returns
    --------
    params_ellipse: np.array
        5 parameter elliptic fit coefficients
    params_ellipse_fit: np.array (optional)
        11 parameter elliptic fit coefficients
    """

    if gaussian_filter_sigma is not None:
        im = gaussian_filter(im, gaussian_filter_sigma)

    # Default values
    if center is None:
        center = np.array(((im.shape[0] - 1) / 2, (im.shape[1] - 1) / 2))
    if radial_range is None:
        radial_range = (im.shape[0] / 4, im.shape[0] / 2)

    # coordinates
    xa, ya = np.meshgrid(
        np.arange(im.shape[0]),
        np.arange(im.shape[1]),
        indexing="ij",
    )

    # Make fitting mask
    ra2 = (xa - center[0]) ** 2 + (ya - center[1]) ** 2
    mask = np.logical_and(
        ra2 >= radial_range[0] ** 2,
        ra2 <= radial_range[1] ** 2,
    )
    if mask_dp is not None:
        # Logical AND the radial mask with the user-provided mask
        mask = np.logical_and(mask, mask_dp)
    vals = im[mask]
    basis = np.vstack((xa[mask], ya[mask]))

    # initial fitting parameters
    if coefs is None:
        # ellipse parameters
        x0 = center[0]
        y0 = center[1]
        R_mean = np.mean(radial_range)
        # A = 1/R_mean**2
        # B = 0
        # C = 1/R_mean**2
        a = R_mean
        b = R_mean
        t = 0

        # Gaussian model parameters
        int_min = np.min(vals)
        int_max = np.max(vals)
        int0 = (int_max - int_min) / 2
        int12 = (int_max - int_min) / 2
        k_bg = int_min
        sigma0 = np.mean(radial_range)
        sigma1 = (radial_range[1] - radial_range[0]) / 4
        sigma2 = (radial_range[1] - radial_range[0]) / 4

        coefs = (x0, y0, a, b, t, int0, int12, k_bg, sigma0, sigma1, sigma2)
    lb = (0, 0, radial_range[0], radial_range[0], -np.inf, 0, 0, 0, 1, 1, 1)
    ub = (
        im.shape[0],
        im.shape[1],
        radial_range[1],
        radial_range[1],
        np.inf,
        np.inf,
        np.inf,
        np.inf,
        np.inf,
        np.inf,
        np.inf,
    )

    if show_fit_mask:
        # show image preview of fitting mask

        # Generate hybrid image for plotting
        if plot_log_scale:
            int_med = np.median(np.log(vals))
            int_std = np.sqrt(np.median((np.log(vals) - int_med) ** 2))
            int_range = (
                int_med + plot_int_scale[0] * int_std,
                int_med + plot_int_scale[1] * int_std,
            )
            im_plot = np.tile(
                np.clip(
                    (np.log(im[:, :, None]) - int_range[0])
                    / (int_range[1] - int_range[0]),
                    0,
                    1,
                ),
                (1, 1, 3),
            )

        else:
            int_med = np.median(vals)
            int_std = np.sqrt(np.median((vals - int_med) ** 2))
            int_range = (
                int_med + plot_int_scale[0] * int_std,
                int_med + plot_int_scale[1] * int_std,
            )
            im_plot = np.tile(
                np.clip(
                    (im[:, :, None] - int_range[0]) / (int_range[1] - int_range[0]),
                    0,
                    1,
                ),
                (1, 1, 3),
            )
        im_plot[:, :, 0] *= 1 - mask

        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax
        ax.imshow(im_plot)

    else:
        # Perform elliptic fitting
        int_mean = np.mean(vals)

        if maxfev is None:
            coefs = curve_fit(
                amorphous_model,
                basis,
                vals / int_mean,
                p0=coefs,
                xtol=1e-8,
                bounds=(lb, ub),
            )[0]
        else:
            coefs = curve_fit(
                amorphous_model,
                basis,
                vals / int_mean,
                p0=coefs,
                xtol=1e-8,
                bounds=(lb, ub),
                maxfev=maxfev,
            )[0]
        coefs[4] = np.mod(coefs[4], 2 * np.pi)

        if robust:
            for a0 in range(robust_steps):
                # find outliers
                int_fit = amorphous_model(basis, *coefs)
                int_diff = vals / int_mean - int_fit
                int_diff /= np.median(np.abs(int_diff))
                sub_fit = int_diff**2 < robust_thresh**2

                # redo fits excluding the outliers
                if maxfev is None:
                    coefs = curve_fit(
                        amorphous_model,
                        basis[:, sub_fit],
                        vals[sub_fit] / int_mean,
                        p0=coefs,
                        xtol=1e-8,
                        bounds=(lb, ub),
                    )[0]
                else:
                    coefs = curve_fit(
                        amorphous_model,
                        basis[:, sub_fit],
                        vals[sub_fit] / int_mean,
                        p0=coefs,
                        xtol=1e-8,
                        bounds=(lb, ub),
                        maxfev=maxfev,
                    )[0]
                coefs[4] = np.mod(coefs[4], 2 * np.pi)

        # Scale intensity coefficients
        coefs[5:8] *= int_mean

    if verbose:
        print("x0 = " + str(np.round(coefs[0], 3)) + " px")
        print("y0 = " + str(np.round(coefs[1], 3)) + " px")
        print("a  = " + str(np.round(coefs[2], 3)) + " px")
        print("b  = " + str(np.round(coefs[3], 3)) + " px")
        print("t  = " + str(np.round(np.rad2deg(coefs[4]), 3)) + " deg")

    if plot_result and not show_fit_mask:
        plot_amorphous_ring(
            im=im,
            coefs=coefs,
            radial_range=radial_range,
            plot_log_scale=plot_log_scale,
            plot_int_scale=plot_int_scale,
            figsize=figsize,
            figax=figax,
        )

    # Return fit parameters
    if return_all_coefs:
        return coefs
    else:
        return coefs[:5]


def fit_amorphous_ring_all(
    datacube=None,
    center=None,
    radial_range=None,
    coefs=None,
    mask_dp=None,
    show_fit_mask=False,
    fit_all_images=False,
    gaussian_filter_sigma=None,
    maxfev=None,
    robust=False,
    robust_steps=3,
    robust_thresh=1.0,
    verbose=False,
    plot_result=True,
    plot_log_scale=False,
    plot_int_scale=(-3, 3),
    figsize=(8, 8),
    return_all_coefs=True,
    progress_bar=True,
    seed_with_mean_dp=True,
    distributed=False,
):
    """
    Fit an amorphous halo with a two-sided Gaussian model, plus a background
    Gaussian function for all patterns in a datacube

    Parameters
    --------
    im: np.array
        2D image array to perform fitting on
    datacube: py4DSTEM.DataCube
        datacube to perform the fitting on
    center: np.array
        (x,y) center coordinates for fitting mask. If not specified
        by the user, we will assume the center coordinate is (im.shape-1)/2.
        Can also be a tuple of arrays the size of real space if the center shifts
        across the diffraction pattern.
    radial_range: np.array
        (radius_inner, radius_outer) radial range to perform fitting over.
        If not specified by the user, we will assume (im.shape[0]/4,im.shape[0]/2).
    coefs: np.array (optional)
        Array containing initial fitting coefficients for the amorphous fit.
    mask_dp: np.array
        Dark field mask for fitting, in addition to the radial range specified above.
    show_fit_mask: bool
        Set to true to preview the fitting mask and initial guess for the ellipse params
    fit_all_images: bool
        Fit the elliptic parameters to all images
    gaussian_filter_sigma: float
        Standard deviation for Gaussian kernel in pixels.
    maxfev: int
        Max number of fitting evaluations for curve_fit.
    robust: bool
        Set to True to use robust fitting.
    robust_steps: int
        Number of robust fitting steps.
    robust_thresh: float
        Threshold for relative errors for outlier detection. Setting to 1.0 means all points beyond
        one standard deviation of the median error will be excluded from the next fit.
    verbose: bool
        Print fit results
    plot_result: bool
        Plot the result of the fitting
    plot_log_scale: bool
        Plot logarithmic image intensities
    plot_int_scale: tuple of 2 values
        Min and max plotting range in standard deviations of image intensity
    figsize: tuple, list, np.array (optional)
        Figure size for plots
    return_all_coefs: bool
        Set to True to return the 11 parameter fit, rather than the 5 parameter ellipse
    seed_with_dp_mean: bool
        If True, performs inital fit on mean dataset and seeds each with mean parameters
    distributed: bool
        If True, runs parallelized with mpire

    Returns
    --------
    params_ellipse: np.array
        5 parameter elliptic fit coefficients
    params_ellipse_fit: np.array (optional)
        11 parameter elliptic fit coefficients
    """

    from py4DSTEM.process.polar import fit_amorphous_ring

    coefs_all = np.zeros((datacube.shape[0], datacube.shape[1], 11))

    dp_mean = datacube.data.mean((0, 1))

    if center[0] is np.ndarray:
        center_all = center
        center_mean = (center_all[0].mean(), center_all[1].mean())

    else:
        center_all = (
            center[0] * np.ones(datacube.shape[0:2]),
            center[1] * np.ones(datacube.shape[0:2]),
        )
        center_mean = center

    if seed_with_mean_dp:
        coefs_mean = fit_amorphous_ring(
            im=dp_mean,
            center=center_mean,
            radial_range=radial_range,
            coefs=coefs,
            mask_dp=mask_dp,
            show_fit_mask=show_fit_mask,
            gaussian_filter_sigma=gaussian_filter_sigma,
            maxfev=maxfev,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
            verbose=verbose,
            plot_result=plot_result,
            plot_log_scale=plot_log_scale,
            plot_int_scale=plot_int_scale,
            figsize=figsize,
            return_all_coefs=True,
        )

    else:
        coefs_mean = None

    if distributed is False:
        for rx, ry in tqdmnd(
            datacube.shape[0],
            datacube.shape[1],
            desc="Radial statistics",
            unit=" probe positions",
            disable=not progress_bar,
        ):
            if seed_with_mean_dp:
                coefs_mean[0] = center_all[0][rx, ry]
                coefs_mean[1] = center_all[1][rx, ry]

            try:
                coefs_single = fit_amorphous_ring(
                    im=datacube.data[rx, ry],
                    center=(center_all[0][rx, ry], center_all[1][rx, ry]),
                    radial_range=radial_range,
                    coefs=coefs_mean,
                    mask_dp=mask_dp,
                    show_fit_mask=False,
                    gaussian_filter_sigma=gaussian_filter_sigma,
                    maxfev=maxfev,
                    robust=robust,
                    robust_steps=robust_steps,
                    robust_thresh=robust_thresh,
                    verbose=False,
                    plot_result=False,
                    return_all_coefs=True,
                )
            except:
                coefs_single = 0
            coefs_all[rx, ry] = coefs_single
    else:
        from mpire import WorkerPool, cpu_count
        from threadpoolctl import threadpool_limits

        threads_per_job = 1

        def f(args):
            with threadpool_limits(limits=threads_per_job):
                return _fig_amorphous_ring(**args)

        def _fig_amorphous_ring(rx, ry, seed_with_mean_dp, coefs_mean, **kwargs):
            if seed_with_mean_dp:
                coefs_mean[0] = center_all[0][rx, ry]
                coefs_mean[1] = center_all[1][rx, ry]
            try:
                coefs_single = fit_amorphous_ring(**kwargs)
            except:
                coefs_single = 0
            return rx, ry, coefs_single

        inputs = [
            (
                {
                    "rx": rx,
                    "ry": ry,
                    "seed_with_mean_dp": seed_with_mean_dp,
                    "coefs_mean": coefs_mean,
                    "im": datacube.data[rx, ry],
                    "center": (center_all[0][rx, ry], center_all[1][rx, ry]),
                    "radial_range": radial_range,
                    "coefs": coefs_mean,
                    "mask_dp": mask_dp,
                    "show_fit_mask": False,
                    "gaussian_filter_sigma": gaussian_filter_sigma,
                    "maxfev": maxfev,
                    "robust": robust,
                    "robust_steps": robust_steps,
                    "robust_thresh": robust_thresh,
                    "verbose": False,
                    "plot_result": False,
                    "return_all_coefs": True,
                },
            )
            for rx in range(datacube.shape[0])
            for ry in range(datacube.shape[1])
        ]

        num_jobs = cpu_count() // threads_per_job

        with WorkerPool(
            n_jobs=num_jobs,
        ) as pool:
            results = pool.map(
                f,
                inputs,
                progress_bar=True,
            )

        for a0 in range(len(results)):
            coefs_all[results[a0][0], results[a0][1]] = results[a0][2]

    if return_all_coefs:
        return coefs_all
    else:
        return coefs_all[:, :, :5]


def plot_amorphous_ring(
    im,
    coefs,
    radial_range=(0, np.inf),
    plot_log_scale=True,
    plot_int_scale=(-3, 3),
    figsize=(8, 8),
    figax=None,
):
    """
    Fit an amorphous halo with a two-sided Gaussian model, plus a background
    Gaussian function.

    Parameters
    --------
    im: np.array
        2D image array to perform fitting on
    coefs: np.array
        all fitting coefficients
    plot_log_scale: bool
        Plot logarithmic image intensities
    plot_int_scale: tuple of 2 values
        Min and max plotting range in standard deviations of image intensity
    figsize: tuple, list, np.array (optional)
        Figure size for plots
    return_all_coefs: bool
        Set to True to return the 11 parameter fit, rather than the 5 parameter ellipse

    Returns
    --------

    """

    # get needed coefs
    center = coefs[0:2]

    # coordinates
    xa, ya = np.meshgrid(
        np.arange(im.shape[0]),
        np.arange(im.shape[1]),
        indexing="ij",
    )

    # Make fitting mask
    ra2 = (xa - center[0]) ** 2 + (ya - center[1]) ** 2
    mask = np.logical_and(
        ra2 >= radial_range[0] ** 2,
        ra2 <= radial_range[1] ** 2,
    )
    vals = im[mask]
    basis = np.vstack((xa[mask], ya[mask]))

    # Generate resulting best fit image
    im_fit = np.reshape(
        amorphous_model(np.vstack((xa.ravel(), ya.ravel())), coefs), im.shape
    )

    # plotting arrays
    phi = np.linspace(0, 2 * np.pi, 360)
    cp = np.cos(phi)
    sp = np.sin(phi)

    # plotting intensity range
    if plot_log_scale:
        int_med = np.median(np.log(vals))
        int_std = np.sqrt(np.median((np.log(vals) - int_med) ** 2))
        int_range = (
            int_med + plot_int_scale[0] * int_std,
            int_med + plot_int_scale[1] * int_std,
        )
        im_plot = np.tile(
            np.clip(
                (np.log(im[:, :, None]) - int_range[0]) / (int_range[1] - int_range[0]),
                0,
                1,
            ),
            (1, 1, 3),
        )
    else:
        int_med = np.median(vals)
        int_std = np.sqrt(np.median((vals - int_med) ** 2))
        int_range = (
            int_med + plot_int_scale[0] * int_std,
            int_med + plot_int_scale[1] * int_std,
        )
        im_plot = np.clip(
            (im[:, :, None] - int_range[0]) / (int_range[1] - int_range[0]), 0, 1
        )
    # vals_mean = np.mean(vals)
    # vals_std = np.std(vals)
    # vmin = vals_mean -

    # plotting
    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax
    ax.imshow(
        im_plot,
        vmin=0,
        vmax=1,
        cmap="gray",
    )

    x0 = coefs[0]
    y0 = coefs[1]
    a = coefs[2]
    b = coefs[3]
    t = coefs[4]
    s1 = coefs[9]
    s2 = coefs[10]

    ax.plot(
        y0 + np.array((-1, 1)) * a * np.sin(t),
        x0 + np.array((-1, 1)) * a * np.cos(t),
        c="r",
    )
    ax.plot(
        y0 + np.array((-1, 1)) * b * np.cos(t),
        x0 + np.array((1, -1)) * b * np.sin(t),
        c="r",
        linestyle="dashed",
    )

    ax.plot(
        y0 + a * np.sin(t) * cp + b * np.cos(t) * sp,
        x0 + a * np.cos(t) * cp - b * np.sin(t) * sp,
        c="r",
    )
    scale = 1 - s1 / a
    ax.plot(
        y0 + scale * a * np.sin(t) * cp + scale * b * np.cos(t) * sp,
        x0 + scale * a * np.cos(t) * cp - scale * b * np.sin(t) * sp,
        c="r",
        linestyle="dashed",
    )
    scale = 1 + s2 / a
    ax.plot(
        y0 + scale * a * np.sin(t) * cp + scale * b * np.cos(t) * sp,
        x0 + scale * a * np.cos(t) * cp - scale * b * np.sin(t) * sp,
        c="r",
        linestyle="dashed",
    )
    ax.set_xlim((0, im.shape[1] - 1))
    ax.set_ylim((im.shape[0] - 1, 0))


def amorphous_model(basis, *coefs):
    coefs = np.squeeze(np.array(coefs))

    x0 = coefs[0]
    y0 = coefs[1]
    a = coefs[2]
    b = coefs[3]
    t = coefs[4]
    # A = coefs[2]
    # B = coefs[3]
    # C = coefs[4]
    int0 = coefs[5]
    int12 = coefs[6]
    k_bg = coefs[7]
    sigma0 = coefs[8]
    sigma1 = coefs[9]
    sigma2 = coefs[10]

    x0 = basis[0, :] - x0
    y0 = basis[1, :] - y0
    x = np.cos(t) * x0 - (b / a) * np.sin(t) * y0
    y = np.sin(t) * x0 + (b / a) * np.cos(t) * y0

    r2 = x**2 + y**2
    dr = np.sqrt(r2) - b
    dr2 = dr**2
    sub = dr < 0

    int_model = k_bg + int0 * np.exp(r2 / (-2 * sigma0**2))
    int_model[sub] += int12 * np.exp(dr2[sub] / (-2 * sigma1**2))
    sub = np.logical_not(sub)
    int_model[sub] += int12 * np.exp(dr2[sub] / (-2 * sigma2**2))

    return int_model


def _calc_transform(params):
    if params[2] < params[3]:
        a = params[3]
        b = params[2]
        theta = params[4] + np.pi / 2
    else:
        a = params[2]
        b = params[3]
        theta = params[4]

    # coefs
    Aprime = a**2 * (np.sin(theta)) ** 2 + b**2 * (np.cos(theta)) ** 2
    Bprime = 2 * (a**2 - b**2) * np.sin(theta) * np.cos(theta)
    Cprime = a**2 * (np.cos(theta)) ** 2 + b**2 * (np.sin(theta)) ** 2

    m_meas = np.array(
        (
            (Aprime, Bprime / 2),
            (Bprime / 2, Cprime),
        )
    )

    e_vals, e_vects = np.linalg.eig(m_meas)
    phi = np.arctan2(e_vects[1, 0], e_vects[0, 0])

    # rotation matrix
    rot_mat = np.array(((np.cos(phi), -np.sin(phi)), (np.sin(phi), np.cos(phi))))

    m_ref = np.diag(np.sqrt(e_vals))

    transformation_matrix = rot_mat @ m_ref @ rot_mat.T

    return transformation_matrix


def _calc_strain(transformation_matrix, transformation_matrix_ref):

    transformation_matrix = transformation_matrix @ np.linalg.inv(
        transformation_matrix_ref
    )
    exx_fit = transformation_matrix[0, 0] - 1
    eyy_fit = transformation_matrix[1, 1] - 1
    exy_fit = 0.5 * (transformation_matrix[0, 1] + transformation_matrix[1, 0])

    return exx_fit, eyy_fit, exy_fit


def calculate_amorphous_strain(
    params_all,
    ref_region_mask=None,
    progress_bar=True,
):

    transformation_matrix_all = np.zeros(
        (params_all.shape[0], params_all.shape[1], 2, 2)
    )

    for rx, ry in tqdmnd(
        params_all.shape[0],
        params_all.shape[1],
        desc="transformation",
        unit="probe positions",
        disable=not progress_bar,
    ):
        transformation_matrix_all[rx, ry, :, :] = _calc_transform(params_all[rx, ry, :])

    if ref_region_mask is None:
        ref_region_mask = np.ones((params_all.shape[0:2]), dtype="bool")

    transformation_matrix_ref = transformation_matrix_all[ref_region_mask].mean(0)

    strain_all = np.zeros((params_all.shape[0], params_all.shape[1], 3))

    for rx, ry in tqdmnd(
        params_all.shape[0],
        params_all.shape[1],
        desc="strain",
        unit="probe positions",
        disable=not progress_bar,
    ):
        strain_all[rx, ry] = _calc_strain(
            transformation_matrix_all[rx, ry], transformation_matrix_ref
        )

    return strain_all
