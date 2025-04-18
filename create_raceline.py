import numpy as np
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

def create_raceline(refline: np.ndarray,
                    normvectors: np.ndarray,
                    boundaries: np.ndarray,
                    alpha: np.ndarray,
                    stepsize_interp: float) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function includes the algorithm part connected to the interpolation of the raceline after the optimization.

    .. inputs::
    :param refline:         array containing the track reference line [x, y] (unit is meter, must be unclosed!)
    :type refline:          np.ndarray
    :param normvectors:     normalized normal vectors for every point of the reference line [x_component, y_component]
                            (unit is meter, must be unclosed!)
    :type normvectors:      np.ndarray
    :param alpha:           solution vector of the optimization problem containing the lateral shift in m for every point.
    :type alpha:            np.ndarray
    :param stepsize_interp: stepsize in meters which is used for the interpolation after the raceline creation.
    :type stepsize_interp:  float

    .. outputs::
    :return raceline_interp:                interpolated raceline [x, y] in m.
    :rtype raceline_interp:                 np.ndarray
    :return A_raceline:                     linear equation system matrix of the splines on the raceline.
    :rtype A_raceline:                      np.ndarray
    :return coeffs_x_raceline:              spline coefficients of the x-component.
    :rtype coeffs_x_raceline:               np.ndarray
    :return coeffs_y_raceline:              spline coefficients of the y-component.
    :rtype coeffs_y_raceline:               np.ndarray
    :return spline_inds_raceline_interp:    contains the indices of the splines that hold the interpolated points.
    :rtype spline_inds_raceline_interp:     np.ndarray
    :return t_values_raceline_interp:       containts the relative spline coordinate values (t) of every point on the
                                            splines.
    :rtype t_values_raceline_interp:        np.ndarray
    :return s_raceline_interp:              total distance in m (i.e. s coordinate) up to every interpolation point.
    :rtype s_raceline_interp:               np.ndarray
    :return spline_lengths_raceline:        lengths of the splines on the raceline in m.
    :rtype spline_lengths_raceline:         np.ndarray
    :return el_lengths_raceline_interp_cl:  distance between every two points on interpolated raceline in m (closed!).
    :rtype el_lengths_raceline_interp_cl:   np.ndarray
    :return normals_interp:                 Interpolated normal vectors corresponding to raceline_interp.
    """

    # calculate raceline on the basis of the optimized alpha values
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # Calculate the width of the track at every point from the raceline
    w_new_right = boundaries[:,0] - alpha 
    w_new_left = boundaries[:,1] + alpha

    left_track_bound = raceline - normvectors * np.expand_dims(w_new_left, 1)
    right_track_bound = raceline + normvectors * np.expand_dims(w_new_right, 1)
    plt.plot(raceline[:, 0], raceline[:, 1], label="Raceline")
    plt.plot(refline[:, 0], refline[:, 1], label="Refline")
    plt.plot(left_track_bound[:, 0], left_track_bound[:, 1], label="Left Track Bound")
    plt.plot(right_track_bound[:, 0], right_track_bound[:, 1], label="Right Track Bound")
    plt.title("Track before interpolation")
    plt.legend()
    plt.show()

    # closed raceline for spline calculation
    raceline_cl = np.vstack((raceline, raceline[0]))

    # calculate new splines on the basis of the raceline
    coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline = tph.calc_splines.\
        calc_splines(path=raceline_cl,
                     use_dist_scaling=False)

    # calculate new spline lengths
    spline_lengths_raceline = tph.calc_spline_lengths. \
        calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                            coeffs_y=coeffs_y_raceline)

    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.\
        interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                      coeffs_x=coeffs_x_raceline,
                                      coeffs_y=coeffs_y_raceline,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp)

    # interpolate normal vectors for evenly spaced raceline points
    raceline_interp_cl = np.vstack((raceline_interp, raceline_interp[0]))
    x_interp, y_interp, A_interp, normvectors_interp = tph.calc_splines.calc_splines(path=raceline_interp_cl)
    norm_x_interp = normvectors_interp[:, 0]
    norm_y_interp = normvectors_interp[:, 1]
    normals_interp = np.vstack((norm_x_interp, norm_y_interp)).T
    normals_interp /= np.linalg.norm(normals_interp, axis=1, keepdims=True)

    print(f"shape of raceline_interp: {raceline_interp.shape}")
    print(f"shape of t_values_raceline_interp: {t_values_raceline_interp.shape}")

    w_right_interp = np.interp(np.linspace(0, 1, len(raceline_interp_cl)), np.linspace(0, 1, len(raceline)), w_new_right)
    w_left_interp = np.interp(np.linspace(0, 1, len(raceline_interp_cl)), np.linspace(0, 1, len(raceline)), w_new_left)

    buffer = 0.1
    w_right_interp = w_right_interp[:-1] - buffer
    w_left_interp = w_left_interp[:-1] - buffer

    print(f"shape of raceline_interp: {raceline_interp.shape}")
    print(f"shape of w_right_interp: {w_right_interp.shape}")

    left_bound_interp = raceline_interp - normals_interp * np.expand_dims(w_left_interp, 1)
    right_bound_interp = raceline_interp + normals_interp * np.expand_dims(w_right_interp, 1)
    
    plt.plot(raceline_interp[:, 0], raceline_interp[:, 1], label="Interpolated Raceline")
    plt.plot(left_bound_interp[:, 0], left_bound_interp[:, 1], label="Interpolated Left Track Bound")
    plt.plot(right_bound_interp[:, 0], right_bound_interp[:, 1], label="Interpolated Right Track Bound")
    plt.plot(refline[:, 0], refline[:, 1], label="Refline")
    plt.plot(left_track_bound[:, 0], left_track_bound[:, 1], label="Left Track Bound before interpolation")
    plt.plot(right_track_bound[:, 0], right_track_bound[:, 1], label="Right Track Bound before interpolation")
    plt.title("Track after interpolation")
    plt.legend()
    plt.show()

    # calculate element lengths
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])

    # plot test
    plt.plot(raceline_cl[:, 0], raceline_cl[:, 1], label="Interpolated Spline Path")
    plt.axis('equal')

    # Scale the normal vectors for visualization
    scale_factor = 2.0  # Adjust as necessary for better visibility
    norm_x_scaled = normvectors_raceline[:,0] * scale_factor
    norm_y_scaled = normvectors_raceline[:,1] * scale_factor

    # Plot the normal vectors as arrows using quiver
    plt.quiver(raceline[:, 0], raceline[:, 1],
               norm_x_scaled, norm_y_scaled, angles='xy', scale_units='xy', scale=1, color='r', label="Normal Vectors")

    plt.legend()
    plt.title("Spline Path with Normal Vectors")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl, normals_interp, w_right_interp, w_left_interp

# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass