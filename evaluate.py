from deep_radiologist.actions import locate_peaks
from deep_radiologist.data_loading import _load_point_data
import torchio as tio
import numpy as np
from scipy import spatial
import napari


def _get_acc_metrics(y_hat, y, k=3):
        """Calculates accuracy metrics for a set of predicted and ground truth coordinates.

        Is a true positive if the distance between the predicted and closest ground truth coordinate
        is less than the correct_prediction_distance config parameter and that ground truth coordinate
        doesn't already have a better matching prediction (tested up to k closest matches). Is a false
        positive if the distance is greater than the correct_prediction_distance parameter or it
        already has a closer true positive. Is a false negative if the ground truth does not have a
        corresponding true positive.

        Args:
            y_hat (np.ndarray): predicted coordinates
            y (np.ndarray): ground truth coordinates

        Returns:
            tp (float): true positives
            fp (float): false positives
            fn (float): false negatives
            loc_errs (np.ndarray): location errors
        """

        correct_prediction_distance = 15

        tree = spatial.cKDTree(y_hat)
        closest_dists, closest_nbrs = tree.query(y, k=k)

        y_match = list()
        y_hat_match = list()
        dists = list()

        for i in range(k):
            nbrs_k = closest_nbrs[:, i]
            dists_k = closest_dists[:, i]

            # sort by closest distance
            sort_idx = np.argsort(dists_k)
            nbrs_k = nbrs_k[sort_idx]
            dists_k = dists_k[sort_idx]

            for j in range(len(nbrs_k)):
                if j not in y_hat_match and y[j, i] not in y_match:
                    y_hat_match.append(j)
                    y_match.append(y[j, i])
                    dists.append(dists_k[j])

        dists = np.array(dists)

        tp = len(dists[dists < correct_prediction_distance])
        fp = len(y_hat) - tp
        fn = len(y) - tp

        loc_errors = dists[dists < correct_prediction_distance]

        if len(loc_errors) == 0:
            loc_errors = np.array([0])


        # fp_prediction = y_
        tp_y = y[y_match]
        tp_y_hat = y_hat[y_hat_match]
        fp_y_hat = y_hat[~y_hat_match]

        # things_to_plot = [y, fp_prediction, fn_groundtruth, tp_groundtruth, tp_prediction]

        return tp, fp, fn, loc_errors#, things_to_plot


def evaluate(x, y, y_hat, plot=True):
    mct = tio.ScalarImage(x)
    prediction = tio.ScalarImage(y_hat)

    prediction_locations = locate_peaks(
        y_hat,
        save=True,
        plot=False,
        peak_min_dist=4,
        peak_min_val=0.5,
    )
    ground_truth_locations=np.loadtxt(
        y,
        delimiter=',',
        dtype=np.float
    ).astype(int).T
    
    # flip axis 0 and 1 to convert from LPS+ to RAS+
    ground_truth_locations[:,0] = mct.shape[1] - ground_truth_locations[:,0]
    ground_truth_locations[:,1] = mct.shape[2] - ground_truth_locations[:,1]

    tp, fp, fn, loc_errors, things_to_plot = _get_acc_metrics(prediction_locations, ground_truth_locations)

    viewer = napari.view_points(things_to_plot[0], name='all ground truth', size=6, face_color='pink')
    viewer.add_points(things_to_plot[1], name='fp prediction', size=6, face_color='red')
    viewer.add_points(things_to_plot[2], name='fn', size=6, face_color='yellow')
    viewer.add_points(things_to_plot[3], name='tp groundtruth', size=6, face_color='blue')
    viewer.add_points(things_to_plot[4], name='tp prediction', size=6, face_color='green')

    return tp, fp, fn, loc_errors


x = './dataset/fiddlercrab_corneas/whole/test_images_10/dampieri_male_16-image.nii'
y = './dataset/fiddlercrab_corneas/whole/test_labels_10/dampieri_male_16-corneas.csv'
y_hat = './output/dampieri_male_16-image.zoo_fiddlercrab_corneas_version_4_checkpoints_last_prediction.nii'
evaluate(x, y, y_hat)
