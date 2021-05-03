import numpy as np
import scipy.linalg


"""
Tableau pour le quantile 0,95 de la distribution du chi carré avec N degrés de
liberté (contient des valeurs pour N = 1, ..., 9). Tiré du chi2inv de MATLAB / Octave
fonction et utilisé comme seuil de déclenchement Mahalanobis.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    #Un filtre de Kalman permet de  suivre les boxes sur image.
    #En utilisant  8 dimensions x, y, a, h, vx, vy, va, vh
    #centre du cadre (x, y)
    # a = largeur/hauteur
    # h = la hauteur h,
    # et leurs vitesses respectives.
    # la vitesse du modèle est considéré comme constante.


    def __init__(self):     #constructeur de la classe
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # création des matrices du model Kalman filter
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # matrice de taille (8,8) avec des 1 sur la diagonale
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)    # # matrice de taille (4,8) avec des 1 sur la diagonale


        # controle de l'incertitude de mouvement d'observation
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement #coordonées de la box centre (x,y), largeur/hautuer et hateur
        mean_vel = np.zeros_like(mean_pos) #création d'un vecteur de zero de la taille du vecteur de coordonée
        mean = np.r_[mean_pos, mean_vel] #ajoute les 2 vecteurs à la suite

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std)) #création d'une matrice avec les élément de std sur la diagonal
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        # mean : vecteur de la box de dimension 8 à la position précédente
        # covariance : matrice de dimension 8X8 de la box à la position précédente
        # return : le vecteur mean et la matrice de covariance à la position prédite
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # matrice avec comme diagonale les éléments de std_pos et std_vel au carré

        mean = np.dot(self._motion_mat, mean) # calcul le nouveau vecteur mean
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov # calcul de la nouvelle matrice de covariance

        return mean, covariance

    def project(self, mean, covariance):
        # mean vecteur de dimention 8 avec les coodonées de la box
        # covariance : matrice de dimention (8x8)
        # cette fonction permet de retourner la projection du vecteur mean et de la matrice de covariance sur l'estimation
        """
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std)) #crétion d'une matrice avec les éléments de std sur la diagonal

        mean = np.dot(self._update_mat, mean)   # multiplie _update_mat et mean
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) # calcul de la matice de covariance
        # on multiplie les 3 matrices ensemble
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        #mean :
        #covariance :
        #measurement : vecteur de dimension 3 avec les élémenets de la box (x, y, a, h)
        #(x, y) est le centre de la box,
        # a = largeur/hauteur
        # h = hauteur
        projected_mean, projected_cov = self.project(mean, covariance) # calcul de la projection du vecteur mean et de la matrice de covariance

        chol_factor, lower = scipy.linalg.cho_factor( projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T,check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        # mean vecteur de dimention 8
        # covariance : matrice de dimension (8x8)
        # measurements : matrice de dimension Nx4 avec les coodonées x,y du centre , rapport largeur/hauteur et hauteur
        # only_position : si true  le calcul de la distance se fait par rapport à la position centrale de la boîte uniquement.
        # return : un vecteur de longueur N, où le i-ème élément contient le
        # distance de Mahalanobis au carré entre (moyenne, covariance) et
        # `mesures [i]`.
        mean, covariance = self.project(mean, covariance) #utilisation de la fonction project
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)# Renvoie la décomposition de Cholesky
        d = measurements - mean # calcul la différence entre measurements et mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False,overwrite_b=True)
        # résou l'équation cholesky_factor*x=d.T pour x , en supposant que cholesky_factor est une matrice triangulaire
        squared_maha = np.sum(z * z, axis=0) # fait la somme du vecteur z * z
        return squared_maha