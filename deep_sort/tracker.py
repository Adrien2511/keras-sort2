from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    # metric permet d'avoir une mesure de distance
    # max_age = nombre maximum d'echec avec la supression d'un trackage
    # n_init est le nombre de détections avant le trackage
    # filtre Kalaman afin de suivre les trajectoires
    # track est l'ensemble des éléments qui sont tracké


    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3): # constructeur qui permet d'enregistrer les différente valeurs
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = [] # création de l'ensemble d'un vecteur pour enregistrer tout les trackages
        self._next_id = 1

    def predict(self): #cette fonction est appelée avant chaque mise à jour du tracking

        for track in self.tracks: #pour tout les éléments qui sont en train d'être tracker
            track.predict(self.kf) # application de la prediction sur chaque élément

    def update(self, detections):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # detection est list d'objet de la classe detection
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets, track_indices,detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()] # enregistrement  des objet où le trackage est confirmé
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()] # enregistrement  des objet où le trackage est non-confirmé

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age,self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah()) #permet d'obtenir le vecteurs de coordonées de la box avec la même dimention que la matrice de covariance
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age,detection.feature))
        #ajout au vecteur tracks l'élément de la class track avec ses différente donnée
        self._next_id += 1 # augmente la valeur de l'id pour la prochaine détection