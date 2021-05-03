class TrackState:


    #Enumeration des piste pour le trackage provisoire, quand il y a suffisament d'élément elles sont dans état confirmé
    # et quand elles ne sont plus confirmé elles sont dans l'état supprimé

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """
    # mean : est un vecteur de l'état initial [(x,y) au centre, largeur/hauteur , hauteur ]
    # covariance : matrice de covariance du vecteur mean
    # track_id : id de chaque detection
    # n_init : nombre de détections avant que le trackage soit confirmé
    # max_age : nombre maximum d'échecs avant de passer dans l'état supprimé
    # hits : nombre total d'updates
    # age : nombre total d'image depuis le début
    # time_since_update : nombre total d'image aprés la derniére mise à jour
    # state : état actuelle
    # features :
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None): #constructeur de la classe
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self): #permet de transformer les coordonée de la box en : haut gauche (x,y) , largeur hauteur

        ret = self.mean[:4].copy() #création d'une copie de mean des indices de 0 à 3

        ret[2] *= ret[3]  #multiplie l'élément d'indice 2 par celui d'indice 3
        ret[:2] -= ret[2:] / 2 #permet d'obtenir la position en haut à gauche en enlevant la moitié de la hauteur et de la largeur du point au centre
        return ret

    def to_tlbr(self): #permet de transformer les coordonée de la box en : (min x, min y, max x, max y)

        ret = self.to_tlwh() #appelle de la fonction to_tlwh
        ret[2:] = ret[:2] + ret[2:]  #permet d'obtenir les éléments en position max en addition la largeur et la hauteur
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1 #augmente le nombre d'image utiliser
        self.time_since_update += 1 #augmenter le nombre d'image utiliser aprés la derniére mise à jour

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature) #ajout de l'élément detection

        self.hits += 1   #augmente le nombre d'uptade
        self.time_since_update = 0 #remise à 0 du nombre d'image utiliser aprés chaque mise à jour
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            #vérifie si l'état est en tentative de detection et si le nombre d'uptade est supérieur au nombre initial
            self.state = TrackState.Confirmed  #change l'état en confirmé

    def mark_missed(self):

        #marque l'objet si le trackage est raté
        if self.state == TrackState.Tentative: #vérifier si l'état est en tentavie de trackage
            self.state = TrackState.Deleted  #passe dans l'état supprimé si c'est vrai
        elif self.time_since_update > self._max_age: #regarde si le nombre total d'image est au dessus du seuil d'échec
            self.state = TrackState.Deleted #passe dans l'état supprimé si c'est vrai

    def is_tentative(self):
        #return true si l'objet à une possibilté dêtre traqué ==> state == 1
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        # return tue si l'objet est tracké ==> state == 2
        return self.state == TrackState.Confirmed

    def is_deleted(self):

        #returne true si le trackage de l'objet est fini est qu'il doit être supprimé ==> state == 3
        return self.state == TrackState.Deleted