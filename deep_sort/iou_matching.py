from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):

    # bbox : vecteur de la box en format (top left x, top left y, largeur,hauteur)
    # candidates : matrice dont les colonnes sont les différente box
    # return un vecteur avec des éléments entre 0 et 1 qui représente le taux de superposition des candidats sur la bbox
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:] # enregistrement des coordonées du coin top left et le coin right bottom
    candidates_tl = candidates[:, :2]   # enregistrement des coordonées top left de toute les boxs
    candidates_br = candidates[:, :2] + candidates[:, 2:] # enregistrement des coordonées roght bottom de toutes les boxs


    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]] #création d'un vecteur composé des coordonées top left les plus grands entre la bbox et les canditats
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]] #création d'un vecteur composé des coordonées bottom right les plus petites  entre la bbox et les canditats
    wh = np.maximum(0., br - tl) #enregistre un vecteur de  la différence entre le point top left et bottom right , ou 0 si la valeur est négative

    area_intersection = wh.prod(axis=1) # calcul la surface de l'intesection de la box et des candidats
    area_bbox = bbox[2:].prod()  # calcul de la surface de la box (largeur * hauteur)
    area_candidates = candidates[:, 2:].prod(axis=1) #calcul la surface des candidats
    return area_intersection / (area_bbox + area_candidates - area_intersection)  # le taux de superposition des candidats sur la bbox


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):


    # tracks : list des éléments tracké
    # detections : les des éléments détecter
    # track_indices : list des indides des éléments tracké
    # detection_indeces : list des indices des éléments detecté
    # return : Renvoie une matrice de coût de forme
    # len (track_indices), len (detection_indices) où l'entrée (i, j) est
    # `1 - iou (tracks [track_indices [i]], detections [detection_indices [j]])`.
    if track_indices is None:
        track_indices = np.arange(len(tracks)) # enregistrement des éléments tracké
    if detection_indices is None:
        detection_indices = np.arange(len(detections)) # enregistrement des éléments detecté

    cost_matrix = np.zeros((len(track_indices), len(detection_indices))) #création d'une matrice de zero de la taille (élément tracké , élément detecté)
    for row, track_idx in enumerate(track_indices): # pour chaque indece
        if tracks[track_idx].time_since_update > 1: # si le time_since_uptade  de l'élément tracker à l'indice track_idx est supérieur à 1
            cost_matrix[row, :] = linear_assignment.INFTY_COST # prends comme valeur INFTY_COST
            continue   # recommence au début de la boucle

        bbox = tracks[track_idx].to_tlwh() # enregistrement des coodonées de la box (haut gauche (x,y) , largeur hautuer)
        candidates = np.asarray([detections[i].tlwh for i in detection_indices]) # enregistrement des coodonées de tout les candidats (haut gauche (x,y) , largeur hautuer)
        cost_matrix[row, :] = 1. - iou(bbox, candidates) # calcul la matrice de 1 - la superpositon des canditas sur la bbox
    return cost_matrix