import os
import sys, importlib

import logging
import warnings
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import xarray as xr
import pandas as pd


# This are important functionsthat are usefull in multiple places

def add_global_ids(*, cells, bonds, track_id_cells=None, track_id_bonds=None,
                   cells__local_id_bonds_split=None):
    # Identify files by counter
    filenames = cells['filename'].unique()
    for id, f in enumerate(filenames):
        cells.loc[cells['filename']==f, 'file_id'] = id
        bonds.loc[bonds['filename']==f, 'file_id'] = id

    cells['file_id'] = cells['file_id'].astype(int)
    bonds['file_id'] = bonds['file_id'].astype(int)

    cells['local_id_cells'].astype(int)
    bonds['local_id_bonds'].astype(int)

    max_cell_id = cells['local_id_cells'].max()
    max_bond_id = bonds['local_id_bonds'].max()

    cells['global_id_cells'] = (  cells['local_id_cells']
                                + max_cell_id * cells['file_id']
                               ).astype(int)
    bonds['global_id_bonds'] = (  bonds['local_id_bonds']
                                + max_bond_id * bonds['file_id']
                               ).astype(int)

    if 'neighbors_local_ids' in cells.columns:
        cells['neighbors_local_ids'] = cells.apply(
            lambda cell:
                np.array([nbs.strip() 
                          for nbs in cell['neighbors_local_ids'][1:-1].split(" ")
                          if nbs],
                         dtype=int),
                axis=1)

        cells['neighbors'] = cells['neighbors_local_ids'] + max_cell_id * cells['file_id']
    if 'hair_neighbors_local_ids' in cells.columns:
        cells['hair_neighbors_local_ids'] = cells.apply(
            lambda cell:
                np.array([nbs.strip() 
                          for nbs in cell['hair_neighbors_local_ids'][1:-1].split(" ")
                          if nbs],
                         dtype=int),
                axis=1)
        cells['hair_neighbors'] = cells['hair_neighbors_local_ids'] + max_cell_id * cells['file_id']

    if cells__local_id_bonds_split is not None:
        cells['local_id_bonds'] = cells['local_id_bonds'].apply(
            lambda bonds: [int(b.strip()) for b in bonds[1:-1].split(cells__local_id_bonds_split) if b]
        )

    cells['global_id_bonds'] = cells.apply(
        lambda cell, *, max_bond_id: (  np.array(cell['local_id_bonds']).astype(int)
                                      + max_bond_id * cell['file_id']),
        max_bond_id=max_bond_id,
        axis=1
    )


    if 'track_id_cells' in cells.columns:
        for i, fpath in enumerate(cells['filepath'].unique()):
            cells.loc[cells['filepath'] == fpath, 'global_track_id_cells'] = (
                cells['track_id_cells']
                + cells['track_id_cells'].max() * i
            )
        cells['global_track_id_cells'] = cells['global_track_id_cells'].astype(int)
    if 'track_id_bonds' in bonds.columns:
        for i, fpath in enumerate(bonds['filepath'].unique()):
            bonds.loc[bonds['filepath'] == fpath, 'global_track_id_bonds'] = (
                bonds['track_id_bonds']
                + bonds['track_id_bonds'].max() * i
            ).astype(int)
        bonds['global_track_id_bonds'] = bonds['global_track_id_bonds'].astype(int)


    # if track_id_bonds is not None:
    #     raise RuntimeError("Tracking not implemented")

    #     track_id_bonds['global_id_bonds'] = (
    #           track_id_bonds['local_id_bonds']
    #         + max_bond_id * track_id_bonds['frame_nb']
    #         + bond_id_offset)

    #     global_track_id_bonds = track_id_bonds.groupby(by='track_id_bonds').apply(
    #         lambda t: t.iloc[0]['global_id_bonds'])
    #     track_id_bonds['_track_id_bonds'] = track_id_bonds['track_id_bonds']
    #     track_id_bonds['track_id_bonds'] = track_id_bonds.apply(
    #         lambda tbond, *, track_id: track_id[tbond['track_id_bonds']],
    #         track_id = global_track_id_bonds,
    #         axis=1)
    #     global_track_id_bonds = None

    

    bonds['global_cell_id_around_bond1'] = (  bonds['cell_id_around_bond1']
                                            + max_cell_id * bonds['file_id']
                                           )
    bonds['global_cell_id_around_bond2'] = (  bonds['cell_id_around_bond2']
                                            + max_cell_id * bonds['file_id']
                                           )
    

    return cells, bonds

def __circular_sin_cos(d, *, eulerian=True):
    if d.empty:
        return np.asarray([]), np.asarray([])
    elif eulerian:
        x = d.iloc[:, 0]
        y = d.iloc[:, 1]
    elif len(d.shape) == 1:
        x = np.cos(d)
        y = np.sin(d)
    elif d.shape[1] == 1:
        x = np.cos(d.iloc[:, 0])
        y = np.sin(d.iloc[:, 0])
    elif d.shape[1] == 2:
        x = d.iloc[:, 0] * np.cos(d.iloc[:, 1])
        y = d.iloc[:, 0] * np.sin(d.iloc[:, 1])
    else:
        raise RuntimeError("Unknown format of d in circular_mean!")

    return x, y

def __circular_sin_cos_mean(d, *, eulerian=True, normalize=False):
    x, y = __circular_sin_cos(d, eulerian=eulerian)

    if normalize:
        norm = ((x**2 + y**2)**0.5).sum()
        mean_x = x.sum() / norm
        mean_y = y.sum() / norm
    else:
        mean_x = x.mean()
        mean_y = y.mean()

    if x.size == 0:
        return np.nan, np.nan

    return mean_x, mean_y

def circular_mean(d, *, eulerian=True):
    mean_x, mean_y = __circular_sin_cos_mean(d, eulerian=eulerian)
    if np.isnan(mean_x) or np.isnan(mean_y):
        return np.nan
    return np.arctan2(mean_y, mean_x)

def circular_mean_length(d, *, eulerian=True, normalize=False):
    mean_x, mean_y = __circular_sin_cos_mean(d, eulerian=eulerian, normalize=normalize)
    if np.isnan(mean_x) or np.isnan(mean_y):
        return np.nan
    return (mean_x**2 + mean_y**2)**0.5

def circular_variance(d, *, eulerian=True):
    return 1. - circular_mean_length(d, eulerian=eulerian, normalize=True)

def circular_stddev(d, *, eulerian=True):
    return (-2. * np.log(circular_mean_length(d, eulerian=eulerian, normalize=True)))**0.5