# Restores data from file
import os
import os.path

import logging
import warnings
import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import math

from utils import add_global_ids


log = logging.getLogger(__name__)


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')


def recover_data(*, paths: List,
                 save_file_prefix: str="treated",
                 cells_suffix: str="cells",
                 bonds_suffix: str="bonds",
                 cells__local_id_bonds_split=' '
                ):
    cells = pd.read_csv(os.path.join(paths[0], save_file_prefix + f"_{cells_suffix}.csv"))
    bonds = pd.read_csv(os.path.join(paths[0], save_file_prefix + f"_{bonds_suffix}.csv"))

    cells = cells.drop(['Unnamed: 0', 'index'], axis=1)
    bonds = bonds.drop(['Unnamed: 0', 'index'], axis=1)
    
    def split_str_to_list_int(coords):
        x = []
        if type(coords) is np.ndarray:
            return coords
        if type(coords) is str:
            for _x in coords[1:-1].split(' '):
                if _x == '':
                    continue
                x.append(_x)

            return np.array(x, dtype=int)
        if type(coords) is float and np.isnan(coords):
            return np.array([], dtype=int)
        raise RuntimeError("Cannot split vertex coords of cell of type {} ({})"
            "".format(type(coords), coords))

    cells['vx_coords_x_cells'] = cells['vx_coords_x_cells'].apply(split_str_to_list_int)
    cells['vx_coords_y_cells'] = cells['vx_coords_y_cells'].apply(split_str_to_list_int)
    if 'next_HC_neighbors' in cells.columns:
        cells['next_HC_neighbors'] = cells['next_HC_neighbors'].apply(split_str_to_list_int)

    for path in paths[1:]:
        cs = pd.read_csv(os.path.join(path, save_file_prefix + "_cells.csv"))
        cs = cs.drop(['Unnamed: 0', 'index'], axis=1)
        cells = pd.concat([cells, cs], axis=0, join='outer')

        bs = pd.read_csv(os.path.join(path, save_file_prefix + "_bonds.csv"))
        bs = bs.drop(['Unnamed: 0', 'index'], axis=1)
        bonds = pd.concat([bonds, bs], axis=0, join='outer')

    cells = cells.reset_index()
    bonds = bonds.reset_index()

    cells, bonds = add_global_ids(cells=cells, bonds=bonds, cells__local_id_bonds_split=cells__local_id_bonds_split)



    cells['SI_position_long'] = cells['SI_position'].where(cells['SI_position'] != 'S', 'Superior')
    cells['SI_position_long'] = cells['SI_position_long'].where(cells['SI_position'] != 'None', 'Mid')
    cells['SI_position_long'] = cells['SI_position_long'].where(cells['SI_position'] != 'I', 'Inferior')

    bonds['SI_position_long'] = bonds['SI_position'].where(bonds['SI_position'] != 'S', 'Superior')
    bonds['SI_position_long'] = bonds['SI_position_long'].where(bonds['SI_position'] != 'None', 'Mid')
    bonds['SI_position_long'] = bonds['SI_position_long'].where(bonds['SI_position'] != 'I', 'Inferior')


    cells['Stage'] = cells['stage']
    cells['S-I position'] = cells['SI_position_long']
    cells['P-D position'] = cells['PD_position']

    bonds['Stage'] = bonds['stage']
    bonds['S-I position'] = bonds['SI_position_long']
    bonds['P-D position'] = bonds['PD_position']
    

    # set correct ordering
    ordering = {
        'E6': 0,
        'E7': 1,
        'E8': 2,
        'E9': 3,
        'E10': 4,
        'E11': 5,
        'E12': 6,
        'E13': 7,
        'E14': 8,
        'E15': 9,
        'E16': 10,
        'E17': 11,

        '0': 0,
        '25S': 1,
        '25I': 2,
        '50S': 3,
        '50I': 4,
        '75S': 5,
        '75I': 6,
        '100': 7,

        'Superior': 0,
        'Mid': 1,
        'Inferior': 2,
    }


    cells.sort_values(by=['stage', 'position', 'file_id'], key=lambda x: x.map(ordering),
                      inplace=True)
    bonds.sort_values(by=['stage', 'position', 'file_id'], key=lambda x: x.map(ordering),
                      inplace=True)

    
    filenames = cells['filename'].unique()
    filepaths = cells['filepath'].unique()
    for id_path, path in enumerate(filepaths):
        for id, f in enumerate(np.unique(cells.loc[cells['filepath'] == path, 'filename'])):
            mask = np.where(
                cells['filename']==f,
                cells['filepath']==path,
                False
            )
            cells.loc[mask, 'file_id'] = id + id_path * len(filenames)
            if cells.loc[mask].empty:
                raise RuntimeError("No cells at {} ({})".format(f, path))

            local_ids = cells.loc[mask, 'local_id_cells']
            if len(local_ids.unique()) != len(local_ids):
                print("ERROR")
                print(cells.loc[mask])
                raise RuntimeError("Failed to identify a frame. Found duplicate cells.")

            mask = np.where(
                bonds['filename']==f,
                bonds['filepath']==path,
                False
            )
            bonds.loc[mask, 'file_id'] = id + id_path * len(filenames)
            if bonds.loc[mask].empty:
                raise RuntimeError("No bonds at {} ({})".format(f, path))


    print("Gathered {} cells, of which {} HC at stages {} and positions {} "
          "from {} files: \n {}"
            "".format(cells.shape[0],
                      cells.loc[cells['is_HC']].shape[0],
                      cells['stage'].unique(),
                      cells['position'].unique(),
                      len(cells['filename'].unique()),
                      cells['filename'].unique()))

    return cells, bonds