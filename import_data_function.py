# Imports data from tissue analyser and performs some further analysis
# The data is exported to excel sheets

import os
import os.path
import argparse

import logging
import warnings
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

import math
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns

import yaml
from PIL import Image
from PIL.TiffTags import TAGS

from utils import *

def plot_frame(
    cells,
    bonds, *,
    filepath: str, 
    filename: str,
    load_images: str=None,
    load_segmentation_path: str=None,
    save_to_images: str,
    plot_polygons=False,
    log
):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(filename)

    frame = cells.loc[cells['filepath'] == filepath]
    frame = frame.loc[frame['filename'] == filename]
    frame_bonds = bonds.loc[bonds['file_id'] == frame['file_id'].unique()[0]]

    # Get the original image
    if load_images is not None:
        image_path = os.path.join(filepath, load_images, filename)
        log.debug("   Plotting %s ...", image_path)
    else:
        image_path = None
        
    if image_path is not None and os.path.isfile(image_path):
        _im = plt.imread(image_path)

        with Image.open(image_path) as img:
            meta_dict = {TAGS[key] : img.tag[key][0] for key in img.tag_v2}
            meta_dict['resolution_scale'] = 'micron'
            meta_dict['resolution_x'] = meta_dict.get('XResolution', [np.nan])[0] / 1.e6
            meta_dict['resolution_y'] = meta_dict.get('YResolution', [np.nan])[0] / 1.e6
        
        seg_path = os.path.join(
            filepath,
            load_segmentation_path,
            filename.split('.')[0], 'handCorrection.tif'
        )
        if plot_polygons:
            color = np.where(frame_bonds['type'] == 'SS', 'gray', 'black')
            color = np.where(frame_bonds['type'] == 'HS', 'orange', color)
            color = np.where(frame_bonds['type'] == 'HH', 'red', color)
            color = np.where(frame_bonds['type'] == 'border', 'skyblue', color)
            color = np.where(frame_bonds['type'] == 'border_plus_one',
                            'deepskyblue',
                            color)
            ax.quiver(frame_bonds['vx_1_x'],
                    frame_bonds['vx_1_y'],
                    frame_bonds['vx_2_x'] - frame_bonds['vx_1_x'],
                    frame_bonds['vx_2_y'] - frame_bonds['vx_1_y'],
                    color=color,
                    scale_units='xy', angles='xy', scale=1,
                    headwidth=0, headlength=0,headaxislength=0)
        if os.path.isfile(seg_path):
            _seg = plt.imread(seg_path)
            if (len(_seg.shape) < len(_im.shape)):
                _seg = np.expand_dims(_seg, axis=len(_im.shape)-1)
            _im = np.where(_seg==255, 255, _im)
        else:
            log.info('NO segmentation found at {}'.format(seg_path))
        im = ax.imshow(_im, interpolation='none')
    else:
        log.warning("Couldn't find raw image at {}. Using voronoi tesselation."
                    "".format(image_path))
        min_x = 0
        min_y = 0
        max_x = frame['center_x_cells'].max()
        max_y = frame['center_y_cells'].max()

        Nx = 200
        Ny = 200
        grid_x, grid_y = np.mgrid[min_x:max_x:Nx*1j, min_y:max_y:Ny*1j]

        data = np.where(frame['is_HC'], 0., frame['local_id_cells'] % 8 + 1)
        grid_z1 = griddata((frame['center_x_cells'],
                            frame['center_y_cells']),
                            data, (grid_x, grid_y), method='nearest')
        im = ax.imshow(grid_z1.T, extent=(min_x, max_x, min_y, max_y),
                        origin='lower', cmap=plt.get_cmap("Set1"))
        ax.invert_yaxis()

        cbar = fig.colorbar(im, extend='both')
        cbar.set_label(label=property)
        cbar.minorticks_on()


    # Scatter the HC
    x = np.where(frame['is_HC'], frame['center_x_cells'], np.nan)
    y = np.where(frame['is_HC'], frame['center_y_cells'], np.nan)
    ax.scatter(x, y, color='blue', s=1.)

    # overlay polarity
    if 'cilium_rho' in frame.columns:
            r = frame['cilium_rho']
            pol_x = r * np.cos(frame['cilium_phi'])
            pol_y = r * np.sin(frame['cilium_phi'])
            ax.quiver(frame['center_x_cells'], frame['center_y_cells'],
                        pol_x, pol_y, angles='xy', scale_units='xy',
                        scale=1.)
    else:
            log.info("No cilia data for image {}", filename)


    
    # Scatter the ppMLC
    x = frame_bonds['ppMLC_foci_X']
    y = frame_bonds['ppMLC_foci_Y']
    foci_size = (  frame_bonds['ppMLC_foci_Total']
                / frame_bonds['ppMLC_foci_Total'].max())
    ax.scatter(x, y, color='red', s=20*foci_size, alpha=0.4)

    ax.quiver(x, y,
            (  (frame_bonds['vx_2_x'] + frame_bonds['vx_1_x']) / 2.
            - frame_bonds['ppMLC_foci_X']),
            (  (frame_bonds['vx_2_y'] + frame_bonds['vx_1_y']) / 2.
            - frame_bonds['ppMLC_foci_Y']),
            color='red',
            scale_units='xy', angles='xy', scale=1
            )

    _save_path = os.path.join(
        filepath,
        save_to_images,
        filename.split('.')[0] + '.png'
    )
    plt.savefig(_save_path, dpi=300)
    plt.close()


### --- IMPORT OF RAW DATA ---------------------------------------------------
def import_data(
    config_file: str,
    log
):
    with open(config_file, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise 


    data_basedir = os.path.abspath(parsed_yaml["data_basedir"])# data/HCA-Actn4/E14"
    data_path = parsed_yaml["data_path"]


    # Where to save the database
    save_treated_data = parsed_yaml["save_treated_data"]
    save_file_prefix = parsed_yaml["save_file_prefix"]

    # whether to create a composed image will all information
    save_to_images = parsed_yaml["save_to_images"]
    load_images = parsed_yaml["load_images"]
    load_segmentation_path = parsed_yaml["load_segmentation_path"]


    # Precise labels of raw data
    ## files all labelled path/PrefixSuffix.csv
    files_prefix = parsed_yaml["files_prefix"]

    ## The suffix of raw data of the cells
    cells_suffix = parsed_yaml["cells_suffix"]

    ## The suffix of raw data of the bonds
    bonds_suffix = parsed_yaml["bonds_suffix"]


    ## The suffix of HC identifier images
    ## If provided, HC are the cells that contain HCA > 0
    ## This file must contain the equivalent of cells, but where the HCA channel 
    ## only contains the identifier of HC (e.g. the max of HCA)
    ## NOTE ensure that it is 0 in non-HC
    HCA_suffix = parsed_yaml["HCA_suffix"]

    # which column of the data to use for HC identifier
    # is HC, if > 0
    HCA_channel = parsed_yaml["HCA_channel"]


    # Split image filenames and associate in the following order
    # e.g. prefix_E10_25S_13.tif -> stage=E10, position=25S, sample_id=13
    split_filename = parsed_yaml["split_filename"]


    ## The tracking ids
    is_movie = parsed_yaml["is_movie"]
    ## Ids associated with cells throughout an image serie
    ## Assumes that a folder contains a series of consecutive images
    # tracked_cells_suffix = "tracked_cells"
    # tracked_bonds_suffix = "tracked_bonds"
    tracked_cells_suffix = parsed_yaml["tracked_cells_suffix"]
    tracked_bonds_suffix = parsed_yaml["tracked_bonds_suffix"]
    # NOTE HC identified if any of the images has HCA > 0

    ## Thetime lag between images
    time_frame_column = parsed_yaml["time_frame_column"]
    time_step = parsed_yaml["time_step"]
    time_step_units = parsed_yaml["time_step_units"]


    # Additional information
    import_cilia = parsed_yaml.get("import_cilia")
    if import_cilia is None:
        import_cilia = {}
    import_actin_foci = parsed_yaml.get("import_actin_foci")
    if import_actin_foci is None:
        import_actin_foci = {}

    ppMLC_foci_path = parsed_yaml["ppMLC_foci_path"]



    # Exclude data from database
    ## NOTE this will speed up calculations significantly
    ## NOTE data will not be available in resulting file

    ## Select a stage
    select_data = parsed_yaml.get('select_data', dict())
    if select_data is None:
        select_data = dict()

    # Whether to exclude border cells from final file
    ## NOTE they will be included in calculations
    exclude_border_cells = parsed_yaml["exclude_border_cells"]


    def import_raw_data(path):
        """Imports the data from Tissue analyser
            Transforms local to global and tracked identities
            Assigns HC
        """
        
        log.info("Loading data at %s", data_basedir)
        start_time = time.time()


        # gather the filepaths
        filepath = os.path.join(path, files_prefix)

        def get_filepath(suffix):
            if suffix is None:
                return None
            return "{}{}.csv".format(filepath, suffix)

        cells_filepath = get_filepath(cells_suffix)
        HCA_filepath = get_filepath(HCA_suffix)
        bonds_filepath = get_filepath(bonds_suffix)
        if is_movie:
            tracked_cells_filepath = get_filepath(tracked_cells_suffix)
            tracked_bonds_filepath = get_filepath(tracked_bonds_suffix)
        else:
            tracked_cells_filepath = None
            tracked_bonds_filepath = None


        def read_csv(filepath, **k):
            """How to import csv files"""
            if filepath is None:
                return None
            return pd.read_csv(filepath, **k)

        cells = read_csv(cells_filepath, sep='\t')
        HCA = read_csv(HCA_filepath, sep='\t')
        bonds = read_csv(bonds_filepath, sep='\t')
        track_id_cells = read_csv(tracked_cells_filepath, sep='\t')
        track_id_bonds = read_csv(tracked_bonds_filepath, sep='\t')

        cells['filepath'] = data_basedir
        bonds['filepath'] = data_basedir

        # check that data available
        if cells is None:
            raise RuntimeError("No cell data has been provided at {}!"
                            "".format(cells_filepath))
        if bonds is None:
            raise RuntimeError("No bond data has been provided at {}!"
                            "".format(bonds_filepath))

        if HCA is None:
            log.info("No seperate HC data provided, using data from cells instead.")
            HCA = cells


        # check for duplicates
        def check_no_duplicates(data, *, name: str, label: str,
                                groupby: str='frame_nb'):
            if data is None:
                return

            frames = data['frame_nb'].unique()

            if groupby is not None:
                data = data.groupby(by=groupby)

            filenames = data.apply(lambda d: d['filename'].unique())
            if (filenames.apply(lambda f: len(f) != 1).any()):
                raise RuntimeError("Found duplicates in {} data!"
                                "".format(name))
                        
            duplicated = data.apply(lambda d: d[label].duplicated())
            if (duplicated.any().any()):
                raise RuntimeError("Found duplicates in {} data!"
                                "".format(name))

            if len(frames) != len(filenames):
                raise RuntimeError("Frames ids are not unique!")


        check_no_duplicates(cells, name='cells', label='local_id_cells')
        check_no_duplicates(HCA, name='HCA', label='local_id_cells')
        check_no_duplicates(bonds, name='bonds', label='local_id_bonds')
        check_no_duplicates(track_id_cells, name='track_id_cells',
                            label='local_id_cells')
        check_no_duplicates(track_id_bonds, name='track_id_bonds',
                            label='local_id_bonds')


        # ensure same sorting
        def sort_data(data, *a, **k):
            """Sort data and reset index.
            Args and kwargs forwarded to data.sort_values
            """
            if data is None:
                return None
            data = data.sort_values(*a, **k)
            data = data.reset_index()
            return data

        cells = sort_data(cells, by=['filename', 'local_id_cells'])
        HCA = sort_data(HCA, by=['filename', 'local_id_cells'])
        bonds = sort_data(bonds, by=['filename', 'local_id_bonds'])
        track_id_cells = sort_data(track_id_cells,
                                by=['filename', 'local_id_cells'])
        track_id_bonds = sort_data(track_id_bonds,
                                by=['filename', 'local_id_bonds'])


        # verify match of HCA and cells
        if cells.shape[0] != HCA.shape[0]:
            raise RuntimeError("Cells and HC don't have same shape: {} != {}"
                                "".format(cells.shape[0], HCA.shape[0]))
        for (name0, group0), (name1, group1) in zip(cells.groupby('filename'),
                                                    HCA.groupby('filename')):
            if name0 != name1:
                raise RuntimeError("Filenames of cells do not match those of HCA!")
            if group0.shape[0] != group1.shape[0]:
                raise RuntimeError("Cells and HCA don't have same shape in file "
                                    "'{}': {} != {}"
                                    "".format(name0,
                                            group0.shape[0],
                                            group1.shape[0]))
        
        # verify match of bonds and cells
        for (name0, group0), (name1, group1) in zip(cells.groupby('filename'),
                                                    bonds.groupby('filename')):
            if name0 != name1:
                raise RuntimeError("Filenames of cells do not match those of "
                                "bonds!")

        # verify match of tracked cells and cells
        if track_id_cells is not None:
            if cells.shape[0] != track_id_cells.shape[0]:
                raise RuntimeError("Cells and tracked cells don't have same shape: "
                                "{} != {}"
                                "".format(cells.shape[0],
                                            track_id_cells.shape[0]))
            for (name0, group0), (name1, group1) in zip(cells.groupby('filename'),
                                                        track_id_cells.groupby(
                                                            'filename')):
                if name0 != name1:
                    raise RuntimeError("Filenames of cells do not match those of "
                                    "tracked cells!")
                if group0.shape[0] != group1.shape[0]:
                    raise RuntimeError("Cells and tracked cells don't have same "
                                    "shape in file '{}': {} != {}"
                                    "".format(name0,
                                                group0.shape[0],
                                                group1.shape[0]))

        # verify match of tracked bonds and bonds
        if track_id_bonds is not None:
            for (name0, group0), (name1, group1) in zip(bonds.groupby('filename'),
                                                        track_id_bonds.groupby(
                                                                    'filename')):
                if name0 != name1:
                    raise RuntimeError("Filenames of cells do not match those of "
                                    "tracked bonds!")


        # Decider on HC identity
        log.info("Using channel {} for HCA criterion".format(HCA_channel))
        cells['HCA_criterion'] = HCA[HCA_channel]
        cells['is_HC'] = cells['HCA_criterion'] >= 1.
        log.info("Detected %.2f perc. HC",
                cells['is_HC'].sum() / cells['is_HC'].count() * 100)
        HCA = None
        
        if track_id_cells is None:
            log.info("HC density per file: \n{}".format(
                cells.groupby('filename').apply(lambda frame: frame[frame['is_HC']]['index'].count() / frame['index'].count())
            ))

        
        # extract information from filename
        # filenames are of structure
        # <label0>_<label1>.<EXT>

        for i, label in enumerate(split_filename):
            cells[label] = cells['filename'].apply(
                lambda f: f.split(".")[0].split("_")[i])
            
            bonds[label] = bonds['filename'].apply(
                lambda f: f.split(".")[0].split("_")[i])


        
        # transform information about type of bonds
        # NOTE are of type [<id0>#<id1>#..]
        cells['local_id_bonds'] = cells['local_id_of_bonds'].apply(
            lambda s: np.array([int(i) for i in s.split("#")]))


        # assign global ids
        max_cell_id = cells['local_id_cells'].max()


        # assign global ids of bonds
        cells, bonds = add_global_ids(cells=cells, bonds=bonds,
                                    track_id_cells=track_id_cells,
                                    track_id_bonds=track_id_bonds)

        # tracking of cells
        if track_id_cells is not None:
            if (cells['local_id_cells'] != track_id_cells['local_id_cells']).any():
                raise RuntimeError("Local ids of cells and tracked cells don't "
                                "match")
            cells['track_id_cells'] = track_id_cells['track_id_cells'].astype(int)
            track_id_cells = None

            # assign minimal track ids
            for i, id in enumerate(cells['track_id_cells'].unique()):
                cells.loc[cells['track_id_cells'] == id, 'track_id_cells'] = i
            
            # A HC must remain a hair cell throughout tracking
            HC = cells.groupby(by='track_id_cells').apply(lambda c: c['is_HC'].any())
            cells['is_HC'] = cells.apply(lambda cell, *, HC: HC[cell['track_id_cells']],
                                        axis=1, HC=HC)

            if time_frame_column in cells.columns:
                cells['frame'] = cells['frame'].astype(int)
                cells['time'] = cells['frame'] * time_step
            else:
                log.warning("Frame identifier '{}' not in cell data"
                    "".format(time_frame_column))

                    
            log.info("HC density per file after cell tracking: ")
            print(cells.groupby('filename').apply(lambda frame: frame[frame['is_HC']]['index'].count() / frame['index'].count()))


        if track_id_bonds is not None:
            for id, f in enumerate(track_id_bonds['filename'].unique()):
                track_id_bonds.loc[track_id_bonds['filename']==f, 'file_id'] = id
            track_id_bonds['file_id'] = track_id_bonds['file_id'].astype(int)
            # join bonds and tracked bonds

            track_id_bonds['global_id_bonds'] = (
                track_id_bonds['local_id_bonds']
                + bonds['local_id_bonds'].max() * track_id_bonds['file_id']
            ).astype(int)
            track_id_bonds = track_id_bonds[['global_id_bonds', 'track_id_bonds']]
            track_id_bonds = track_id_bonds.set_index("global_id_bonds")
            bonds = bonds.join(
                track_id_bonds,
                on="global_id_bonds"
            )
            track_id_bonds = None
            bonds['track_id_bonds'] = np.where(np.isnan(bonds['track_id_bonds']),
                                            bonds['global_id_bonds'],
                                            bonds['track_id_bonds'])
            bonds['track_id_bonds'] = bonds['track_id_bonds'].astype(int)

            # assign minimal track ids
            for i, id in enumerate(bonds['track_id_bonds'].unique()):
                bonds.loc[bonds['track_id_bonds'] == id, 'track_id_bonds'] = i


            if time_frame_column in bonds.columns:
                bonds['frame'] = bonds['frame'].astype(int)
                bonds['time'] = bonds['frame'] * time_step
            else:
                log.warning("Drame identifier '{}' not in bond data"
                    "".format(time_frame_column))

        # To rad
        bonds['bond_orientation'] = bonds['bond_orientation'] / 180 * math.pi

        if 'provide_defaults' in parsed_yaml:
            defaults = parsed_yaml['provide_defaults']
        else:
            defaults = dict()
        for k, v in defaults.items():
            if not k in cells.columns:
                log.info("Setting column {}: `{}' (from default).".format(k, v))
                cells[k] = v
                bonds[k] = v


        log.info("Loaded data at %s in %.2fs.", data_basedir, time.time() - start_time)

        return cells, bonds



    cells, bonds = import_raw_data(os.path.join(
                        data_basedir, data_path
                ))

    cells['sample_id'] = cells['sample_id'].astype(int)
    bonds['sample_id'] = bonds['sample_id'].astype(int)

    log.info("---------")
    log.info("Important Info:")
    log.info("")
    log.info("HCs have an average HCA criterion of: \n{}".format(
        cells.loc[cells['is_HC']].groupby(by='file_id')['HCA_criterion'].mean()
    ))

    if (cells.groupby(by='file_id')['HCA_criterion'].mean().max() > 255 * 10):
        log.error("=======================================")
        log.error("=======================================")
        log.error("===  Unexpected high HC criterion!  ===")
        log.error("=======================================")
        log.error("=======================================")
        log.warning("CHECK average HC CRITERION ABOVE!")
        log.warning("")
        log.warning("Waiting 15 sec before continuing ...")
        time.sleep(15)

    def strip_to_PD(pos: str):
        if pos == 'base':
            return 25
        elif pos == 'mid':
            return 50
        elif pos == 'apex':
            return 75
        return int(pos.strip('I').strip('S'))

    def strip_to_SI(pos: str):
        return pos.replace('0', '').replace('1', '').replace('2', '')\
                .replace('3', '').replace('4', '').replace('5', '')\
                .replace('6', '').replace('7', '').replace('8', '')\
                .replace('9', '')

    cells['PD_position'] = cells['position'].apply(strip_to_PD)
    cells['SI_position'] = cells['position'].apply(strip_to_SI)
    cells['SI_position'] = cells['SI_position'].where(cells['SI_position'] != '',
                                                    'None')
    bonds['PD_position'] = bonds['position'].apply(strip_to_PD)
    bonds['SI_position'] = bonds['position'].apply(strip_to_SI)
    bonds['SI_position'] = bonds['SI_position'].where(bonds['SI_position'] != '', 
                                                    'None')

    for s, f in select_data.items():
        if s not in cells.columns:
            log.warning("Skipping selection at %s = %s because not in data", s, f)
            continue

        log.info("Selecting data at %s = %s ..", s, f)
        cells = cells.loc[cells[s] == f]
        bonds = bonds.loc[bonds[s] == f]

    if 'stage' in cells.columns and 'position' in cells.columns:
        log.info("Gathered %i cells (incl. %i HC) at stages %s and positions %s "
                "from %i files: \n %s",
                cells.shape[0],
                cells.loc[cells['is_HC']].shape[0],
                cells['stage'].unique(),
                cells['position'].unique(),
                len(cells['filename'].unique()),
                cells['filename'].unique())
    elif 'frame' in cells.columns:
        log.info("Gathered {} cells (of which {} are HC) in {} frames"
            "".format(cells.shape[0], cells.loc[cells['is_HC']].shape[0], len(cells['frame'].unique())))
    else:
        log.info("Gathered {} cells, of which {} are HC"
            "".format(cells.shape[0], cells.loc[cells['is_HC']].shape[0]))

    if cells.empty or bonds.empty:
        raise RuntimeError("Cell data is empty. Nothing to process!")
    
    log.info("--------------------")
    log.info("Retrieving meta-data")
    log.info("--------------------")

    for (filepath, filename), frame in cells.groupby(by=['filepath', 'filename']):
        # Get the original image
        image_path = os.path.join(filepath, load_images, filename)

        meta_dict = dict()
        meta_dict['resolution_scale'] = 'pixel_per_micron'
        meta_dict['resolution_x'] = np.nan
        meta_dict['resolution_y'] = np.nan
        if os.path.isfile(image_path):
            log.debug(" Searching %s for meta data ...", image_path)
            with Image.open(image_path) as img:
                for key in img.tag_v2:
                    meta_dict[TAGS[key]] = img.tag[key][0]
                meta_dict['resolution_x'] = meta_dict.get('XResolution', [np.nan])[0] / 1.e6
                meta_dict['resolution_y'] = meta_dict.get('YResolution', [np.nan])[0] / 1.e6

            if np.isnan(meta_dict['resolution_x']) or np.isnan(meta_dict['resolution_x']):
                log.warning("No resolution found for image {}".format(filename))

            else:
                log.debug("   Found image resolution: {:.2f}x{:.2f} {}"
                        "".format(
                            meta_dict['resolution_x'],
                            meta_dict['resolution_y'],
                            meta_dict['resolution_scale'],
                        ))
        
        else:
            log.warning("No resolution found for image '{}'".format(filename))


        cells.loc[cells['filename'] == filename, 'resolution_x'] = meta_dict['resolution_x']
        cells.loc[cells['filename'] == filename, 'resolution_y'] = meta_dict['resolution_y']
        cells.loc[cells['filename'] == filename, 'resolution_scale'] = meta_dict['resolution_scale']
        
        bonds.loc[bonds['filename'] == filename, 'resolution_x'] = meta_dict['resolution_x']
        bonds.loc[bonds['filename'] == filename, 'resolution_y'] = meta_dict['resolution_y']
        bonds.loc[bonds['filename'] == filename, 'resolution_scale'] = meta_dict['resolution_scale']

    cells['area'] = cells['area_cells'] / cells['resolution_x'] / cells['resolution_y']



    ### --- POST-PROCESS DATA ------------------------------------------------------

    log.info("------------------------")
    log.info("Post-processing data ...")
    log.info("------------------------")
    start_time = time.time()


    cells['shape_index'] = (  cells['perimeter_length']
                            / np.sqrt(cells['area_cells']))

    def split_coords_vx(coords):
        x = []
        y = []
        if type(coords) is float:
            return np.array([], dtype=int), np.array([], dtype=int)
        for xy in coords.split('#'):
            _xy = xy.split(':')
            x.append(_xy[0])
            y.append(_xy[1])

        return np.array(x, dtype=int),  np.array(y, dtype=int)

    result = cells['vx_coords_cells'].apply(split_coords_vx)
    cells['vx_coords_x_cells'] = result.apply(lambda r: r[0])
    cells['vx_coords_y_cells'] = result.apply(lambda r: r[1])


    log.debug("Assigning neighbourhood ...")
    _time = time.time()

    # Assign a neighborhood
    def get_neighbors(cell):
        """"Information on neighbors as per bond information"""
        nbs = bonds.loc[   bonds['global_cell_id_around_bond1']
                        == cell['global_id_cells']]\
                ['global_cell_id_around_bond2']
        nbs = pd.concat([nbs,
                        bonds.loc[   bonds['global_cell_id_around_bond2']
                                == cell['global_id_cells']]\
                            ['global_cell_id_around_bond1']],
                        axis=0, join='outer')
        return np.array(nbs).astype(int)

    cells['neighbors'] = cells.apply(get_neighbors, axis=1)
    cells['neighbors_local_ids'] = cells['neighbors'] - (cells['global_id_cells'] - cells['local_id_cells'])
    cells['num_neighbors'] = cells['neighbors'].apply(len)


    def get_hair_neighbors(cell):
        """The neighbors of type HC"""
        h_nbs = [cells.loc[cells['global_id_cells'] == n_id].iloc[0]['is_HC']
                    for n_id in cell['neighbors']]
        h_nbs = np.where(h_nbs, cell['neighbors'], np.nan)
        return h_nbs[~np.isnan(h_nbs)].astype(int)

    cells['hair_neighbors'] = cells.apply(get_hair_neighbors, axis=1)
    cells['hair_neighbors_local_ids'] = cells['hair_neighbors'] - (cells['global_id_cells'] - cells['local_id_cells'])
    cells['num_hair_neighbors'] = cells['hair_neighbors'].apply(len)


    # Assign neighborhoods
    def get_bond_type(bond):
        a = cells.loc[   cells['global_id_cells']
                    == bond['global_cell_id_around_bond1']]
        b = cells.loc[   cells['global_id_cells']
                    == bond['global_cell_id_around_bond2']]
        if a.size == 0 or b.size == 0:
            return 'border'
        
        a = a.iloc[0]
        b = b.iloc[0]

        if a['is_border_cell'] and b['is_border_cell']:
            return 'border_plus_one'
        
        if a['is_HC'] and b['is_HC']:
            return 'HH'
        elif a['is_HC'] or b['is_HC']:
            return 'HS'
        else:
            return 'SS'

    bonds['type'] = bonds.apply(get_bond_type, axis=1)

    log.info("Assigned neighbourhood in %.2f s.", time.time() - _time)

    if not bonds.loc[bonds['type']=='HH'].empty:
        log.info("Number of HC-HC junctions: \n{}".format(bonds.loc[bonds['type']=='HH'].groupby(by=['filename', 'type'])['type'].count()))
    else:
        log.info("No HC-HC junctions segmented.")

    log.info("---")
    # --------------------------

    # Normalize area
    log.debug("Normalizing length scales ...")
    _time = time.time()

    bulk_cells = cells[~cells['is_border_cell']].dropna(how='all')

    normalization_area = bulk_cells.groupby(by=['file_id'])
    normalization_area = bulk_cells.groupby(by=['file_id'])['area_cells'].mean()

    cells['normalized_area_cells'] = cells.apply(
        lambda d, *, normalization_area:
            d['area_cells'] / normalization_area[d['file_id']],
        normalization_area=normalization_area,
        axis=1)


    bulk_cells = cells[~cells['is_border_cell']]
    bulk_HC = bulk_cells.loc[bulk_cells['is_HC']]
    HC_normalized_area = bulk_HC.groupby(by=['file_id'])
    HC_normalized_area = HC_normalized_area['normalized_area_cells'].mean()

    cells['HC_normalized_area'] = cells.apply(
        lambda cell, *, HC_normalized_area:  HC_normalized_area[cell['file_id']],
        HC_normalized_area=HC_normalized_area, axis=1)

    bulk_SC = bulk_cells.loc[~bulk_cells['is_HC']].dropna(how='all')
    SC_normalized_area = bulk_SC.groupby(by=['file_id'])
    SC_normalized_area = SC_normalized_area['normalized_area_cells'].mean()

    cells['SC_normalized_area'] = cells.apply(
        lambda cell, *, SC_normalized_area: SC_normalized_area[cell['file_id']],
        SC_normalized_area=SC_normalized_area, axis=1)


    # Normalize bond length
    bulk_bonds = bonds[bonds['type'] != 'border_plus_one'].dropna(how='all')
    normalization_length = bulk_bonds.groupby(by=['file_id'])
    normalization_length = normalization_length['bond_length_in_px'].mean()
    bonds['normalized_bond_length'] = bonds.apply(
        lambda b, *, normalization:
            b['bond_length_in_px'] / normalization[b['file_id']],
        normalization=normalization_length, axis=1)

    log.info("Normalized length scales in %.2f s.", time.time() - _time)


    log.debug("Calculating hexatic order ..")
    _time = time.time()


    def __elongation(x, y):
        """The elongation of a polygon defined by the set of x and y coordinated.
        
        The coordinates will be sorted anti-clockwise.

        Returns:
            - Elongation tensor Q
            - Area of polygon
            - Angle of rotation
            - absolute of Q tensor
            - the ratio of long to short axis of an ellipse fitting this polygon
        """
        def decompose(mat: np.array):
            """Decomposition of a matrix into trace, symmetric and asymmetric parts
            """
            tr = np.trace(mat)
            sym = 0.5 * (mat + np.transpose(mat)) - tr / 2. * np.eye(2)
            asym = 0.5 * (mat - np.transpose(mat))

            return tr, sym, asym
        
        def rotation(S: np.array):
            tr, sym, asym = decompose(S)

            h = asym + np.eye(2) * tr / 2.
            s = (np.trace(np.matmul(h, np.transpose(h))) / 2.)**0.5
            rot = h / max(s, 1.e-12)

            return rot, np.arctan2(rot[0, 1], rot[0, 0])

        if len(x) < 2:
            raise

        # sort the vertices
        angles = np.arctan2(y, x)
        order = np.argsort(angles)
        x = x[order]
        y = y[order]

        dual_A = 0
        q = np.zeros((2, 2))
        for i in range(len(x)):
            # vertices of a triangle
            ax = 0
            ay = 0

            bx = x[i]
            by = y[i]

            cx = x[(i + 1) % len(x)]
            cy = y[(i + 1) % len(x)]


            # transform to shape tensor
            base_1 = np.array([[bx - ax, cx - bx],
                               [by - ay, cy - by]])
            base_equilateral = np.array([[1.0, -0.5],
                                        [0.0, 3.**0.5 / 2.]])
            
            # the triangle shape in reference to equilateral triangle
            S = np.matmul(base_1, np.linalg.inv(base_equilateral))

            # decompose shape tensor
            tr_S, sym_S, asym_S = decompose(S)

            # rotational component
            rot, theta = rotation(S)

            # area
            area = np.linalg.det(S)

            # shape deformation
            s = (np.trace(np.matmul(sym_S, np.transpose(sym_S))) / 2.)**0.5

            _q = (  np.arcsinh(s / area**0.5) / max(s, 1.e-12)
                  * np.matmul(sym_S, np.transpose(rot)))
            
            dual_A += area
            q += area * _q

        q /= dual_A

        abs_q = (np.trace(np.matmul(q, np.transpose(q))) / 2.)**0.5
        # ratio_WH = np.exp(2 * abs_q * (dual_A / math.pi)**0.5)
        ratio_WH = np.exp(2 * abs_q)
        theta = np.arctan2(q[1, 0], q[0, 0]) / 2.
        
        return q, dual_A, theta, abs_q, ratio_WH 


    def elongation(cell):
        if cell['is_border_cell']:
            return np.nan, np.nan, np.nan

        x = cell['vx_coords_x_cells'] - cell['center_x_cells']
        y = cell['vx_coords_y_cells'] - cell['center_y_cells']

        if len(x) < 2 or len(x) != len(y):
            return np.nan, np.nan, np.nan

        q, a, theta, abs_q, r = __elongation(x, y)

        return abs_q, theta, r

    result = cells.apply(elongation, axis=1)
    cells['elongation_cells'] = result.apply(lambda res: res[0])
    cells['orientation_cells'] = result.apply(lambda res: res[1])
    cells['elongation_ratio_WH_cells'] = result.apply(lambda res: res[2])


    def hexatic_order(cell, *, cells):
        """Calculate the corrected hexatic order"""
        def r_ellipse(theta, a, b, theta0):
            return a * b / (  (b * np.cos(theta - theta0))**2
                            + (a * np.sin(theta - theta0))**2)**0.5


        if not cell['is_HC']:
            return [], np.nan, np.nan, np.nan

        nbs = [cells.loc[cells['global_id_cells'] == n_id].iloc[0]
                for n_id in cell['neighbors']]
        nbs = [nb for nb in nbs if not nb['is_HC']]
        
        nnbs_ids = [nn_id for nb in nbs for nn_id in nb['hair_neighbors']]
        if len(nnbs_ids) == 0:
            return nnbs_ids, 0., 0., 0.

        nnbs_ids = np.unique(np.array(nnbs_ids).astype(int))

        nnbs = pd.DataFrame([cells.loc[   cells['global_id_cells']
                                        == nn_id].iloc[0] for nn_id in nnbs_ids])
        nnbs = nnbs.loc[nnbs['global_id_cells'] != cell['global_id_cells']]

        if nnbs.shape[0] < 3:
            return nnbs_ids, 0., 0., 0.

        displ_x = nnbs['center_x_cells'] - cell['center_x_cells']
        displ_y = nnbs['center_y_cells'] - cell['center_y_cells']

        # # use elongation to calculate correction of hexatic order
        hex = pd.DataFrame()
        hex['x'] = displ_x
        hex['y'] = displ_y
        hex['r'] = (displ_x**2 + displ_y**2)**0.5
        hex['phi'] = np.arctan2(displ_y, displ_x)

        # sort anti-clockwise
        hex = hex.sort_values(by='phi')

        def __hexatic_order_parameter(angles):
            return np.abs((np.exp(6j * angles)).sum()) / angles.shape[0]
        
        # the uncorrected hexatic order
        hexatic_order = __hexatic_order_parameter(np.arctan2(displ_y, displ_x))


        # correct by elongation of cell
        theta = cell['orientation_cells']
        ratio_WH = cell['elongation_ratio_WH_cells']

        # q, dual_A, theta, abs_q, ratio_WH = __elongation(hex['x'].values, hex['y'].values)

        def __rescale_dx_dy(displ_x, displ_y, *, ratio_WH, theta):
            """Rescale to circle for correction"""
            dx_ = (displ_x * np.cos(-theta) - displ_y * np.sin(-theta)) / ratio_WH
            dy_ =  displ_x * np.sin(-theta) + displ_y * np.cos(-theta)

            dx = dx_ * np.cos(theta) - dy_ * np.sin(theta)
            dy = dx_ * np.sin(theta) + dy_ * np.cos(theta)

            return dx, dy

        dx, dy = __rescale_dx_dy(displ_x, displ_y, ratio_WH=ratio_WH, theta=theta)

        hexatic_order_corrected = __hexatic_order_parameter(np.arctan2(dy, dx))


        # fit an ellipse through data
        if nnbs.shape[0] > 3:
            p0 = 1.01, 0.99, 0.01
            popt, pcov = curve_fit(r_ellipse,
                                hex['phi'].values, hex['r'].values,
                                p0, sigma = hex['r'].values)
            A, B, Alpha = popt
            dx, dy = __rescale_dx_dy(displ_x, displ_y, ratio_WH=A/B, theta=Alpha)
            __hexatic_order_corrected = __hexatic_order_parameter(np.arctan2(dy, dx))
        else:
            __hexatic_order_corrected = 0.

        if False:
            fig, ax = plt.subplots(1, 1, sharex=True)

            A = hex['r'].mean() * ratio_WH**0.5#**hex['r'].mean()
            B = hex['r'].mean() / ratio_WH**0.5

            ax.scatter(x=displ_x, y=displ_y, c='red', label='original / ellipse')
            ax.scatter(x=cell['vx_coords_x_cells'] - cell['center_x_cells'],
                    y=cell['vx_coords_y_cells'] - cell['center_y_cells'],
                    c='blue', label='cell', s=0.5)
            ax.scatter(x=0., y=0.)
            PHI = np.linspace(0., 2 * math.pi, 501)
            R = r_ellipse(PHI, A, B, theta)
            ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='red')

            ax.scatter(dx, dy, c='black', label='rescaled / circle')
            ax.scatter(x=(dx**2 + dy**2).mean()**0.5 * np.cos(PHI), y=(dx**2 + dy**2).mean()**0.5 * np.sin(PHI), s=0.5, c='black')

            # transform to polar coordinates
            r = (displ_x**2 + displ_y**2)**0.5
            phi = np.arctan2(displ_y, displ_x)

            # fit an ellipse through data
            p0 = 1.01, 0.99, 0.01
            popt, pcov = curve_fit(r_ellipse, phi, r, p0, sigma = r)
            A, B, Alpha = popt
            R = r_ellipse(PHI, A, B, Alpha)

            ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='green', label='ellipse')
            ax.quiver(0., 0., abs_q * np.cos(theta), abs_q * np.sin(theta),
                    angles='xy', scale_units='xy')

            plt.axis('equal')
            plt.legend(loc=1, facecolor='white')

            ax.set_title("ratio {}\n abs {}, A / B {}, radius {}".format(ratio_WH, abs_q, A / B, hex['r'].mean()))

            plt.savefig(f"./"+ "fit_ellipse_{}.svg".format(cell['global_id_cells']))
            plt.close()

        return nnbs_ids, hexatic_order, hexatic_order_corrected, __hexatic_order_corrected

    result = cells.apply(hexatic_order, cells=cells, axis=1)
    
    cells['next_HC_neighbors'] = result.apply(lambda r: np.asarray(r[0]).astype(int))
    cells['num_next_HC_neighbors'] = result.apply(lambda r: len(r[0]))
    cells['hexatic_order'] = result.apply(lambda r: r[1])
    cells['hexatic_order_corrected'] = result.apply(lambda r: r[2])
    cells['hexatic_order_ellipse_fit'] = result.apply(lambda r: r[3])


    log.info("Calculated hexatic order in %.2f s.", time.time() - _time)


    # # Tracking
    # # TODO not debugged
    # def relative_length(bond, *, ref_state):
    #     if not np.isfinite(bond['track_id_bonds']):
    #         return np.nan

    #     if bond['type'] == 'border_plus_one' or bond['type'] == 'border':
    #         return np.nan

    #     ref = ref_state.loc[ref_state['track_id_bonds'] == bond['track_id_bonds']]
    #     if ref.shape[0] == 0:
    #         return np.nan

    #     ref = ref.iloc[0]

    #     if ref['type'] == 'border_plus_one' or ref['type'] == 'border':
    #         return np.nan

    #     return bond['bond_length_in_px'] / ref['bond_length_in_px']

    # def relative_vertex_distance(bond, *, ref_state):
    #     if not np.isfinite(bond['track_id_bonds']):
    #         return np.nan

    #     if bond['type'] == 'border_plus_one' or bond['type'] == 'border':
    #         return np.nan

    #     ref = ref_state.loc[ref_state['track_id_bonds'] == bond['track_id_bonds']]
    #     if ref.shape[0] == 0:
    #         return np.nan

    #     ref = ref.iloc[0]

    #     if ref['type'] == 'border_plus_one' or ref['type'] == 'border':
    #         return np.nan

    #     return bond['vertex_distance'] / ref['vertex_distance']

    # def relative_area(cell, *, ref_state):
    #     if not np.isfinite(cell['track_id_cells']):
    #         return np.nan

    #     ref = ref_state.loc[ref_state['track_id_cells'] == cell['track_id_cells']]
    #     if ref.shape[0] == 0:
    #         return np.nan

    #     ref = ref.iloc[0]
    #     if ref['is_border_cell']:
    #         return np.nan

    #     return cell['area_cells'] / ref['area_cells']

    # def relative_normalized_area(cell, *, ref_state):
    #     if not np.isfinite(cell['track_id_cells']):
    #         return np.nan

    #     ref = ref_state.loc[ref_state['track_id_cells'] == cell['track_id_cells']]
    #     if ref.shape[0] == 0:
    #         return np.nan

    #     ref = ref.iloc[0]
    #     if ref['is_border_cell']:
    #         return np.nan

    #     return cell['normalized_area_cells'] / ref['normalized_area_cells']

    if tracked_cells_suffix is not None and False:
        log.debug("Tracking cells...")
        _time = time.time()


        for track_id in bonds['track_id_bonds'].unique():
            tracked_bond = bonds.loc[bonds['track_id_bonds'] == track_id]

            ref_state = tracked_bond.loc[tracked_bond['time'] == tracked_bond['time'].min()]


        bonds['bond_rel_length_in_px'] = bonds.apply(
            relative_length, axis=1, ref_state=ref_state)
        bonds['bond_rel_length_change_in_px'] = (
            bonds['bond_rel_length_in_px'] - 1.)
        bonds['bond_length_var_in_px'] = (
            bonds['bond_rel_length_change_in_px']**2)

        ref_state = cells.loc[cells['frame'] == 0]
        cells['area_rel_cells'] = cells.apply(
            relative_area, axis=1, ref_state=ref_state)
        cells['area_rel_change_cells'] = cells['area_rel_cells'] - 1.
        cells['area_var_cells'] = cells['area_rel_change_cells']**2

        cells['normalized_area_rel_cells'] = cells.apply(
            relative_normalized_area, axis=1, ref_state=ref_state)
        cells['normalized_area_rel_change_cells'] = (
            cells['normalized_area_rel_cells'] - 1.)
        cells['normalized_area_var_cells'] = (
            cells['normalized_area_rel_change_cells']**2)

        log.info("Completed tracking of cells in {:.2f}s.".format(time.time() - _time))

        raise RuntimeError("Cell tracking not implemented!")

    else:
        log.info("Cell tracking skipped.")



    if import_cilia is not None and import_cilia:
        log.debug("Importing cilia ..")
    else:
        log.info("NOT importing cilia")
        import_cilia = {'path': None}
    _time = time.time()

    def import_coords_from_csv(path, filename, *, pixel_per_micron=[1., 1.], **kwargs):
        if path is None:
            return None
        filename = filename.split(".")[0]
        _path = os.path.join(data_basedir, path, filename + '.csv')
        
        if not os.path.isfile(_path):
            log.warning("No cilia data at {}".format(_path))
            return None
        
        coords = pd.read_csv(_path, **kwargs)
        coords['X0'] = coords['X']
        coords['Y0'] = coords['Y']
        coords['X'] = coords['X'] * pixel_per_micron[0]
        coords['Y'] = coords['Y'] * pixel_per_micron[1]
        return coords


    log.debug("Parsing cilia import args: \n%s", import_cilia)

    cilia = None
    cilia_import_path = import_cilia.get('path')

    actin_foci = None
    actin_foci_import_path = import_actin_foci.get('path')

    for (filepath, filename), cs in cells.groupby(by=['filepath', 'filename']):
        _cilia = import_coords_from_csv(cilia_import_path, filename, 
                                        **import_cilia.get('import_kwargs', dict()))
        if _cilia is None:
            continue

        _cilia['filename'] = filename
        _cilia['filepath'] = filepath
        _cilia['file_id'] = cs['file_id'].unique()[0]

        if cilia is None:
            cilia = _cilia
        else:
            cilia = pd.concat([cilia, _cilia], axis=0, join='outer')

    for (filepath, filename), cs in cells.groupby(by=['filepath', 'filename']):
        _foci = import_coords_from_csv(
            actin_foci_import_path, filename, 
            **import_actin_foci.get('import_kwargs', dict())
        )

        if _foci is None:
            continue

        _foci['filename'] = filename
        _foci['filepath'] = filepath
        _foci['file_id'] = cs['file_id'].unique()[0]

        # def min_distance(f, *, cis):        
        #     d = ((cis['X'] - f['X'])**2 + (cis['Y'] - f['Y'])**2)**0.5

        #     _ci = cis.iloc[d.argmin()]
        #     _ci['d_cilium_foci'] = d.min()

        #     return _ci

        # combined = _foci.apply(min_distance, axis=1, cis=_cilia)
        # combined = combined.loc[combined['d_cilium_foci'] <= 5]

        # combined['filename'] = filename
        # combined['filepath'] = filepath
        # combined['file_id'] = cs['file_id'].unique()[0]

        if actin_foci is None:
            actin_foci = _foci
        else:
            actin_foci = pd.concat([actin_foci, _foci], axis=0, join='outer')


    def closest_cilium(cell, *, cilia):
        """Assign to a cell the cilium that is the closest in space

        Return:
            - global id of cilium
            - distance of cilium to cell center in units of cell radius
            - X position of cilium
            - Y position of cilium
            - Number of cilia that have distance rho < 1 to cell center

        """
        _cilia = cilia.loc[cilia['file_id'] == cell['file_id']].reset_index()
        if _cilia.empty or cell['is_border_cell']:
            cilium = pd.Series(data=None, index=_cilia.columns, dtype=object)
            cilium['rho'] = np.nan
            cilium['count_other'] = np.nan
            return cilium

        r = (  (_cilia['X'] - cell['center_x_cells'])**2
             + (_cilia['Y'] - cell['center_y_cells'])**2)**0.5

        N = np.count_nonzero(r < (cell['area_cells']/math.pi)**0.5)
        if N <= 1:
            id = r.loc[r == r.min()].index[0]
        elif N > 1:
            _r = r.loc[r < (cell['area_cells']/math.pi)**0.5]
            cs = _cilia.iloc[_r.index]
            cs['rho'] = _r
            cs = cs.sort_values(**import_cilia.get('sort_values', dict()))
            id = cs.index[0]

        cilium = _cilia.iloc[id]
        cilium['rho'] = r.iloc[id]
        cilium['count_other'] = N

        return cilium


    cells['local_id_cilia'] = np.nan
    cells['global_id_cilia'] = np.nan
    cells['cilium_X'] = np.nan
    cells['cilium_Y'] = np.nan
    cells['cilium_DX'] = np.nan
    cells['cilium_DY'] = np.nan
    cells['cilium_rho'] = np.nan
    cells['cilium_rho_normalized'] = np.nan
    cells['cilium_rho_corrected'] = np.nan
    cells['cilium_phi'] = np.nan
    cells['cilium_phi_corrected'] = np.nan
    cells['num_cilia_cells'] = np.nan
    if cilia is not None:
        cilia = cilia.rename(columns={" ": "local_id_cilia"})
        cilia['global_id_cilia'] = (  cilia['local_id_cilia']
                                    + cilia['file_id'] * cilia['local_id_cilia'].max())
        
        _cells = cells
        if import_cilia.get('HC_only', False):
            _cells = cells.where(cells['is_HC'])
        assigned_cilia = _cells.apply(closest_cilium, axis=1, cilia=cilia)

        assigned_cilia['DX'] = assigned_cilia['X'] - cells['center_x_cells']
        assigned_cilia['DY'] = assigned_cilia['Y'] - cells['center_y_cells']

        assigned_cilia['phi'] = np.arctan2(assigned_cilia['DY'],
                                        assigned_cilia['DX'])

        assigned_cilia = assigned_cilia.where(cells['is_HC'])
        
        # keep 98 percentile
        # _cilia_rho_98_percentile = (  assigned_cilia['rho'].mean()
        #                             + 2.5*assigned_cilia['rho'].std())

        # assigned_cilia = assigned_cilia.where(  assigned_cilia['rho']
        #                                       < _cilia_rho_98_percentile)

        assigned_cilia = assigned_cilia.where(
            assigned_cilia['rho']  < (  import_cilia.get('max_radius', np.inf)
                                    * (_cells['area_cells']/math.pi)**0.5)
        )

        # assign to cells
        cells['local_id_cilia'] = assigned_cilia['local_id_cilia']
        cells['global_id_cilia'] = assigned_cilia['global_id_cilia']
        cells['cilium_X'] = assigned_cilia['X']
        cells['cilium_Y'] = assigned_cilia['Y']
        cells['cilium_DX'] = assigned_cilia['DX']
        cells['cilium_DY'] = assigned_cilia['DY']
        cells['cilium_rho'] = assigned_cilia['rho']
        cells['cilium_rho_normalized'] = assigned_cilia['rho'] / (cells['area_cells'] / math.pi)**0.5
        cells['cilium_phi'] = assigned_cilia['phi']
        cells['num_cilia_cells'] = assigned_cilia['count_other']


        # correct by elongation of cell
        theta = cells['orientation_cells']
        ratio_WH = cells['elongation_ratio_WH_cells']

        # q, dual_A, theta, abs_q, ratio_WH = __elongation(hex['x'].values, hex['y'].values)

        def __rescale_dx_dy(displ_x, displ_y, *, ratio_WH, theta):
            """Rescale to circle for correction"""
            dx_ = (displ_x * np.cos(-theta) - displ_y * np.sin(-theta)) / ratio_WH
            dy_ =  displ_x * np.sin(-theta) + displ_y * np.cos(-theta)

            dx = dx_ * np.cos(theta) - dy_ * np.sin(theta)
            dy = dx_ * np.sin(theta) + dy_ * np.cos(theta)

            return dx, dy

        dx, dy = __rescale_dx_dy(cells['cilium_DX'], cells['cilium_DY'], ratio_WH=ratio_WH, theta=theta)

        cells['cilium_rho_corrected'] = (dx**2 + dy**2)**0.5 / (cells['area_cells'] / math.pi)**0.5
        cells['cilium_phi_corrected'] = np.arctan2(dy, dx)


        if False:
            for index, cell in cells.loc[cells['is_HC']].iterrows():
                if np.isnan(cell['cilium_DX']):
                    continue
                
                fig, ax = plt.subplots(1, 1, sharex=True)

                ax.scatter(x=cell['cilium_DX'], y=cell['cilium_DY'], c='red', label='original')
                ax.scatter(x=0., y=0., c='black', label='original')
                ax.scatter(x=cell['vx_coords_x_cells'] - cell['center_x_cells'],
                        y=cell['vx_coords_y_cells'] - cell['center_y_cells'],
                        c='gray', label='cell vertices')


                def r_ellipse(theta, a, b, theta0):
                    return a * b / (  (b * np.cos(theta - theta0))**2
                                    + (a * np.sin(theta - theta0))**2)**0.5

                norm_area = (cell['area_cells'] / math.pi)**0.5

                PHI = np.linspace(0., 2 * math.pi, 501)

                R = r_ellipse(PHI, norm_area, norm_area, 0.)
                ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='black')


                # correction by ellipse
                ratio_WH = cell['elongation_ratio_WH_cells']
                theta = cell['orientation_cells']
                R = r_ellipse(PHI, norm_area * ratio_WH**0.5, norm_area / ratio_WH**0.5, theta)
                ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='green')

                rho = cell['cilium_rho_corrected'] * norm_area
                phi = cell['cilium_phi_corrected']
                ax.scatter(x=rho * np.cos(phi), y=rho * np.sin(phi), c='green', label='corrected')


                # correction by max and min
                vx = cell['vx_coords_x_cells'] - cell['center_x_cells']
                vy = cell['vx_coords_y_cells'] - cell['center_y_cells']
                W = vx.max() - vx.min()
                H = vy.max() - vy.min()

                ax.scatter(x=cell['cilium_DX'] / W * norm_area,
                        y=cell['cilium_DY'] / H * norm_area,
                        c='yellow', label='min/max')

                

                # ax.scatter(x=0., y=0.)
                # PHI = np.linspace(0., 2 * math.pi, 501)
                # R = r_ellipse(PHI, A, B, theta)
                # ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='red')

                # ax.scatter(dx, dy, c='black', label='rescaled / circle')
                # ax.scatter(x=(dx**2 + dy**2).mean()**0.5 * np.cos(PHI), y=(dx**2 + dy**2).mean()**0.5 * np.sin(PHI), s=0.5, c='black')

                # # transform to polar coordinates
                # r = (displ_x**2 + displ_y**2)**0.5
                # phi = np.arctan2(displ_y, displ_x)

                # # fit an ellipse through data
                # p0 = 1.01, 0.99, 0.01
                # popt, pcov = curve_fit(r_ellipse, phi, r, p0, sigma = r)
                # A, B, Alpha = popt
                # R = r_ellipse(PHI, A, B, Alpha)

                # ax.scatter(x=R * np.cos(PHI), y=R * np.sin(PHI), s=0.5, c='green', label='ellipse')
                # ax.quiver(0., 0., abs_q * np.cos(theta), abs_q * np.sin(theta),
                #         angles='xy', scale_units='xy')

                plt.axis('equal')
                plt.legend(loc=1, facecolor='white')

                # ax.set_title("ratio {}\n abs {}, A / B {}, radius {}".format(ratio_WH, abs_q, A / B, hex['r'].mean()))

                plt.savefig(f"./"+ "cilia_correction_{}.svg".format(cell['global_id_cells']))
                plt.close()

        log.info("Assigned %i cilia for %i HC in %.2fs.", 
                cells['local_id_cilia'].count(),
                cells['is_HC'].sum(),
                time.time() - _time)


    if ppMLC_foci_path is not None:
        log.debug("Importing ppMLC foci ..")
    else:
        log.info("NOT importing ppMLC foci")

    _time = time.time()


    def import_ppMLC_foci(path, filename, **kwargs):
        if ppMLC_foci_path is None:
            return None

        filename = filename.split(".")[0]
        _path = os.path.join(path, ppMLC_foci_path, filename) + ".csv"
        if not os.path.isfile(_path):
            log.warning("No ppMLC foci data at {}".format(_path))
            return None
        
        return pd.read_csv(_path, **kwargs)


    def focus_distance(focus, bonds):
        """Calculate the distance between a focus and the closest bond

        Returns:
            global id of closest bond
            (normal) distance to that bond
        """
        _bonds = bonds.loc[bonds['file_id'] == focus['file_id']]

        # the bond
        bdx = _bonds['vx_2_x'] - _bonds['vx_1_x']
        bdy = _bonds['vx_2_y'] - _bonds['vx_1_y']
        l = (bdx**2 + bdy**2)**0.5
        
        # the bond unit vector
        nx = bdx / l
        ny = bdy / l

        # the focus' position
        fx = focus['X']
        fy = focus['Y']

        # the vector between bond start and focus
        afx = fx - _bonds['vx_1_x']
        afy = fy - _bonds['vx_1_y']

        # the parallel and normal distance to bond
        a = (afx * nx + afy * ny)
        d = ((afx - a * nx)**2 + (afy - a * ny)**2)**0.5

        # normal distance can only count along bond
        d = d.where(a > 0)
        d = d.where(a < l)
        d = d.dropna()

        if d.empty:
            return None, np.nan

        return _bonds.loc[d.idxmin()]['global_id_bonds'], d.min()

    ppMLC = None
    # Iterate the filenames for ppMLC foci
    for (filepath, filename), bs in bonds.groupby(by=['filepath', 'filename']):
        _ppMLC = import_ppMLC_foci(data_basedir, filename, sep=",", header=0)
        if _ppMLC is None:
            continue

        _ppMLC['filename'] = filename
        _ppMLC['filepath'] = filepath
        _ppMLC['frame_nb'] = bs['frame_nb'].unique()[0].astype(int)
        _ppMLC['file_id'] = bs['file_id'].unique()[0].astype(int)

        if ppMLC is None:
            ppMLC = _ppMLC
        else:
            ppMLC = pd.concat([ppMLC, _ppMLC], axis=0, join='outer')


    bonds['ppMLC_foci_X'] = np.nan
    bonds['ppMLC_foci_Y'] = np.nan
    bonds['ppMLC_foci_distance'] = np.nan
    bonds['ppMLC_foci_Size'] = 0.
    bonds['ppMLC_foci_Max'] = 0.
    bonds['ppMLC_foci_Total'] = 0.
    if ppMLC is not None:
        ppMLC['global_id_foci'] = (  ppMLC['file_id'] * ppMLC['Peak #'].max()
                                + ppMLC['Peak #'])

        for i, focus in ppMLC.iterrows():
            bond_id, d = focus_distance(focus, bonds)

            if bond_id is None:
                continue

            # check that new foci is closer
            _d = bonds.loc[bonds['global_id_bonds'] == bond_id, 'ppMLC_foci_distance']
            _d = _d.iloc[0]
            if np.isfinite(_d) and _d < d:
                continue


            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_X'] = focus['X']
            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_Y'] = focus['Y']
            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_distance'] = d
            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_Size'] = focus['Size']
            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_Max'] = focus['Max']
            bonds.loc[bonds['global_id_bonds'] == bond_id,
                    'ppMLC_foci_Total'] = focus['Total']

        
        # keep 98 percentile
        mask = bonds['ppMLC_foci_distance'] < (  bonds['ppMLC_foci_distance'].mean()
                                            + 2.5*bonds['ppMLC_foci_distance'].std())
        mask = mask.where(bonds['ppMLC_foci_Total'] > 0., True)
        bonds['ppMLC_foci_X'] = bonds['ppMLC_foci_X'].where(mask)
        bonds['ppMLC_foci_Y'] = bonds['ppMLC_foci_Y'].where(mask)
        bonds['ppMLC_foci_distance'] = bonds['ppMLC_foci_distance'].where(mask)
        bonds['ppMLC_foci_Size'] = bonds['ppMLC_foci_Size'].where(mask)
        bonds['ppMLC_foci_Max'] = bonds['ppMLC_foci_Max'].where(mask)
        bonds['ppMLC_foci_Total'] = bonds['ppMLC_foci_Total'].where(mask)

        quantiles = ppMLC['Total'].quantile([1/3., 2/3.], interpolation='nearest')

        bonds['ppMLC_foci_quantile'] = 'mid'
        bonds.loc[bonds['ppMLC_foci_Total'] < quantiles.iloc[0],
                'ppMLC_foci_quantile'] = 'lower'
        bonds.loc[bonds['ppMLC_foci_Total'] > quantiles.iloc[1],
                'ppMLC_foci_quantile'] = 'upper'
        bonds.loc[bonds['ppMLC_foci_Total'] == 0.,    
                'ppMLC_foci_quantile'] = 'None'


        log.info("Assigned %i ppMLC foci for %i bonds in %.2f s.",
                bonds.loc[bonds['ppMLC_foci_Total'] > 0.]['local_id_bonds'].count(),
                bonds['local_id_bonds'].count(),
                time.time() - _time)


    log.info("Completed post-processing of data in %.2fs.",
            time.time() - start_time)



    log.debug("Saving data with suffix '%s' in %s...",
            save_file_prefix, save_treated_data)

    for filepath, cs in cells.groupby(by='filepath'):
        cs.to_csv("{}_cells.csv"
                  "".format(os.path.join(filepath, save_treated_data, save_file_prefix)))
        for filename, _cs in cs.groupby(by='filename'):
            _cs.to_csv("{}_cells.csv"
                       "".format(os.path.join(filepath, save_treated_data, filename.split('.')[0])))

    for filepath, bs in bonds.groupby(by='filepath'):
        bs.to_csv("{}_bonds.csv"
                "".format(os.path.join(filepath, save_treated_data, save_file_prefix)))
        for filename, _bs in bs.groupby(by='filename'):
            _bs.to_csv("{}_bonds.csv"
                    "".format(os.path.join(filepath, save_treated_data, filename.split('.')[0])))

    log.info("Saved data with suffix '%s' in %s.",
            save_file_prefix, save_treated_data)

    log.debug("Creating composite images as recapitulation of data ...")

    for (filepath, filename), frame in cells.groupby(by=['filepath', 'filename']):
        plot_frame(
            cells, bonds,
            filepath=filepath, filename=filename,
            load_images=load_images,
            load_segmentation_path=load_segmentation_path,
            save_to_images=save_to_images,
            plot_polygons=True,
            log=log
        )

    log.info("Created composite images as recapitulation of data in {}.".format(save_to_images))

    log.info("Success!")

    return cells, bonds
