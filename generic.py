from heapq import merge
import os
import os.path
import time
import logging
from typing import Union, List
import gc

import yaml
import collections
from copy import deepcopy

import numpy as np
import xarray as xr

import tifffile
from PIL import Image
from PIL.TiffTags import TAGS

def prepare_file_path(dir: str, *, sep='/'):
    """Removes trailing `/` of directories"""
    if dir is None:
        return None
    if len(dir) == 0:
        return dir
    if dir[-1] == sep:
        return dir[:-2]

    return dir

def merge_paths(dirs: List[str]):
    """Merges base_path, directory and file to a str"""
    if dirs is None:
        return None
    if len(dirs) == 0:
        return None
    merge = None
    for dir in dirs:
        if dir is None:
            continue
        
        if merge is None:
            merge = prepare_file_path(dir)
        else:
            merge = "{}/{}".format(merge, prepare_file_path(dir))

    return merge

def load_image(*,
    base_dir: str,
    in_dir: str=None,
    filename: str,
    dims=['z', 'c', 'x', 'y'],
    normalise: bool,
    logger=None,
    metadata: dict=None
):
    """Loads a .tif at FilePath image and its meta data"""
    
    def extract_image_metadata(FilePath, *, logger, metadata: dict=None):
        """Searches the image at FilePath for meta data, e.g. resolution"""
        if not os.path.isfile(FilePath):
            raise RuntimeError(
                "Failed to locate file {}!".format(FilePath)
            )
        logger.debug("Searching %s for meta data ...", FilePath)

        meta_dict = dict()
        meta_dict['resolution_scale'] = 'pixel_per_micron'
        meta_dict['resolution_x'] = np.nan
        meta_dict['resolution_y'] = np.nan
        meta_dict['resolution_z'] = np.nan
        meta_dict['spacing'] = np.nan
        with Image.open(FilePath) as img:
            for key in img.tag_v2:
                meta_dict[TAGS[key]] = img.tag[key][0]
            meta_dict['resolution_x'] = meta_dict.get(
                'XResolution',
                [np.nan]
            )[0] / 1.e6
            meta_dict['resolution_y'] = meta_dict.get(
                'YResolution',
                [np.nan]
            )[0] / 1.e6
            meta_dict['ImageDescription'] = meta_dict.get(
                'ImageDescription', dict())
            for m in meta_dict['ImageDescription'].split('\n'):
                if len(m) == 0:
                    continue            
                k, v = m.split('=')
                if k == 'spacing':
                    meta_dict['spacing'] = float(v)
                    meta_dict['resolution_z'] = 1./float(v)
                    break
        if (   np.isnan(meta_dict['resolution_x'])
            or np.isnan(meta_dict['resolution_y'])
        ):
            if metadata is None:
                logger.warning("No resolution found for image {}".format(FilePath))
                metadata = dict()
            meta_dict['resolution_x'] = metadata.get('resolution_x', np.nan)
            meta_dict['resolution_y'] = metadata.get('resolution_y', np.nan)
            meta_dict['resolution_z'] = metadata.get('resolution_z', np.nan)
            meta_dict['resolution_scale'] = metadata.get('resolution_scale', np.nan)
            logger.debug(
                "   Assigned image resolution: {:.2f}x{:.2f} {}"
                "".format(
                meta_dict['resolution_x'],
                meta_dict['resolution_y'],
                meta_dict['resolution_scale']
            ))

        else:
            logger.debug(
                "   Found image resolution: {:.2f}x{:.2f} {}"
                "".format(
                meta_dict['resolution_x'],
                meta_dict['resolution_y'],
                meta_dict['resolution_scale']
            ))
        return meta_dict


    if logger is None:
        logger = logging.getLogger("ImageLoader")

    FilePath = merge_paths([base_dir, in_dir, filename])

    if not os.path.isfile(FilePath):
        logger.error("No image available at %s", FilePath)
        raise RuntimeError("No image available at {}".format(FilePath))

    logger.debug("Loading image: %s ...", FilePath)

    image = np.array(tifffile.imread(FilePath), dtype='float32')

    attrs = extract_image_metadata(FilePath, logger=logger, metadata=metadata)

    coords = dict()
    for i, dim in enumerate(dims):
        if dim != 'c':
            coords[dim] = (np.linspace(
                    0.5, image.shape[i]-0.5, image.shape[i])
                ) / attrs.get(f"resolution_{dim}", 1)
        else:
             coords[dim] = range(image.shape[i])

    image = xr.DataArray(
        image,
        dims=dims,
        coords=coords,
        name='image',
        attrs=attrs
    )

    if 'z' not in image.coords:
        image = image.expand_dims(dim={'z': [0]})
    if 'c' not in image.coords:
        image = image.expand_dims(dim={'c': [1]})

    image = image.transpose('x', 'y', 'z', 'c')

    if normalise:
        logger.debug("Normalising color intensities to 1 ...")
        max_channels = image.max(dim=['x', 'y', 'z'])
        max_channels = max_channels.where(max_channels > 0, 1)
        image = image / max_channels.data
        image.attrs = attrs
    else:
        logger.debug("NOT Normalising color intensities ...")

    return image

def save_image(image, *, base_dir: str, out_dir: str=None, filename: str,
               normalise: bool=False, logger=None):
    
    out_file = merge_paths([base_dir, out_dir, filename])
    if logger is not None:
        logger.info("Saving result to file %s", out_file)

    if len(image.dims) == 2:
        image = image.expand_dims(dim={'z': [0]})
        image = image.expand_dims(dim={'c': [1]})
    if len(image.dims) == 3 and 'c' in image.dims:
        image = image.expand_dims('z')
    if len(image.dims) == 3 and 'z' in image.dims:
        image = image.expand_dims('c')

    __image = image.transpose('c', 'z', 'y', 'x')

    if normalise:
        __image = __image / __image.max(dim=['x', 'y', 'z']) * np.finfo(np.float32).max

    result = np.asarray(__image, dtype='float32')


    tifffile.imwrite(
        out_file,
        np.moveaxis(result, 0, 1),
        imagej=True,
        resolution=(
            image.attrs['resolution_x'],
            image.attrs['resolution_y']
        ),
        metadata={
            'spacing': image.attrs['spacing'],
            'unit': 'um',
            'axes': 'ZCYX'
        }
    )

def load_yaml(file):
    with open(file, 'r') as stream:
        try:
            cfg=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise 
    return cfg


def recursive_update(d: dict, u: dict) -> dict:
    """Recursively updates the Mapping-like object ``d`` with the Mapping-like
    object ``u`` and returns it. Note that this does *not* create a copy of
    ``d``, but changes it mutably!

    Based on: http://stackoverflow.com/a/32357112/1827608

    Args:
        d (dict): The mapping to update
        u (dict): The mapping whose values are used to update ``d``

    Returns:
        dict: The updated dict ``d``
    """
    if u is None:
        return d
    for k, v in u.items():
        if isinstance(d, collections.abc.Mapping):
            # Already a Mapping
            if isinstance(v, collections.abc.Mapping):
                # Already a Mapping, continue recursion
                d[k] = recursive_update(d.get(k, {}), v)
                # This already creates a mapping if the key was not available
            else:
                # Not a mapping -> at leaf -> update value
                d[k] = v  # ... which is just u[k]

        else:
            # Not a mapping -> create one
            d = {k: u[k]}
    return d

def treat_batch(*, config_path: str, func: callable, logger):
    base_cfg = load_yaml(config_path)
    basedir = base_cfg["base_dir"]
    base_cfg.pop("base_dir")

    update_cfg_name = base_cfg['update_cfg_name']
    base_cfg.pop("update_cfg_name")

    time_total = time.time()
    total_files = 0

    directories = None
    if 'directories' in base_cfg:
        directories = base_cfg["directories"]
        base_cfg.pop('directories')
    if directories is None:
        directories = [None]
    elif len(directories) > 1:
        logger.info("Treating %i directories: %s",
                    len(directories),
                    directories)
    for dir in directories:
        time_directory = time.time()

        logger.info("Treating directory %s ...", dir)

        dir = merge_paths([basedir, dir])

        dir_cfg = deepcopy(base_cfg)
        if os.path.isfile(merge_paths([dir, update_cfg_name])):
            logger.debug("Updating config from file `%s`.", update_cfg_name)
            dir_cfg = recursive_update(
                dir_cfg, 
                load_yaml(merge_paths([dir, update_cfg_name]))
            )

        if 'update_cfg__suffix' in dir_cfg:
            update_cfg__suffix = dir_cfg['update_cfg__suffix']
            dir_cfg.pop('update_cfg__suffix')
        else:
            update_cfg__suffix = '.yml'
        

        in_dir = merge_paths([dir, dir_cfg['in_dir']])

        __files = []
        __cfgs = []
        for file in os.listdir(in_dir):
            if not os.path.isfile(merge_paths([in_dir, file])):
                continue
            if file.endswith(update_cfg__suffix):
                __cfgs.append(file)
            elif file.split('.')[1] == 'tif':
                __files.append(file)
            else:
                RuntimeError("Unknown file ending %s on file %s. "
                            "Please provide only '.tif' or config files!",
                            file.split('.')[1], file)

        files = []
        for file in __files:
            base = file.split(".")[0]
            if base + update_cfg__suffix in __cfgs:
                files.append((file, base + update_cfg__suffix))
            else:
                files.append((file, None))

        logger.debug("Starting to treat %i files in directory %s ...",
                     len(files), in_dir)
        
        treated_files = 0
        for file, file_cfg in files:
            time_file = time.time()
            
            logger.debug("Treating file %s (%i / %i) ...",
                         file, treated_files+1, len(files))

            cfg = deepcopy(dir_cfg)
            if file_cfg is not None:
                cfg = recursive_update(
                    cfg, 
                    load_yaml(merge_paths([in_dir, file_cfg]))
                )

            if 'cfg_out_dirs' in cfg:
                cfg_out_dirs = cfg['cfg_out_dirs']
                cfg.pop('cfg_out_dirs')
            else:
                cfg_out_dirs = []

            if 'out_dir' in cfg:
                cfg_out_dirs.append(cfg['out_dir'])

            result = func(
                filename=file,
                base_dir=dir,
                **cfg
            )


            out_dirs = [d for d in cfg_out_dirs if d is not None]
            for d in np.unique(out_dirs):
                save_cfg = merge_paths([dir, d, file.split('.')[0]+'__config.yml'])
                with open(save_cfg, 'w') as outfile:
                    yaml.dump(cfg, outfile, default_flow_style=False)

            del result
            gc.collect()

            treated_files += 1
            logger.info(
                "Treated %s in %i secs. Time remaining: %i secs.",
                file,
                time.time() - time_file,
                ((time.time() - time_directory)
                 / treated_files * (len(files) - treated_files))
            )

            total_files += 1


        if len(files) > 0:
            logger.info(
                "Successfully treated %i files in %i seconds "
                "in directory %s.",
                len(files), time.time() - time_directory, dir
            )
            logger.info("==============================================")
        else:
            logger.error("No files found in %s!", in_dir)
            logger.info("==============================================")


    logger.info("==============================================")
    logger.info(
        "Treated %i file(s) in %i seconds.",
        total_files,
        time.time() - time_total
    )
    logger.info("==============================================")
    logger.info("==============================================")



