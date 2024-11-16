import h5py
import numpy as np
import pickle
import re
import os

CURRENT_VERSION = 1


def recursively_save_dict_contents_to_group(h5file,
                                            path,
                                            dic,
                                            allow_pickle=False):
    """
    Recursively saves dictionary contents to HDF5 groups and datasets.
    """
    for key, item in dic.items():
        key_path = f"{path}/{key}"
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file,
                                                    key_path,
                                                    item,
                                                    allow_pickle=allow_pickle)
        elif item is None:
            h5file.create_dataset(key_path, data=0)
            h5file[key_path].attrs['type'] = 'None'
        elif isinstance(item, (list, tuple)):
            is_list = isinstance(item, list)
            if all(isinstance(x, type(item[0])) for x in item) and item[
                    0] is not None:  # Ensure all elements are of the same type
                arr = np.array(item)
                if arr.dtype.char == 'U':
                    ds = h5file.create_dataset(key_path,
                                               shape=len(item),
                                               dtype=h5py.string_dtype())
                    ds[:] = arr
                else:
                    h5file.create_dataset(key_path, data=arr)
                if is_list:
                    h5file[key_path].attrs['type'] = 'list'
                else:
                    h5file[key_path].attrs['type'] = 'tuple'
            elif len(item) == 0:  # Empty list or tuple
                h5file.create_dataset(key_path, data=np.array(item))
                h5file[key_path].attrs['type'] = 'empty_array'
            else:
                fake_dict = {}
                pref = '_tuple'
                if is_list:
                    pref = '_list'
                for i, cur_it in enumerate(item):
                    fake_dict['__item_%d' % i] = cur_it
                recursively_save_dict_contents_to_group(
                    h5file, key_path, fake_dict, allow_pickle=allow_pickle)
                h5file[key_path].attrs['type'] = 'flattened' + pref
                # raise Exception('x')
                # different types

        elif isinstance(item, np.ndarray):
            if item.dtype.char == 'U':
                # Unicode strings are handled differently
                ds = h5file.create_dataset(key_path,
                                           shape=len(item),
                                           dtype=h5py.string_dtype())
                ds[:] = item

            else:
                h5file.create_dataset(key_path,
                                      data=item)  # Directly save numpy arrays
            h5file[key_path].attrs['type'] = 'ndarray'
        elif isinstance(item, str):
            dt = h5py.string_dtype('utf-8')
            h5file.create_dataset(key_path, data=item, dtype=dt)
            h5file[key_path].attrs['type'] = 'str'
        elif isinstance(item, (int, float, complex, np.generic)):
            # Handle numbers: int, float
            h5file.create_dataset(key_path, data=item)
            h5file[key_path].attrs['type'] = 'scalar'
        else:
            if allow_pickle:
                print('Warning, type not understood, pickling', type(item))
                item = pickle.dumps(item)
                h5file[key_path] = np.void(item)
                h5file[key_path].attrs['type'] = 'pickle'
            else:
                raise ValueError(
                    f'Cannot save {type(item)} and pickling is not allowed')


def save_dict_to_hdf5(filename, dic, allow_pickle=False):
    """
    Saves the provided dictionary to an HDF5 file.
    """
    with h5py.File(filename, 'w') as h5file:
        h5file.attrs['version'] = CURRENT_VERSION
        recursively_save_dict_contents_to_group(h5file,
                                                '/',
                                                dic,
                                                allow_pickle=allow_pickle)


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively loads dictionary contents from HDF5 groups and datasets.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):  # Handle datasets
            curtyp = item.attrs['type']
            if curtyp in ['list', 'tuple']:
                ans[key] = item[:]
                if item.dtype.kind == 'O':  # Decode strings properly
                    ans[key] = ans[key].astype(str)
                if curtyp == 'list':
                    ans[key] = list(ans[key])
                else:
                    ans[key] = tuple(ans[key])
            elif item.attrs['type'] == 'ndarray':
                ans[key] = item[:]
                if item.dtype.kind == 'O':  # Decode strings properly
                    ans[key] = ans[key].astype(str)
            elif item.attrs['type'] == 'str':
                ans[key] = item[()].decode('utf-8')
            elif item.attrs['type'] in ['scalar', 'empty_array']:
                ans[key] = item[()]
            elif item.attrs['type'] == 'pickle':
                ans[key] = pickle.loads(item[()])
            elif item.attrs['type'] == 'None':
                ans[key] = None
            else:
                raise Exception('unsupported')
        elif isinstance(item,
                        h5py.Group):  # Handle groups (nested dictionaries)
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, f"{path}/{key}")
            curtyp = item.attrs.get('type')
            if curtyp in ['flattened_tuple', 'flattened_list']:
                keys = sorted(ans[key].keys())
                assert all(
                    [re.match('__item_.*', _) is not None for _ in keys])
                keys = ['__item_%d' % i for i in range(len(keys))]
                ans[key] = [ans[key][_] for _ in keys]
                if curtyp == 'flattened_tuple':
                    ans[key] = tuple(ans[key])
    return ans


def load_dict_from_hdf5(filename):
    """
    Loads the dictionary from an HDF5 file.
    """
    if not os.path.exists(filename):
        raise RuntimeError(f'Filename {filename} does not exist')
    with h5py.File(filename, 'r') as h5file:
        version = h5file.attrs.get('version', None)
        if version != CURRENT_VERSION:
            raise ValueError(f'Incompatible version: {version}')
        return recursively_load_dict_contents_from_group(h5file, '/')


def verify_data(original, loaded, path='/'):
    """
    Recursively verify that two dictionaries are identical in both value
    and type.
    """
    if not isinstance(loaded, type(original)):
        print('fail1', path, original, loaded, type(original), type(loaded))
        return False
    if isinstance(original, dict):
        if original.keys() != loaded.keys():
            print('fail2', path)
            return False
        return all(
            verify_data(original[key], loaded[key], path + '/' + key)
            for key in original)
    if isinstance(original, (list, tuple, np.ndarray)):
        if len(original) != len(loaded):
            print('fail3', path)
            return False
        return all(verify_data(o, l, path) for o, l in zip(original, loaded))
    return original == loaded


class TestClass:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def test_code():

    # Example data containing various data types
    data = {
        'x': np.int64(2),
        'vv': np.arange(3, dtype=np.float64),
        'y': {
            'inside_y': np.arange(5),
            'intside_y_dict': {
                'x': np.int64(55),
                'y': np.int64(66)
            },
        },
        't': None,
        'z': 'Hello world!',
        'tuple_data': (np.int64(1), np.int64(2), np.int64(3)),
        'list_data': [1.1, 2.2, 3.3],
        'xliststr': ['test', 'example'],
        'xliststr1': [np.str_('test'), np.str_('example')],
        'qq': np.array(['x', 'y', 'z']),
        'a1': [],
        'a2': tuple(),
        'a3': (np.int64(1), 'x'),
        'a4': [None, None, np.int64(3)],
        'myclass': TestClass(1, 2)
    }

    # Save dictionary to an HDF5 file
    fname = 'data.h5'
    save_dict_to_hdf5(fname, data, allow_pickle=True)

    # Load dictionary from HDF5 file
    loaded_data = load_dict_from_hdf5(fname)
    print(loaded_data)
    print(verify_data(data, loaded_data))


if __name__ == '__main__':
    test_code()
