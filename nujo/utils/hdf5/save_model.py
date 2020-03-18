from nujo.utils.hdf5.HDF5dict import H5Dict, save_to_binary_h5py

h5dict = H5Dict


def _serialize_model(model, h5dict):
    pass


def save_model(model, filepath):
    ''' Save a model to a HDF5 file

    Parameters:
    -----------
    model : a model instance to be saved

    '''

    if H5Dict.is_supported_type(filepath):
        with H5Dict(filepath, mode='w') as h5dict:
            _serialize_model(model, h5dict)
    else:
        # write as binary stream
        def save_function(h5file):
            _serialize_model(model, H5Dict(h5file))

        save_to_binary_h5py(save_function, filepath)
