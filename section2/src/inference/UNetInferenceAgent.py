"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        mask3d = np.zeros(volume.shape, dtype=np.uint8)

        for slice_index in range(volume.shape[0]):
            # 1. one slice (64, 64) -> tensor [1, 1, 64, 64] (batch, channel, H, W), float32, on device
            slice_2d = volume[slice_index, :, :]
            slice_2d = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # 2. predict without learning; argmax over the class dim -> (64, 64) mask with 0/1/2
            with torch.no_grad():
                prediction = self.model(slice_2d)
                predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

            # 3. put this slice's mask into the 3D mask
            mask3d[slice_index, :, :] = predicted_mask

        return mask3d