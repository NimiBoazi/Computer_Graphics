import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools


def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0
        self.is_h = False
        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        np_img = np_img.astype(np.float32)
        padded_img = np.pad(np_img, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.5)
        grayscale_image_padded = np.dot(padded_img[..., :3], self.gs_weights)
        grayscale_image = grayscale_image_padded[1:-1, 1:-1]
        return grayscale_image.squeeze()

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        if self.resized_gs.size == 0:
            return np.array([])
        
        np_sample = self.resized_gs.squeeze()

        horizontal = np.diff(np.pad(np_sample, ((0, 0), (0, 1)), 'constant', constant_values=0.5), axis=1)
        vertical = np.diff(np.pad(np_sample, ((0, 1), (0, 0)), 'constant', constant_values=0.5), axis=0)
        np_sample = np.sqrt(np.square(horizontal) + np.square(vertical))
        np_sample[np_sample > 1] = 1

        return np_sample

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            if self.is_h:
                self.idx_map_h[i, s:] += 1
            else:
                self.idx_map_v[i, s:] += 1

    def mark_last_seam_path(self):
        path_indices = self.seam_history[-1]

        # Iterate over the rows and the corresponding column index for that row in the seam
        for r, c_current in enumerate(path_indices):
            # Check if current (r, c_current) are valid indices for the *current* index maps
            if r < self.idx_map_h.shape[0] and c_current < self.idx_map_h.shape[1]:
                # Get the original coordinates from the index maps
                c_original = self.idx_map_h[r, c_current]
                r_original = self.idx_map_v[r, c_current]

                # Check if the *original* coordinates are valid for the cumulative mask
                if (0 <= r_original < self.cumm_mask.shape[0]) and \
                (0 <= c_original < self.cumm_mask.shape[1]):
                    # Mark this original pixel as removed in the mask
                    self.cumm_mask[r_original, c_original] = False

    def paint_seams(self):
        """ Paints seams according to cumm_mask """
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1, 0, 0])

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0


    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool) 

            seam = self.find_minimal_seam()
            self.seam_history = []
            self.seam_history.append(seam)

            self.remove_seam(seam)

            if self.vis_seams:
                self.mark_last_seam_path()
                self.paint_seams()

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """

        # raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")


    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        h = self.h
        w = self.w

        # Create a mask for pixels to keep (True = keep, False = remove)
        mask = np.ones((h, w), dtype=bool)
        for i in range(h):
            mask[i, seam[i]] = False

        # Apply mask to remove the seam from all relevant matrices
        self.resized_rgb = self.resized_rgb[mask].reshape((h, w - 1, 3))
        self.resized_gs = self.resized_gs[mask].reshape((h, w - 1))

        self.E = self.E[mask].reshape((h, w - 1))

        self.idx_map_h = self.idx_map_h[mask].reshape((h, w - 1))
        self.idx_map_v = self.idx_map_v[mask].reshape((h, w - 1))

        self.w -= 1

    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        k = -1 if clockwise else 1 

        self.resized_rgb = np.rot90(self.resized_rgb, k, axes=(0, 1))
        self.resized_gs = np.rot90(self.resized_gs, k, axes=(0, 1))

        self.E = np.rot90(self.E, k, axes=(0, 1))

        self.idx_map_h = np.rot90(self.idx_map_h, k, axes=(0, 1))
        self.idx_map_v = np.rot90(self.idx_map_v, k, axes=(0, 1))

        self.h, self.w = self.resized_rgb.shape[:2]

    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.is_h = False
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.is_h = True
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.rotate_mats(clockwise=False)

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        # raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.rotate_mats(clockwise=True)
        self.seams_addition_vertical(num_add)
        self.rotate_mats(clockwise=False)

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """
        self.seams_removal_vertical(num_add)
        self.resized_rgb = np.zeros((self.rgb.shape[0], self.rgb.shape[1] + num_add, 3))

        for r in range(self.seams_rgb.shape[0]):
            i = 0
            for c in range(self.seams_rgb.shape[1]):
                if c + i < self.resized_rgb.shape[1]:
                    self.resized_rgb[r, c + i, :] = self.rgb[r, c, :]
                if self.seams_rgb[r, c, 0] == 1 and self.seams_rgb[r, c, 1] == 0 and self.seams_rgb[r, c, 2] == 0:
                    i += 1
                    if c + i < self.resized_rgb.shape[1]:
                        self.resized_rgb[r, c + i, :] = self.rgb[r, c, :]

class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""
    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        h, w = self.E.shape
        seam = []

        j = np.argmin(self.E[0])
        seam.append(j)

        for i in range(1, h):
            prev_j = seam[-1]

            candidates = []
            if prev_j > 0:
                candidates.append((self.E[i, prev_j - 1], prev_j - 1))
            candidates.append((self.E[i, prev_j], prev_j))
            if prev_j < w - 1:
                candidates.append((self.E[i, prev_j + 1], prev_j + 1))

            _, j_min = min(candidates, key=lambda x: x[0])
            seam.append(j_min)

        return seam


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """
    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        h, w = self.E.shape

        M = self.E.copy().astype(np.float32)
        backtrack = np.zeros_like(M, dtype=np.int32)

        for i in range(1, h):
            for j in range(w):
                left = M[i - 1, j - 1] if j > 0 else np.inf
                up = M[i - 1, j]
                right = M[i - 1, j + 1] if j < w - 1 else np.inf

                min_val = min(left, up, right)
                if min_val == left:
                    backtrack[i, j] = j - 1
                elif min_val == up:
                    backtrack[i, j] = j
                else:
                    backtrack[i, j] = j + 1

                M[i, j] += min_val

        seam = []
        j = np.argmin(M[-1]) 
        seam.append(j)

        for i in range(h - 1, 0, -1):
            j = backtrack[i, j]
            seam.append(j)

        seam.reverse()
        return seam

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        M = np.zeros(self.E.shape, dtype = np.float32)
        M[0 , :] = self.E[0 , :]
        padded_gs = np.pad(self.resized_gs, ((1, 1), (1, 1)), mode='constant', constant_values=0.5)

        left_shift = np.roll(padded_gs , -1, axis=1)
        right_shift = np.roll(padded_gs , 1, axis=1)
        down_shift = np.roll(padded_gs , 1, axis=0)
        
        cv = np.abs(left_shift - right_shift)
        cl = np.abs(left_shift - right_shift) + np.abs(right_shift - down_shift)
        cr = np.abs(left_shift - right_shift) + np.abs(left_shift - down_shift)
     
        cv = cv.squeeze()
        cl = cl.squeeze()  
        cr = cr.squeeze() 
        
        cv = cv[1:-1, 1:-1] 
        cl = cl[1:-1, 1:-1]
        cr = cr[1:-1, 1:-1]
        
        for i in range(1,M.shape[0]):
            L_roll = np.roll(M[i-1,:], 1)
            L_roll[0] = np.inf
            R_roll = np.roll(M[i-1,:], -1)
            R_roll[-1] = np.Inf
            M[i,:] = self.E[i,:] + np.minimum(M[i-1,:] + cv[i] , np.minimum(L_roll + cl[i],R_roll + cr[i]))
            
        return M

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    return np.round(orig_shape * np.array(scale_factors)).astype(int)


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img.reinit()
    seam_img.seams_removal_vertical(np.abs(shapes[0][1] - shapes[1][1]))
    seam_img.seams_removal_horizontal(np.abs(shapes[0][0] - shapes[1][0]))
    
    return seam_img.resized_rgb


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


