import pickle

import cv2
import numpy as np
import skimage
from libsvm import svmutil
from scipy import optimize, signal, special


def estimate_phi(alpha):
    numerator = special.gamma(2 / alpha) ** 2
    denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
    return numerator / denominator


def estimate_r_hat(x):
    size = x.size
    return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)


def estimate_R_hat(r_hat, gamma):
    numerator = (gamma ** 3 + 1) * (gamma + 1)
    denominator = (gamma ** 2 + 1) ** 2
    return r_hat * numerator / denominator


def squares_mean(x):
    squares = x ** 2
    left_mask = x < 0
    left_squares = squares[left_mask]
    left_squares_mean = np.mean(left_squares)
    right_squares = squares[~left_mask]
    right_squares_mean = np.mean(right_squares)
    return np.sqrt(left_squares_mean), np.sqrt(right_squares_mean)


def estimate_alpha(x):
    r_hat = estimate_r_hat(x)
    left_squares_mean, right_squares_mean = squares_mean(x)
    gamma = left_squares_mean / right_squares_mean
    R_hat = estimate_R_hat(r_hat, gamma)
    solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
    return solution[0]


def estimate_mean(alpha, sigma_l, sigma_r, constant):
    return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - n // 2
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)


def local_deviation(image, local_mean, kernel):
    """Vectorized approximation of local deviation"""
    sigma = signal.convolve2d(image ** 2, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def local_mean_opencv(image, kernel):
    return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)


def asymmetric_generalized_gaussian_fit(x):
    r_hat = estimate_r_hat(x)
    sigma_l, sigma_r = squares_mean(x)
    gamma = sigma_l / sigma_r
    R_hat = estimate_R_hat(r_hat, gamma)
    alpha = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x[0]
    return alpha, sigma_l, sigma_r


def calculate_features(coefficients):
    alpha, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)
    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r, constant)
    return alpha, mean, sigma_l ** 2, sigma_r ** 2


def compute_pairwise_products(mscn_coefficients):
    horizontal = mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:]
    vertical = mscn_coefficients[:-1, :] * mscn_coefficients[1:, :]
    main_diagonal = mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:]
    secondary_diagonal = mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    return horizontal, vertical, main_diagonal, secondary_diagonal


class BRISQUE:
    def __init__(self):
        self.model = "assets/svm.txt"
        self.norm = "assets/normalize.pickle"

        # Load in model
        self.model = svmutil.svm_load_model(self.model)
        with open(self.norm, 'rb') as f:
            scale_params = pickle.load(f)
        self.min = np.array(scale_params['min_'], dtype=np.float32)
        self.scale_coef = 2.0 / (np.array(scale_params['max_'], dtype=np.float32) - self.min)

        self.kernel = gaussian_kernel2d(7, 7 / 6)
        self.C = 1 / 255
        self.prob_estimates = svmutil.c_double()

    def score(self, img):
        gray_image = skimage.color.rgb2gray(img)
        brisque_features = self.calculate_brisque_features(gray_image)
        downscaled_image = cv2.resize(gray_image, None, fx=0.5, fy=0.5)
        downscale_brisque_features = self.calculate_brisque_features(downscaled_image)
        brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
        return self.calculate_image_quality_score(brisque_features)

    def calculate_image_quality_score(self, brisque_features):
        scaled_brisque_features = self.scale_coef * (brisque_features - self.min) - 1
        x, idx = svmutil.gen_svm_nodearray(
            scaled_brisque_features,
            isKernel=(self.model.param.kernel_type == svmutil.PRECOMPUTED))
        return svmutil.libsvm.svm_predict_probability(self.model, x, self.prob_estimates)

    def calculate_mscn_coefficients(self, image):
        local_mean = local_mean_opencv(image, self.kernel)
        local_var = local_mean_opencv(image * image, self.kernel) - local_mean * local_mean
        local_std = np.sqrt(np.clip(local_var, 0, None))
        return (image - local_mean) / (local_std + self.C)

    def calculate_brisque_features(self, image):
        mscn_coefficients = self.calculate_mscn_coefficients(image)
        alpha, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(mscn_coefficients)
        var = (sigma_l ** 2 + sigma_r ** 2) / 2
        horizontal, vertical, main_diagonal, secondary_diagonal = compute_pairwise_products(mscn_coefficients)
        return np.array((alpha, var, *calculate_features(horizontal), *calculate_features(vertical),
                         *calculate_features(main_diagonal), *calculate_features(secondary_diagonal)), dtype=np.float32)
