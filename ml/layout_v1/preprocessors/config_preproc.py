from typing import Any

import numpy as np
import numpy.typing as npt


class ConfigFeatureGenerator:
    OUTPUT_MASK = np.arange(6)
    INPUT_MASK = np.arange(6, 12)
    KERNEL_MASK = np.arange(12, 18)

    def __call__(
        self,
        config_features: npt.NDArray[Any],  # config is nc x c x 18
    ) -> npt.NDArray[Any]:
        output_features = config_features[:, :, self.OUTPUT_MASK]
        input_features = config_features[:, :, self.INPUT_MASK]
        kernel_features = config_features[:, :, self.KERNEL_MASK]

        # Compute default flags
        output_is_default = np.all(output_features == -1, axis=-1)
        input_is_default = np.all(input_features == -1, axis=-1)
        kernel_is_default = np.all(kernel_features == -1, axis=-1)

        # Get Active Dims (count of non-default dims)
        output_active_dims = np.sum(output_features != -1, axis=-1)
        input_active_dims = np.sum(input_features != -1, axis=-1)
        kernel_active_dims = np.sum(kernel_features != -1, axis=-1)

        # Get max order
        output_max_order = np.max(output_features, axis=-1)
        input_max_order = np.max(input_features, axis=-1)
        kernel_max_order = np.max(kernel_features, axis=-1)

        # Get contiguity (# of contiguous dims)
        # Temporarily ignore the -1s
        output_contiguity_count = self.calculate_contiguity(output_features)
        input_contiguity_count = self.calculate_contiguity(input_features)
        kernel_contiguity_count = self.calculate_contiguity(kernel_features)

        # is square
        output_is_square = (output_features[:, :, 0] == output_features[:, :, 1]) & (
            output_features[:, :, 0] != -1
        )
        input_is_square = (input_features[:, :, 0] == input_features[:, :, 1]) & (
            input_features[:, :, 0] != -1
        )
        kernel_is_square = (kernel_features[:, :, 0] == kernel_features[:, :, 1]) & (
            kernel_features[:, :, 0] != -1
        )

        # output_is_sequential_until_neg = np.all(
        #     output_features[:, :, :-1] == output_features[:, :, 1:], axis=-1
        # )
        # Calculate dimension variance
        output_dim_variance = self.calculate_variance(output_features)
        input_dim_variance = self.calculate_variance(input_features)
        kernel_dim_variance = self.calculate_variance(kernel_features)

        # Calculate dimension permutation
        output_dim_permutation = self.calculate_permutations(output_features)
        input_dim_permutation = self.calculate_permutations(input_features)
        kernel_dim_permutation = self.calculate_permutations(kernel_features)

        # Calculate layout similarity
        output_input_layout_similarity = self.calculate_layout_similarity(
            output_features, input_features
        )
        output_kernel_layout_similarity = self.calculate_layout_similarity(
            output_features, kernel_features
        )
        input_kernel_layout_similarity = self.calculate_layout_similarity(
            input_features, kernel_features
        )

        # Get contiguity rank (contiguity / active_dims)
        output_contiguity_rank = output_contiguity_count / (output_active_dims + 1e-4)
        input_contiguity_rank = input_contiguity_count / (input_active_dims + 1e-4)
        kernel_contiguity_rank = kernel_contiguity_count / (kernel_active_dims + 1e-4)

        # normalize counts
        output_active_dims = output_active_dims / 6
        input_active_dims = input_active_dims / 6
        kernel_active_dims = kernel_active_dims / 6

        # normalize orders
        output_max_order = output_max_order / 6
        input_max_order = input_max_order / 6
        kernel_max_order = kernel_max_order / 6

        # normalize contiguity
        output_contiguity_count = output_contiguity_count / 5
        input_contiguity_count = input_contiguity_count / 5
        kernel_contiguity_count = kernel_contiguity_count / 5

        # combine all features
        new_config_features = np.stack(
            [
                output_is_default,
                input_is_default,
                kernel_is_default,
                output_active_dims,
                input_active_dims,
                kernel_active_dims,
                output_max_order,
                input_max_order,
                kernel_max_order,
                output_contiguity_count,
                input_contiguity_count,
                kernel_contiguity_count,
                output_contiguity_rank,
                input_contiguity_rank,
                kernel_contiguity_rank,
                output_is_square,
                input_is_square,
                kernel_is_square,
                output_dim_variance,
                input_dim_variance,
                kernel_dim_variance,
                output_dim_permutation,
                input_dim_permutation,
                kernel_dim_permutation,
                output_input_layout_similarity,
                output_kernel_layout_similarity,
                input_kernel_layout_similarity,
            ],
            axis=-1,
        )

        # stack with original features
        config_features = np.concatenate(
            [config_features, new_config_features], axis=-1
        )
        return config_features

    def calculate_contiguity(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        valid_mask = features != -1

        # Apply the mask to keep only valid values and fill the rest with `np.nan` for later use with `np.diff`
        valid_features = np.where(valid_mask, features, np.nan)

        # Calculate the difference along the last axis, ignoring NaNs
        diffs = np.diff(valid_features, axis=-1, prepend=np.nan)

        # Check if the differences are 0 or 1 which indicates contiguity
        contiguity_mask = np.isin(diffs, [0, 1])

        # Sum over the last axis to get the count of contiguous elements, ignoring NaNs
        contiguity_count = np.nansum(contiguity_mask, axis=-1)

        return contiguity_count

    def calculate_variance(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculates variance along the last axis, ignoring -1s.
        The array is shape (n, m, 6) where n is the number of configs and m is the number of features.
        Our resulting array should be shape (n, m, 1) where each value is the variance of the 6 features,
        ignoring -1s.
        """
        mask = features != -1
        valid_features = np.where(mask, features, np.nan)
        variance = np.nanvar(valid_features, axis=-1)
        variance = np.nan_to_num(variance)

        return variance

    def calculate_permutations(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculates the number of permutations along the last axis, ignoring -1s"""

        def count_permutations(layout: npt.NDArray[Any]):
            layout = layout[layout != -1]  # remove padding
            return sum(
                1
                for i in range(len(layout))
                for j in range(i + 1, len(layout))
                if layout[i] > layout[j]
            )

        permutations = np.apply_along_axis(count_permutations, -1, features)
        return permutations

    def calculate_layout_similarity(
        self, features1: npt.NDArray[Any], features2: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Calculates the similarity between two layouts along the last axis, ignoring -1s"""
        # For simplicity, define layout similarity as the count of matching dimension indices
        similarity = np.sum(features1 == features2, axis=-1)
        return similarity
