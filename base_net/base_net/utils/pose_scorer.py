import math
import torch
from torch import Tensor

class PoseScorer:
    def __init__(self, max_angular_window = torch.pi/2, num_position_connections = 8):
        if num_position_connections not in [0, 4, 8]:
            raise ValueError(f'num_position_connections must be 0, 4, or 8. You gave {num_position_connections}')
        
        if num_position_connections == 0:
            self._max_score = 1.0
        elif num_position_connections == 4:
            self._max_score = (1.0 + 4/(1 + math.sqrt(1))) / 5
        elif num_position_connections == 8:
            self._max_score = (1.0 + 4/(1 + math.sqrt(1)) + 4/(1 + math.sqrt(2))) / 9
        
        self._max_angular_window = max_angular_window
        self._num_position_connections = num_position_connections

    def score_pose_array(self, pose_array: Tensor) -> Tensor:
        """
        Args:
            pose_array: Boolean tensor of base poses aranged as (N, ny, nx, nt) 
                        for batch size N, and coordinates x, y, theta
        Returns:
            torch.Tensor: Same size as the input with floating point values 
                          indicating the scores of each pose
        """
        # We only score layers that have valid poses
        output_scores = torch.zeros_like(pose_array, dtype=torch.float32)
        valid_layer_mask = torch.any(pose_array.flatten(start_dim=1), dim=1)
        if not torch.any(valid_layer_mask):
            return output_scores
        pose_array = pose_array[valid_layer_mask]

        # Start by getting the score for every orientation at each position
        pose_scores = self.score_at_positions(pose_array.view(-1, pose_array.size(3))).reshape(pose_array.shape)
        augmented_pose_scores = pose_scores.clone().detach()

        # Then incorporate neighboring scores based on the number of neighbors to consider
        position_offsets = []
        if self._num_position_connections > 0:
            position_offsets = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
        if self._num_position_connections > 4:
            position_offsets += [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]

        # Algorithm for adding neighboring values in a grid
        for offset_vec in position_offsets:
            pads = []
            slices = []
            dist = torch.norm(torch.tensor(offset_vec, dtype=torch.float))
            for offset_val in offset_vec:
                start_padding = 0 if offset_val >= 0 else abs(offset_val)
                end_padding   = 0 if offset_val  < 0 else abs(offset_val)
                pads.extend([end_padding, start_padding])

                start_slice = offset_val if offset_val >= 0 else 0
                end_slice   = None       if offset_val >= 0 else offset_val
                slices.append(slice(start_slice, end_slice))

            # Pads are reversed to account for the reverse order of indices in torch.nn.functional.pad
            padded_scores = torch.nn.functional.pad(pose_scores, pad=list(reversed(pads)))
            offset_tensor = padded_scores[(slice(None), *slices)]

            # Use the pose_array mask to only update valid poses, and weight by inverse distance
            augmented_pose_scores += pose_array * offset_tensor / (1 + dist)

        # Normalize and reshape to the same as the input shape
        output_scores[valid_layer_mask] = augmented_pose_scores / (self._max_score * (len(position_offsets) + 1))
        return output_scores
        
    def score_at_positions(self, valid_angle_mask: Tensor) -> Tensor:
        """
        Args:
            valid_angle_mask: Boolean tensor of valid orientations at a given
                              position of shape (N, nt)
        Returns:
            torch.Tensor: Tensor of float scores indicating the number of orientations
                          which are valid on BOTH sides of the current index for each 
                          orientation index. Size (N, nt)        
        """

        batch_size, num_angles = valid_angle_mask.shape
        pos_increment = 2*torch.pi / num_angles
        index_offset_max = int(min(self._max_angular_window / (2*pos_increment), (num_angles + 1) // 2))

        batch_indices = torch.arange(batch_size, dtype=torch.int64, device=valid_angle_mask.device).unsqueeze(1).repeat(1, num_angles)
        current_index = torch.arange(num_angles, dtype=torch.int64, device=valid_angle_mask.device).unsqueeze(0).repeat(batch_size, 1)
        valid_mask = valid_angle_mask.clone().detach()
        scores = torch.zeros_like(valid_angle_mask, dtype=torch.float32, requires_grad=False)

        index_offset = 0
        while index_offset <= index_offset_max and torch.any(valid_mask):
            index_offset += 1
            scores += valid_mask.int()

            positive_offset_indices = torch.remainder(current_index + index_offset, num_angles)
            negative_offset_indices = torch.remainder(current_index - index_offset, num_angles)
            valid_mask = torch.logical_and(valid_mask, valid_angle_mask[batch_indices, positive_offset_indices])
            valid_mask = torch.logical_and(valid_mask, valid_angle_mask[batch_indices, negative_offset_indices])

        return scores / (index_offset_max + 1)
    
    def select_best_pose(self, pose_array: Tensor, already_scored: bool = False) -> tuple[Tensor, Tensor]:
        pose_scores = pose_array if already_scored else self.score_pose_array(pose_array) 
        
        index_tensor = -1*torch.ones(pose_array.size(0), dtype=int, device=pose_array.device)
        for idx, layer_pose_scores in enumerate(pose_scores):
            if not torch.any(layer_pose_scores):
                continue
            
            layer_pose_scores = (layer_pose_scores/torch.max(layer_pose_scores) >= 0.99).float()

            iteration_diff = torch.ones_like(layer_pose_scores, dtype=torch.float32)
            while torch.any(iteration_diff > 1e-2):
                new_scores = self.score_pose_array(layer_pose_scores.unsqueeze(0)).squeeze(0)
                new_scores = (new_scores/torch.max(new_scores) >= 0.99).float()

                iteration_diff = torch.abs(new_scores - layer_pose_scores)
                layer_pose_scores = new_scores
                    
            layer_pose_scores = self.score_pose_array(layer_pose_scores.unsqueeze(0)).squeeze(0)
            index_tensor[idx] = layer_pose_scores.argmax()

        batch_indices = torch.arange(pose_scores.size(0), device=index_tensor.device)
        return batch_indices, index_tensor