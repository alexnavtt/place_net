# Place Net Ros Params Parameters

Default Config
```yaml
place_net_ros_params:
  ros__parameters:
    checkpoint_path: ''
    device: 0.0
    inverse_reachability_map_path: ''
    max_batch_size: 0.0
    max_ik_count: 0.0
    visualize: true
    world_frame: map

```

## checkpoint_path

Absolute path to the model checkpoint to load

* Type: `string`
* Default Value: ""
* Read only: True

## inverse_reachability_map_path

Absolute path to the inverse reachability map to load

* Type: `string`
* Default Value: ""
* Read only: True

## device

Index of the CUDA device to use. CPU and other GPU architectures are not supported due to reliance on cuRobo

* Type: `int`
* Default Value: 0
* Read only: True

*Constraints:*
 - greater than or equal to 0

*Additional Constraints:*



## max_batch_size

The maximum number of task poses to run in parallel as a batch. Reduce if you are running into 'out of memory' problems. Set to zero for unlimited

* Type: `int`
* Default Value: 0
* Read only: True

*Constraints:*
 - greater than or equal to 0

*Additional Constraints:*



## world_frame

The gravity-aligned world frame on which z=0 indicates the bottom of the robot

* Type: `string`
* Default Value: "map"
* Read only: True

## max_ik_count

Maximum number of IK solutions to calculate in parallel for ground truth calculations

* Type: `int`
* Default Value: 0
* Read only: True

## visualize

Whether or not to publish visualization topics, which can take some time to process

* Type: `bool`
* Default Value: true
* Read only: True

