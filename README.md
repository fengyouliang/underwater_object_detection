# underwater object detection

## Results and Models

| model name | local map | leardboard A map | config | remark |
|:---------:|:-------:|:-------:|:--------:|:--------:|
| faster_rcnn | 0.466 | 0.45530633 | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/faster_rcnn.py) | base faster rcnn |
| cascade_rcnn | 0.4790 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/cascade_rcnn.py) | base cascade rcnn |
| atss | 0.484 | 0.48804623 | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/atss.py) | base atss |
| cascade_rcnn_mstrain | 0.4930 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/cascade_rcnn_mstrain.py) | base cascade rcnn + mstrain |
| dcn_cascade_rcnn | 0.4910 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/dcn_cascade_rcnn.py) | base cascade rcnn + dcn |
| detectors_cascade_rcnn | 0.4910 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/detectors_cascade_rcnn.py) | base cascade rcnn + detectors |
| fcos | 0.4690 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/fcos.py) | base fcos |
| gfl | 0.4950 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/gfl.py) | base gfl |
| gfl_cb | 0.4960 | - | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/gfl_cb.py) | base gfl + class balance(1e-3) |
| gfl_mstrain| 0.5040 | 0.51099049 | [github](https://github.com/fengyouliang/underwater_object_detection/blob/master/configs/gfl_mstrain.py) | base gfl + mstrain |


### Notes
- base: schedules-1x, backbone-resnet50
- mstrain: multi scale train
- cb: class balance