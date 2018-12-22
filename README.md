# Depth Map Prediction from a Single Image
This code provides depth estimation function, while supporting Web deployment, through the Web page to try to single image depth estimation.

The depth prediction model adopted is [Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677)

## Requirements
This code was tested with Tensorflow 1.10.0, CUDA 9.0, falsk, Python 3 and Ubuntu 16.04

## How to try it on an image or an videos
Make sure your first put test images or video frames into a folder, and generate a list of file names on test_files_eigen.txt

You can test an image by running:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=/path/to/images/ \
--filenames_file=/path/to/test_files_eigen.txt \
--output_directory=/path/to/output_directory \
--checkpoint_path=/path/to/models/model_city2kitti.meta
```

## How to deploy it on Web
If you want to access the deployed web page from an external network, you should first modify the *host* and *port* for the *./visualization/app.py* file.

You can deploy it on Web by running:
```shell
CUDA_VISIBLE_DEVICES=0 python app.py
```

## Models
You can download our pre-trained models to an existing directory by running:
```shell
sh ./models/get_model.sh model_city2kitti/model_city2eigen ./models
```

## Acknowledgements
Thanks to [mrharicot](https://github.com/mrharicot), the initial code of *depthmodel* is references to his project [monodepth](https://github.com/mrharicot/monodepth).