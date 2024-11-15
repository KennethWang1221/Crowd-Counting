# Crowd Counting End-to-End Solution

This repository provides a purely point-based framework to directly predict the locations of crowd individuals.

## Requirements

To set up the environment, clone the repository and install the necessary packages:

```bash
conda create --name crowd_counting_env

conda activate crowd_counting_env

pip3 install ipykernel

python3 -m ipykernel install --user --name crowd_counting_env --display-name "crowd_counting_env"

pip3 install -r requirements.txt
```

## Data

This repository uses the [ShanghaiTech Crowd Counting Dataset](https://github.com/desenzhou/ShanghaiTechDataset) for demonstration. To download the dataset, please visit the link above.

## Usage

To run the model on an image folder, use the following command:

```bash
python3 ./test_onnx.py
```

To run the model on a video, use the following command:

```bash
python3 ./test_nbg_video.py
```

The scripts will save the visualized results.

### Visualized Image

![assets](./assets/IMG_133.png)

*Note: This is a demonstration. For higher accuracy, please customize the training strategy.*

## References

The torch weight is from [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)

