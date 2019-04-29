# Pytorch_Adain_from_scratch
Unofficial Pytorch implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

Original torch implementation from the author can be found [here](https://github.com/xunhuang1995/AdaIN-style).

Other implementations such as [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) or [Chainer_implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN) are also available.I have learned a lot from them and try the pure Pytorch implementation from scratch in this repository.This repository provides a pre-trained model for you to generate your own image given content image and style image. Also, you can download the training dataset or prepare your own dataset to train the model from scratch.

I give a brief qiita blog and you can check it from [here]().

If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)

------

## Requirements

- Python 3.7
- PyTorch 1.0+
- TorchVision
- Pillow
- Skimage
- tqdm

Anaconda environment recommended here!

(optional)

- GPU environment for training



## Usage

------

## test

1. Clone this repository 

   ```bash
   git clone
   cd 
   ```

2. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

3. Generate the output image. A transferred output image and a content_output_pair image and a NST_demo_like image will be generated.

   ```python
   python test -c content_image_path -s style_image_path
   ```

   ```
   usage: test.py [-h] 
                  [--content CONTENT] 
                  [--style STYLE]
                  [--output_name OUTPUT_NAME] 
                  [--alpha ALPHA] 
                  [--gpu GPU]
                  [--model_state_path MODEL_STATE_PATH]
   
   
   ```

   If output_name is not given, it will use the combination of content image name and style image name.

------

## train

1. Download [COCO](http://cocodataset.org/#download) (as content dataset)and [Wikiart](https://www.kaggle.com/c/painter-by-numbers) (as style dataset) and unzip them, rename them as `content` and `style`  respectively (recommended).

2. Modify the argument in the` train.py` such as the path of directory, epoch, learning_rate or you can add your own training code.

3. Train the model using gpu.

4. ```python
   python train.py
   ```

   ```
   usage: train.py [-h] 
                   [--batch_size BATCH_SIZE] 
                   [--epoch EPOCH]
                   [--gpu GPU]
                   [--learning_rate LEARNING_RATE]
                   [--snapshot_interval SNAPSHOT_INTERVAL]
                   [--train_content_dir TRAIN_CONTENT_DIR]
                   [--train_style_dir TRAIN_STYLE_DIR]
                   [--test_content_dir TEST_CONTENT_DIR]
                   [--test_style_dir TEST_STYLE_DIR] 
                   [--save_dir SAVE_DIR]
                   [--reuse REUSE]
   ```

   

## Result

Some results of content image and my cat (called Sora) will be shown here.

![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res1.gif)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res2.gif)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res3.gif)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res4.gif)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res5.gif)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/resout_gif/res6.gif)




## References

- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) 
- [Chainer implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN)

