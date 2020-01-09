# Pytorch_Adain_from_scratch
Unofficial Pytorch implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

Original torch implementation from the author can be found [here](https://github.com/xunhuang1995/AdaIN-style).

Other implementations such as [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) or [Chainer_implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN) are also available.I have learned a lot from them and try the pure Pytorch implementation from scratch in this repository.This repository provides a pre-trained model for you to generate your own image given content image and style image. Also, you can download the training dataset or prepare your own dataset to train the model from scratch.

I give a brief qiita blog and you can check it from [here](https://qiita.com/edad811/items/02ca5292276572f9dad8).

If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)

## Notice
I propose a structure-emphasized multimodal style transfer(SEMST), feel free to use it [here](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer).

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
   git clone https://github.com/irasin/Pytorch_Adain_from_scratch
   cd Pytorch_Adain_from_scratch
   ```

2. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

3. Download the pretrained model [here](https://drive.google.com/file/d/1aTS_O3FfLzq5peh20vbWfU4kNAnng6UT/view?usp=sharing)
4. Generate the output image. A transferred output image and a content_output_pair image and a NST_demo_like image will be generated.

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

Some results will be shown here.

![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_03_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_04_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_05_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_088_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_101308_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_10_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_1348_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_1_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_27_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_3314_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_876_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_8_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_antimonocromatismo_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_asheville_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_brick1_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_brick_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_bridge_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_brushstrokers_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_candy_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_charles-reid-art-04_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_composition_vii_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_contrast_of_forms_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_en_campo_gris_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_escher_sphere_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_feathers_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_flower_of_life_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_frida_kahlo_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_horse_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_hosi_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_house_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_hs6_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_impronte_d_artista_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_in1_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_in2_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_la_muse_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_mondrian_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_mosaic_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_mosaic_ducks_massimo_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_news1_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_picasso_seated_nude_hr_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_plum_flower_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_rain-princess_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_Robert_Delaunay_1906_Portrait_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_scene_de_rue_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_seated-nude_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_shipwreck_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_sketch_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_stars2_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_strip_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_Sunset_in_Venice_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_the_resevoir_at_poitiers_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_trial_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_udnie_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_wave_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_woman_in_peasant_dress_cropped_demo.jpg)
![image](https://github.com/irasin/Pytorch_Adain_from_scratch/blob/master/res/IMG_0565_woman_with_hat_matisse_demo.jpg)




## References

- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) 
- [Chainer implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN)

