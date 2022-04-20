# PR-DAD
PyTorch implementation of the fellwoing paper: [PR-DAD: Phase Retrieval Using Deep Auto-Decoders](https://www.shaidekel.com/_files/ugd/1de1d9_54de55d7e23c49e091bd4b6aaa2ecf05.pdf) join with [prof. Shai Dekel](https://www.shaidekel.com/), School of mathematical sciences, Tel-Aviv University
<!-- ## Overview -->

## Algorithm Pipeline
## Package Intsallation
 - Python3 =>3.8
 - PyTorch =>1.9
 - Cuda =>11
 - Hardware requred: Ubuntu, NVIDIA Tesla V100 16Gib and 8x Intel Xeon E5-2686v4, recomeded AWS EC2 type: `p3.2xlarge`
 - Requred packages `requirements.txt`
## Train Model
We agregate For each dataset per type of features we trained model and json config with hyperparameters in [Table](https://github.com/gugas81/pr-dad/edit/master/README.md#trained-models)
 - Run trainer:
 Trainer uses [ClearML](https://app.clear.ml/) logger.
  ```
 python training/phase_retrieval_trainer.py --experiment_name my-experiment --config_path url_path/config-trainer.json **kwargs
 ```
 `my-experiment` - ClearML experiment name
 
 `config_path` - path(local/s3) to json with trainings hyperparameters
 
 `**kwargs` (optinal) - change spefic parameters
 Example:
 ```
 python training/phase_retrieval_trainer.py 
   --experiment_name my-experiment 
   --config_path s3://url_path/config-trainer.json 
   --path_pretrained s3://model_url/model.pt
   --batch_size 16
   --ae_type wavelet-net 
   --wavelet_type haar
 ```
 - Run evaluation: 
```
 python training/phase_retrival_evaluator.py --model_type path_to_model/model.pt --config  url_path/config-trainer.json 
 ```
### Trained models and Configurations
| Dataset | Haar Features  | ConvNet Features  | 
| --- | --- | --- |
| MNIST | [Model](https://pr-dad.s3.amazonaws.com/mnist/2022_02_01_19_22_47-ae-features-prediction-mnist-rfft-pad-wavelet-ae-haar-deep3-no-ref-net-dwt-coeff-loss-special.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/mnist/2022_02_01_19_22_47-ae-features-prediction-mnist-rfft-pad-wavelet-ae-haar-deep3-no-ref-net-dwt-coeff-loss-special.json) | [Model](https://pr-dad.s3.amazonaws.com/mnist/2022_04_09_20_01_43-ae-features-prediction-mnist-pad050-features64-int-f-128-spetial-pred-epoch100.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/mnist/2022_04_09_20_01_43-ae-features-prediction-mnist-pad050-features64-int-f-128-spetial-pred-epoch100.json)|
| EMNIST | [Model](https://pr-dad.s3.amazonaws.com/emnist/2022_02_27_20_26_49-ae-features-prediction-emnist-rfft-pad05-no-gan-no-refnet-prelu-spec-norm0125-spetial-wavelet-haar.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/emnist/2022_02_27_20_26_49-ae-features-prediction-emnist-rfft-pad05-no-gan-no-refnet-prelu-spec-norm0125-spetial-wavelet-haar.json) | [Model](https://pr-dad.s3.amazonaws.com/emnist/2022_01_02_13_54_52-ae-features-prediction-emnist-rfft-pad05-no-gan-prelu-dct-out-spec-norm025-ae-train.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/emnist/2022_01_02_13_54_52-ae-features-prediction-emnist-rfft-pad05-no-gan-prelu-dct-out-spec-norm025-ae-train.json)|
| KMNIST | [Model](https://pr-dad.s3.amazonaws.com/kmnist/022_02_14_10_25_22-ae-features-prediction-kmnist-rfft-pad05-no-gan-predict-conv-block-lmabda-recon-magl160-activ-fc-prelu-spetial.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/kmnist/2022_02_14_10_25_22-ae-features-prediction-kmnist-rfft-pad05-no-gan-predict-conv-block-lmabda-recon-magl160-activ-fc-prelu-spetial.json) | [Model](https://pr-dad.s3.amazonaws.com/kmnist/022_02_14_10_25_22-ae-features-prediction-kmnist-rfft-pad05-no-gan-predict-conv-block-lmabda-recon-magl160-activ-fc-prelu-spetial.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/kmnist/2022_02_14_10_25_22-ae-features-prediction-kmnist-rfft-pad05-no-gan-predict-conv-block-lmabda-recon-magl160-activ-fc-prelu-spetial.json)|
| Fashion-MNIST | [Model](https://pr-dad.s3.amazonaws.com/fashion-mnist/2022_01_30_22_49_05-ae-features-prediction-fashion-mnist-rfft-fc-relu-augprob025-pad025-wavelet-ae-haar-deep5-no-gan-special-lmbdaf-40.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/fashion-mnist/2022_01_30_22_49_05-ae-features-prediction-fashion-mnist-rfft-fc-relu-augprob025-pad025-wavelet-ae-haar-deep5-no-gan-special-lmbdaf-40.json) | [Model](https://pr-dad.s3.amazonaws.com/fashion-mnist/022_01_11_11_13_13-ae-features-prediction-fashion-mnist-rfft-fc-prelu-dct-scale025-augprob025-pad025-decoder-finetune-no-gan-with-ref-unet-lr5e-06.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/fashion-mnist/2022_01_11_11_13_13-ae-features-prediction-fashion-mnist-rfft-fc-prelu-dct-scale025-augprob025-pad025-decoder-finetune-no-gan-with-ref-unet-lr5e-06.json)|
| CelebA | [Model](https://pr-dad.s3.amazonaws.com/celeba64/2022_02_05_17_05_24-ae-features-prediction-celeba-celeb64-cop-rfft-f-predict-ae256-fc_multi-coeff2-nogan-batch-tr64-use_aug_tr_with_small_gamma_correct_prob050-inter-activ-relu-wavelet-ae-haar-deep5-specail.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/celeba64/2022_02_05_17_05_24-ae-features-prediction-celeba-celeb64-cop-rfft-f-predict-ae256-fc_multi-coeff2-nogan-batch-tr64-use_aug_tr_with_small_gamma_correct_prob050-inter-activ-relu-wavelet-ae-haar-deep5-specail.json) | [Model](https://pr-dad.s3.amazonaws.com/celeba64/2022_02_08_22_38_38-ae-features-prediction-celeba-celeb64crop-rfft-f-predict-ae256-fc_multi-coeff2-nogan-batch-tr64-use_aug_tr_with_small_gamma_correct_prob050-inter-activ-relu-special-decoder-finetune.pt), [Config-Trainer](https://pr-dad.s3.amazonaws.com/celeba64/2022_02_08_22_38_38-ae-features-prediction-celeba-celeb64crop-rfft-f-predict-ae256-fc_multi-coeff2-nogan-batch-tr64-use_aug_tr_with_small_gamma_correct_prob050-inter-activ-relu-special-decoder-finetune.json)|
