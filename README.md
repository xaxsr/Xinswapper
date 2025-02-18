## Xinswapper
Train or finetune the insightface inswapper_128 model on a custom dataset.

Xinswapper is my personal attempt to recreate and train the insightface inswapper_128 using PyTorch, implementing the same architecture so it may be used to finetune or train the original inswapper_128 model.

After finetuning/training, you may use the included conversion script to convert the finetuned Xinswapper pytorch model back to ONNX in the same format expected by tools that implement the original model such as Rope or facefusion. After you've converted the finetuned PyTorch model to ONNX, you simply replace the old inswapper_128.onnx with the new finetuned model.

>##### __Note/Disclaimer: Xinswapper is my own personal implementation of the same architecture used in inswapper_128 in an attempt to finetune the original model from insightface. I am not affiliated with insightface and this is not the original training code for inswapper_128 and still has much room for improvement. This is intended for responsible and ethical use only.__ 

---
## Comparison

Below is an example of the difference between my code and the original from a test run finetuning the original using only a small number of different identities in the dataset and <1000 steps
<p align="center">
<img src="./asset/1.jpg" width=100%>
<img src="./asset/2.jpg" width=100%>
</p>


---
## Training
### 1. prepare environment
Download required models/weights from [here](https://github.com/xaxsr/Xinswapper/releases/tag/weights) and put them into `./weights/`:

- inswapper_128.onnx
- det_10g.onnx
- w600k_r50.onnx
- WFLW_4HG.pth


```bash
# clone repo
git clone https://github.com/xaxsr/Xinswapper.git
cd Xinswapper
git submodule init
git submodule update

# create environment (tested with python 3.10)
python -m venv venv
source venv/bin/activate

# install dependancies 
pip install -r requirements.txt

# convert w600k_r50 to pytorch
python -c "import torch; import onnx; from onnx2torch import convert; onnx_model = onnx.load('./weights/w600k_r50.onnx'); pytorch_model = convert(onnx_model); torch.save(pytorch_model, './weights/w600k_r50.pt')"
```

### 2. prepare dataset
Use `crop_align.py` to crop and align faces from your images (you may crop to any size as images are resized during training. use `--help` for more option)

```bash
python crop_align.py --input_dir './raw_images/' --output_dir './raw_images/aligned/'
```

Organize your aligned faces based on identity using this folder structure:
    
    .
    ├── dataset root         
    │   ├── 00001 (id #1)                  
    │   │   └── aligned_face1.jpg  
    │   │   └── aligned_face2.jpg  
    │   │   └── aligned_face3.jpg  
    │   │   └── ...
    │   ├── 00002 (id #2)                   
    │   │   └── aligned_face1.jpg  
    │   │   └── aligned_face2.jpg  
    │   │   └── aligned_face3.jpg  
    │   │   └── ...
    │   └── ...
  

### 2. prepare weights from inswapper_128.onnx
Use `convert_to_pytorch.py` script to prepare the weights

```bash
python convert_to_pytorch.py --onnx_file './weights/inswapper_128.onnx' --output_file './weights/inswapper_128.pth'
```

### 3. training
1. use `train.py` to start training
    ```bash
    python train.py \
        --pretrained True\
        --name 'my_run' \
        --batch_size 10 \
        --data_root './dataset_root/' \
        --log_path './logs/' \
        --checkpoint_path './checkpoints/' \
        --inswapper_path './weights/inswapper_128.pth' \
        --recognition_model_path './weights/w600k_r50.pt' \
        --fan_path './weights/WFLW_4HG.pth' \
        --lr_g 2e-6 \
        --lr_d 5e-6
   
   ```
   use `--resume True` to load an existing run
   
   the defaults are set to values used in my personal experimentations and may still have room for improvement. use `--help` for more option

### 4. convert back to onnx
Use `convert_to_onnx.py` script to convert your finetuned model back to original form

```bash
python convert_to_onnx.py --checkpoint_file './checkpoints/my_run/my_run_netG.pth' --output_file './checkpoints/my_run/inswapper_128.onnx'
```
you may now replace the original inswapper_128.onnx wherever you were using it with the new finetuned inswapper_128.onnx

---

### acknowledgements
- [insightface](https://insightface.ai/) - for the original model
- [Rope](https://github.com/Hillobar/Rope) - borrowed some code for cropping/alignment
- [SimSwap](https://github.com/neuralchen/SimSwap) using the projected discriminator from SimSwap
- [GHOST](https://github.com/ai-forever/ghost) - applied some of the same methodology for training (incl. AWL for eye landmark)
