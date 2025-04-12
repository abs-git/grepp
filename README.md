## grepp test

### install
```shell
cd ~/grepp

docker build -f docker/dev.Dockerfile -t grepp:latest .
docker run -it --rm --gpus all \
           -v $(pwd):/workspace/grepp \
           grepp:latest /bin/bash
```

### train
```shell
cd /workspace/grepp
export PYTHONPATH=/worksapce:${PYTHONPATH}
torchrun --nproc_per_node=1 train/trainer.py  \
                            --config config/base.yaml \
                            --checkpoint model/conv_net.pth \
                            --output_dir outputs
```

### deploy to onnx & test
```shell
cd /workspace/grepp
export PYTHONPATH=/worksapce:${PYTHONPATH}

python3 deploy/deploy-onnx.py --checkpoint_path outputs/output_8/best.pt \
                              --output_dir outputs/output_8 \
                              --image_dir data/val/apple/

python3 deploy/test-onnx.py --onnx_path outputs/output_8/best_int8.onnx \
                            --image_dir data/test/

```
