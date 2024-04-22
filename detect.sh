python detect.py \
    --data_config configs/hagrid.yaml \
    --cls_weight output/gelans_192x192_batch_64/weight/best.onnx \
    --det_weight yolov7-tiny-diver.onnx \
    --data_path data/diver.avi