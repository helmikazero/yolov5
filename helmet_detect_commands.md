

python detect.py --img 416 --weights runs\train\HelmetDetection\weights\best.onnx --source 0

python helmet_detect.py --img 416 --weights runs\train\HelmetDetection\weights\best.pt --source 0
python helmet_detect.py --img 416 --weights runs\train\HelmetDetection\weights\best.onnx --source 0

python detect.py --img 416 --weights runs\train\HelmetDetection\weights\best.pt --source 0
python detect.py --img 416 --weights runs\train\HelmetDetection\weights\best.onnx --source 0

python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5n\weights\best.pt --source 0
python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5n\weights\best.onnx --source 0

python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5m\weights\best.pt --source 0
python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5m\weights\best.onnx --source 0

python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5l\weights\best.pt --source 0
python detect.py --img 416 --weights runs\train\HelmetDetection_yolov5l\weights\best.onnx --source 0

python export.py --weights runs\train\HelmetDetection\weights\best.pt --include onnx --dynamic

python export.py --weights runs\train\HelmetDetection_yolov5l\weights\best.pt --include onnx --dynamic

python export.py --weights runs\train\HelmetDetection_yolov5m\weights\best.pt --include onnx --dynamic

python export.py --weights runs\train\HelmetDetection_yolov5n\weights\best.pt --include onnx --dynamic