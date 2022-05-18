

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

python helmet_detect2.py --img 640 --weights runs\train\HelmetDetection\weights\best.pt --source test_image\hard_hat_workers43_png.rf.1ef7bacdf272ebe4a911dfbe1926fcb8.jpg

python helmet_detect2.py --img 640 --weights custom_checkpoint\hd_yolov5m.pt --source 0 --nosave


python helmet_detect2.py --img 640 --weights runs\train\HelmetDetection\weights\best.pt --source test_image\hard_hat_workers262_png_jpg.rf.f47ec92314bd030b8a77b17c9577c9e0.jpg

python helmet_detect2.py --img 640 --weights custom_checkpoint\hd_yolov5s.pt --source test_image\video1.mp4

python helmet_detect2.py --img 640 --weights custom_checkpoint\hd_yolov5s.pt --source test_image\video2.mp4

python helmet_detect2.py --img 640 --weights custom_checkpoint\hd_yolov5s.pt --source 0

python helmet_detect2.py --img 640 --weights custom_checkpoint\hd_yolov5s.pt --source test_image\over5mallheadgear.mp4

python helmet_detect2.py --img 640 --weights models\models_\hedec_yolov5s.pt --source 0


python helmet_detect2.py --img 480 --weights models\models_\hedec_yolov5s.pt --source 'rtps://192.168.0.118:1935'

python helmet_detect2.py --img 240 --weights hedec_weight\hedec_yolov5s.pt --source 0

python helmet_detect2.py --img 240 --weights hedec_weight\hedec_yolov5s_pure.pt --source 0