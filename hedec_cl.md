python hedect_main.py --weights weightHedect\hedec_yolov5s.pt --nosave --imgsz 640 --source 0

FOR UBUNTU
python3 hedect_noaudio.py --weights weightHedect/hedec_yolov5s.pt --nosave --imgsz 240 --source 0

python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pretrain_N.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pretrain_S.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pretrain_M.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pretrain_L.pt --nosave --imgsz 256 --source 0

python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pure_N.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pure_S.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pure_M.pt --nosave --imgsz 256 --source 0
python3 hedect_noaudio.py --weights weightHedect/FINAL_WEIGHTS/hedec_pure_L.pt --nosave --imgsz 256 --source 0