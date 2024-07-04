import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import psutil
import subprocess
import streamlit as st
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, 
                                  non_max_suppression, scale_boxes, 
                                  xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Inisialisasi lokasi ROOT
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Atur nilai kendaraan 0
data_mobil = []
data_bus = []
data_truk = []
data_motor = []
data_jumlah = []
data = []
roi = None

def detect(opt, stframe, mobil, bus, truk, motor, jumlah, ram_text, cpu_text, gpu_text, fps_text, koordinat, class_id, hasil, model):
    source, yolo_model, deep_sort_model, show_vid, save_vid, imgsz, half, project, name, exist_ok= \
        opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.imgsz, opt.half, opt.project, opt.name, opt.exist_ok
    
    # Pilih Kelas
    opt.classes = class_id
    save_vid = hasil
    yolo_model = model
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    sum_fps = 0
    roi = koordinat
    roi = [tuple(map(int, coord.split(','))) for coord in roi]
    roi = np.array(roi).reshape((-1, 1, 2))
    
    # Cek GPU
    def get_gpu_memory():
        result = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=memory.used',
             '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory[0]

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    device = select_device(opt.device)
    half &= device.type != 'cpu'

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    vid_path, vid_writer = None, None

    if webcam:
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        cudnn.benchmark = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        prev_time = time.time()
        
        pred = model(img, augment=opt.augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                im0, _ = im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                im0, _ = im0s.copy(), getattr(dataset, 'frame', 0)

            save_path = str(save_dir / "file.mp4")
            s += '%gx%g ' % img.shape[2:]

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                if len(outputs) > 0:
                    for output, conf in zip(outputs, confs):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        count_obj(bboxes, id, names[c], roi)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            im0 = annotator.result()
            if show_vid:
                cv2.polylines(im0, [roi], isClosed=True, color=(0, 255, 0), thickness=2)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
                
                curr_time = time.time()
                fps_ = curr_time - prev_time
                fps_ = round(1/round(fps_, 3),1)
                prev_time = curr_time
                sum_fps += fps_

                stframe.image(im0, channels="BGR", use_column_width=True)
                mobil.write(f"##### Mobil: {str(len(data_mobil))}")
                bus.write(f"##### Bus: {str(len(data_bus))}")
                truk.write(f"##### Truk: {str(len(data_truk))}")
                motor.write(f"##### Motor: {str(len(data_motor))}")
                jumlah.write(f"##### Total Kendaraan: {str(len(data_jumlah))}")
                ram_text.write(f"##### RAM: {str(psutil.virtual_memory()[2])+'%'}")
                cpu_text.write(f"##### CPU: {str(psutil.cpu_percent())+'%'}")
                try:
                    gpu_text.write(f"##### Memori GPU: {str(get_gpu_memory())+' MB'}")
                except:
                    gpu_text.write(f"##### Memori GPU: {str('NA')}")
                fps_text.markdown(f"##### FPS: {fps_}")

            if save_vid:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                
                vid_writer.write(im0)
    
    # Simpan data perhitungan
    with open(save_dir/"data.txt", "w") as file:
        t = tuple(x / seen * 1E3 for x in dt)  
        if data_mobil:
            file.write(f"Jumlah Mobil: {len(data_mobil)}\n")
        if data_motor:
            file.write(f"Jumlah Motor: {len(data_motor)}\n")
        if data_truk:
            file.write(f"Jumlah Truk: {len(data_truk)}\n")
        if data_bus:
            file.write(f"Jumlah Bus: {len(data_bus)}\n")
        if data_jumlah:
            file.write(f"Total Jumlah: {len(data_jumlah)}\n")
        
        rata2 = round(1 / (sum(list(t)) / 1000), 1)
        file.write(f"Rata-rata FPS: {rata2}\n")
        file.write(f"Kecepatan: %.1fms Pre-process, %.1fms Inference, %.1fms NMS, %.1fms DeepSORT per gambar" % t)

def count_obj(box, id, label, roi):
    global data_mobil, data_bus, data_truk, data_motor, data_jumlah, data
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    if cv2.pointPolygonTest(roi, center_coordinates, False) >= 0:
        if id not in data:
            data_jumlah.append(id)
            data.append(id)
            if label == 'Mobil' and id not in data_mobil:
                data_mobil.append(id)
            elif label == 'Bus' and id not in data_bus:
                data_bus.append(id)
            elif label == 'Truk' and id not in data_truk:
                data_truk.append(id)
            elif label == 'Motor' and id not in data_motor:
                data_motor.append(id)

def reset():
    global data_mobil, data_bus, data_truk, data_motor, data_jumlah, data
    data_mobil = []
    data_bus = []
    data_truk = []
    data_motor = []
    data_jumlah = []
    data = []

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='./model/bestrb7e50.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_false', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_false', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_false', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'data/output', help='save results to project/name')
    parser.add_argument('--name', default='rekaman0', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

if __name__ == '__main__':
    opt = parse_opt()

    with torch.no_grad():
        detect(opt)
