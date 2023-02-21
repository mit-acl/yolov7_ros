import sys
import random
import pathlib

import numpy as np
import torch

# add yolov7 submodule to path
FILE_ABS_DIR = pathlib.Path(__file__).absolute().parent
YOLOV7_ROOT = (FILE_ABS_DIR / 'yolov7').as_posix()
if YOLOV7_ROOT not in sys.path:
    sys.path.append(YOLOV7_ROOT)

from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import letterbox
from models.experimental import attempt_load


class YoloV7:
    def __init__(self, weights, conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 img_size: int = 640, device: str = "cuda"):

        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__device = device
        self.__weights = weights

        self.model = attempt_load(self.__weights, map_location=device)
        self.__stride = int(self.model.stride.max())
        self.__img_size = check_img_size(img_size, s=self.__stride)
        self.__names = self.model.names

        self.__half = False
        
        self.model_info()

    def model_info(self, verbose=False, img_size=640):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in self.model.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.model.parameters() if x.requires_grad)  # number gradients
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name',
                        'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(self.model.named_parameters()):
                name = name.replace('module_list.', '')
                print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                        (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        try:  # FLOPS
            from thop import profile
            stride = max(int(self.model.stride.max()), 32) if hasattr(
                self.model, 'stride') else 32
            img = torch.zeros((1, self.model.yaml.get('ch', 3), stride, stride), device=next(
                self.model.parameters()).device)  # input
            flops = profile(deepcopy(self.model), inputs=(img,), verbose=False)[
                0] / 1E9 * 2  # stride GFLOPS
            img_size = img_size if isinstance(img_size, list) else [
                img_size, img_size]  # expand if int/float
            fs = ', %.1f GFLOPS' % (
                flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ''
        summary = f"\N{rocket}\N{rocket}\N{rocket} Yolov7 Detector summary:\n" \
            + f"Weights: {self.__weights}\n" \
            + f"Confidence Threshold: {self.__conf_thresh}\n" \
            + f"IOU Threshold: {self.__iou_thresh}\n"\
            + f"{len(list(self.model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}\n" \
            + f"Classes ({len(self.__names)}): {self.__names}"
        print(summary)

    @torch.no_grad()
    def _inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        pred_results = self.model(img)[0]
        detections = non_max_suppression(pred_results, conf_thres=self.__conf_thresh, iou_thres=self.__iou_thresh)

        if detections:
            detections = detections[0]

        return detections

    def _process_img(self, img0):

        # Padded resize (i.e., maintain aspect ratio)
        img = letterbox(img0, self.__img_size, stride=self.__stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # processing -- cf. yolov7/detect.py
        img = torch.from_numpy(img).to(self.__device)
        img = img.half() if self.__half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, img0):
        """
        Perform inference on an image to detect classes.
        
        Parameters
        ----------
        img0 : (h, w, c) np.array -- the input image

        Returns
        -------
        dets : (n, 6) np.array -- n detections
                Each detection is 2d bbox xyxy, confidence, class
        """

        # process the input image to make it appropriate for inference
        img = self._process_img(img0)

        # apply inference model to processed image
        dets = self._inference(img)

        # Rescale boxes from img_size to im0 size
        dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()

        return dets.cpu().detach().numpy()
