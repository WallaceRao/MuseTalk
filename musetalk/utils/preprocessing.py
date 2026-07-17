import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# PyTorch 2.6+ defaults to weights_only=True; allow trusted local checkpoints.
_original_torch_load = torch.load

def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range
    

def _detect_bbox_for_frame(frame, upperbondrange=0):
    """Run DWPose + face detection on a single frame. Returns (bbox, range_minus, range_plus)."""
    results = inference_topdown(model, frame)
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    face_land_mark = keypoints[0][23:91]
    face_land_mark = face_land_mark.astype(np.int32)

    bbox = fa.get_detections_for_batch(np.asarray([frame]))
    f = bbox[0]
    if f is None:
        return coord_placeholder, None, None

    half_face_coord = face_land_mark[29].copy()
    range_minus = (face_land_mark[30] - face_land_mark[29])[1]
    range_plus = (face_land_mark[29] - face_land_mark[28])[1]
    if upperbondrange != 0:
        half_face_coord[1] = upperbondrange + half_face_coord[1]
    half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
    upper_bond = max(0, half_face_coord[1] - half_face_dist)

    f_landmark = (
        np.min(face_land_mark[:, 0]),
        int(upper_bond),
        np.max(face_land_mark[:, 0]),
        np.max(face_land_mark[:, 1]),
    )
    x1, y1, x2, y2 = f_landmark
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        print("error bbox:", f)
        return f, range_minus, range_plus
    return f_landmark, range_minus, range_plus


def _lerp_bbox(b0, b1, t):
    if b0 == coord_placeholder or b1 == coord_placeholder:
        return b0 if t < 0.5 else b1
    return tuple(int(round(b0[j] + (b1[j] - b0[j]) * t)) for j in range(4))


def _interpolate_sparse_coords(sparse_coords, n_frames):
    """Fill None slots by linear interpolation between detected keyframes."""
    result = [coord_placeholder] * n_frames
    key_idxs = [i for i, c in enumerate(sparse_coords) if c is not None]
    if not key_idxs:
        return result

    first = key_idxs[0]
    for i in range(first):
        result[i] = sparse_coords[first]

    for k, i0 in enumerate(key_idxs):
        result[i0] = sparse_coords[i0]
        if k + 1 >= len(key_idxs):
            for i in range(i0 + 1, n_frames):
                result[i] = sparse_coords[i0]
            break
        i1 = key_idxs[k + 1]
        b0, b1 = sparse_coords[i0], sparse_coords[i1]
        span = i1 - i0
        for i in range(i0 + 1, i1):
            result[i] = _lerp_bbox(b0, b1, (i - i0) / span)
    return result


def get_landmark_and_bbox(img_list=None, upperbondrange=0, detect_stride=3, frames=None):
    """
    Detect face bboxes. When detect_stride > 1, only every N-th frame (and the
    last frame) is detected; intermediate frames use linear bbox interpolation.

    Pass either img_list (paths) or frames (already decoded BGR arrays).
    """
    if frames is None:
        if not img_list:
            raise ValueError("Either img_list or frames must be provided")
        frames = read_imgs(img_list)
    n_frames = len(frames)
    detect_stride = max(1, int(detect_stride))

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    if detect_stride > 1:
        print(f'bbox detect_stride={detect_stride} (intermediate frames interpolated)')

    sparse_coords = [None] * n_frames
    average_range_minus = []
    average_range_plus = []

    detect_indices = list(range(0, n_frames, detect_stride))
    if n_frames > 0 and (n_frames - 1) not in detect_indices:
        detect_indices.append(n_frames - 1)

    for idx in tqdm(detect_indices):
        bbox, range_minus, range_plus = _detect_bbox_for_frame(frames[idx], upperbondrange)
        sparse_coords[idx] = bbox
        if range_minus is not None:
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)

    coords_list = (
        sparse_coords
        if detect_stride == 1
        else _interpolate_sparse_coords(sparse_coords, n_frames)
    )

    print("********************************************bbox_shift parameter adjustment**********************************************************")
    if average_range_minus:
        print(
            f"Total frame:「{n_frames}」 Manually adjust range : "
            f"[ -{int(sum(average_range_minus) / len(average_range_minus))}"
            f"~{int(sum(average_range_plus) / len(average_range_plus))} ] , "
            f"the current value: {upperbondrange}"
        )
    else:
        print(f"Total frame:「{n_frames}」 No valid face ranges, current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
