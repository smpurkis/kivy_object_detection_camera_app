import numpy as np
from math import ceil
import cv2
from kivy.logger import Logger
from kivy.graphics.texture import Texture

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def overlap_images(background, overlap):
    alpha = overlap[:, :, 3] / 255.0
    background[:, :, 0] = (1. - alpha) * background[:, :, 0] + alpha * overlap[:, :, 0]
    background[:, :, 1] = (1. - alpha) * background[:, :, 1] + alpha * overlap[:, :, 1]
    background[:, :, 2] = (1. - alpha) * background[:, :, 2] + alpha * overlap[:, :, 2]
    return background


def find_faces(self, platform):
    if (platform == 'android'):
        Logger.info(f"Camera: Rotating frame")
        self.frame = np.rot90(self.frame)
    Logger.info(f"Camera: frame size {self.frame.shape}")
    rect = cv2.resize(self.frame, (self.width, self.height))
    if (platform == 'android'):
        rect = cv2.flip(rect, 0)
    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
    self.model.setInput(cv2.dnn.blobFromImage(rect, 1 / image_std, (self.width, self.height), 127))
    boxes, scores = self.model.forward(["boxes", "scores"])
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = convert_locations_to_boxes(boxes, self.priors, center_variance, size_variance)
    boxes = center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(self.frame.shape[1], self.frame.shape[0], scores, boxes, self.threshold)
    Logger.info(f"Model: boxes detected {boxes}")
    return boxes


def draw_on_faces(self, boxes, platform):
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        if any([b < 0 for b in box]):
            break
        if self.thug_life:
            face = self.frame[box[1]:box[3], box[0]:box[2]]
            result = self.frame.copy()
            if (platform == 'android'):
                result = cv2.flip(result, 0)
            try:
                overlap_image = cv2.resize(self.overlap_image, dsize=(face.shape[1], face.shape[0]))
                transparent_background = np.zeros(self.frame.shape, dtype="uint8")
                transparent_background[box[1]:box[3], box[0]:box[2]] = overlap_image
                overlap_image = transparent_background
                result = overlap_images(result, overlap_image)
                if (platform == 'android'):
                    result = cv2.flip(result, 0)
                self.frame = result
            except Exception as e:
                Logger.info(f"Camera: Failed to resize overlap image {e}")
        else:
            cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
    return self.frame


def make_new_texture_frame(self):
    Logger.info(f"Camera: Displaying frame")
    if self.display_speed == 0:
        self.frame = cv2.resize(self.frame, (int(self.window_height * self.size_ratio), self.window_height))
    elif self.display_speed == 1:
        self.frame = cv2.resize(self.frame, (int(800 * self.size_ratio), 800))
    else:
        self.frame = cv2.resize(self.frame, (int(600 * self.size_ratio), 600))
    self.frame = self.frame.reshape((self.frame.shape[1], self.frame.shape[0], 4))
    buf = self.frame.tostring()
    Logger.info(f"Camera: converted to bytes {len(buf)}")
    texture1 = Texture.create(size=(self.frame.shape[0], self.frame.shape[1]), colorfmt='rgba')
    texture1.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
    return texture1
