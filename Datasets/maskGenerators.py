from typing import Any
import math
import random
import hashlib
import logging
from enum import Enum

import cv2
import numpy as np


"""
Irregular, Rectangular and Outpainting masks copied from
    https://github.com/advimman/lama/blob/main/saicinpainting/training/data/masks.py
Semantic masks adpated and extended from
    https://github.com/ericsujw/LGPN-net/blob/main/src/dataset.py#L438
"""


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class DrawMethod(Enum):
    LINE = "line"
    CIRCLE = "circle"
    SQUARE = "square"


def make_random_irregular_mask(
    shape,
    max_angle=4,
    max_len=60,
    max_width=20,
    min_times=0,
    max_times=10,
    draw_method=DrawMethod.LINE,
):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip(
                (start_x + length * np.sin(angle)).astype(np.int32), 0, width
            )
            end_y = np.clip(
                (start_y + length * np.cos(angle)).astype(np.int32), 0, height
            )
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(
                    mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1
                )
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[
                    start_y - radius : start_y + radius,
                    start_x - radius : start_x + radius,
                ] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(
        self,
        max_angle=120,
        max_len=80,#50,#80,
        max_width=50,#30,#50,
        min_times=5,
        max_times=25,
        ramp_kwargs=None,
        draw_method=DrawMethod.LINE,
    ):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None, **kwargs):
        coef = (
            self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        )
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(
            self.min_times + 1 + (self.max_times - self.min_times) * coef
        )
        return make_random_irregular_mask(
            img.shape[1:],
            max_angle=self.max_angle,
            max_len=cur_max_len,
            max_width=cur_max_width,
            min_times=self.min_times,
            max_times=cur_max_times,
            draw_method=self.draw_method,
        )


def make_random_rectangle_mask(
    shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3
):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y : start_y + box_height, start_x : start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(
        self,
        margin=10,
        bbox_min_size=30,#15,#30,
        bbox_max_size=80,#50,#80,
        min_times=5,
        max_times=30,
        ramp_kwargs=None,
    ):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None, **kwargs):
        coef = (
            self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        )
        cur_bbox_max_size = int(
            self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef
        )
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(
            img.shape[1:],
            margin=self.margin,
            bbox_min_size=self.bbox_min_size,
            bbox_max_size=cur_bbox_max_size,
            min_times=self.min_times,
            max_times=cur_max_times,
        )


def make_quadrant_mask(shape, mask_width, mask_height):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    start_mask_x = 0 if random.random() < 0.5 else (width - mask_width)
    start_mask_y = 0 if random.random() < 0.5 else (height - mask_height)
    mask[
        start_mask_y : start_mask_y + mask_height,
        start_mask_x : start_mask_x + mask_width,
    ] = 1
    return mask[None, ...]


class QuadrantsMaskGenerator:
    def __init__(
        self,
        min_width_ratio=0.1,
        min_height_ratio=0.1,
        max_width_ratio=0.9,
        max_height_ratio=0.9,
    ):
        self.min_width_ratio = min_width_ratio
        self.min_height_ratio = min_height_ratio
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio

    def __call__(self, img, **kwargs) -> Any:
        shape = img.shape[1:]
        mask_width = random.randint(
            int(shape[1] * self.min_width_ratio), int(shape[1] * self.max_width_ratio)
        )
        mask_height = random.randint(
            int(shape[0] * self.min_height_ratio), int(shape[0] * self.max_height_ratio)
        )

        return make_quadrant_mask(shape, mask_width, mask_height)


class OutpaintingMaskGenerator:
    def __init__(
        self,
        min_padding_percent: float = 0.05,
        max_padding_percent: int = 0.4,
        left_padding_prob: float = 0.6,
        top_padding_prob: float = 0.6,
        right_padding_prob: float = 0.6,
        bottom_padding_prob: float = 0.6,
        is_fixed_randomness: bool = False,
    ):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [
            left_padding_prob,
            top_padding_prob,
            right_padding_prob,
            bottom_padding_prob,
        ]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert (
            len(
                [
                    x
                    for x in [self.min_padding_percent, self.max_padding_percent]
                    if (x >= 0 and x <= 1)
                ]
            )
            == 2
        ), f"Padding percentage should be in [0,1]"
        assert (
            sum(self.probs) > 0
        ), f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert (
            len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4
        ), f"At least one of padding probs is not in [0,1] - {self.probs}"

    def apply_padding(self, mask, coord):
        mask[
            int(coord[0][0] * self.img_h) : int(coord[1][0] * self.img_h),
            int(coord[0][1] * self.img_w) : int(coord[1][1] * self.img_w),
        ] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent * size)
        n2 = int(self.max_padding_percent * size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash) % (2**32)
        return res

    def __call__(self, img, iter_i=None, raw_image=None, **kwargs):
        c, self.img_h, self.img_w = img.shape
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, f"Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [
            [(0, 0), (1, self.get_padding(size=self.img_h))],
            [(0, 0), (self.get_padding(size=self.img_w), 1)],
            [(0, 1 - self.get_padding(size=self.img_h)), (1, 1)],
            [(1 - self.get_padding(size=self.img_w), 0), (1, 1)],
        ]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(
                range(len(coords)), p=np.array(self.probs) / sum(self.probs)
            )
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask[None, ...]


class ObjectSegmentationMaskGenerator:
    def __init__(
        self,
        min_mask_area=0.000,
        max_mask_area=0.9,
        seed=42,
        convex_shape=False,
        dilate=True,
        n_choices=25,
    ):
        self.rng = random.Random(seed)
        self._classes4masking = {
            3,
            4,
            5,
            6,
            7,
            10,
            11,
            12,
            14,
            15,
            17,
            19,
            24,
            25,
            29,
            30,
            32,
            33,
            34,
            36,
        }  # 40
        self._min_mask_area, self._max_mask_area = min_mask_area, max_mask_area
        self.convex_shape = convex_shape
        self.dilate = dilate
        self.n_choices = n_choices

    def __call__(self, empty_semantic_map, full_semantic_map, **kwargs):
        objects_empty, objects_full = self._extract_scene_objects(
            empty_semantic_map, full_semantic_map
        )
        candidate_objects_for_removal = self._select_candidates(
            objects_empty, objects_full
        )

        if len(candidate_objects_for_removal) == 0:
            return self._center_mask(full_semantic_map)[None, ...]
        else:
            mask = self._compute_mask(
                full_semantic_map, list(candidate_objects_for_removal)
            )
            if mask is None:
                return self._center_mask(full_semantic_map)[None, ...]
        return mask[None, ...]

    def _select_candidates(self, objects_in_empty, objects_in_full):
        return (objects_in_full - objects_in_empty).intersection(self._classes4masking)

    def _extract_scene_objects(self, empty_semantic_map, full_semantic_map):
        objects_in_empty = np.unique(empty_semantic_map.flatten())
        objects_in_full = np.unique(full_semantic_map.flatten())
        return set(objects_in_empty), set(objects_in_full)

    def _center_mask(self, semantic_map):
        return make_random_rectangle_mask(
            semantic_map.shape[1:],
            margin=20,
            bbox_min_size=10,
            bbox_max_size=70,
            min_times=10,
            max_times=50,
        ).squeeze(0)


    def _compute_mask(self, semantic_map, candidate_objects_for_removal):
        image_area = semantic_map.size
        while len(candidate_objects_for_removal):
            if self.convex_shape:
                chosen_id = self.rng.choice(candidate_objects_for_removal)
                object_mask = (semantic_map == chosen_id).astype(np.uint8).squeeze(0)
                contours, _ = cv2.findContours(
                    object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
                # Find the convex hull object for each contour

                boundary_objects = []
                for i in range(len(contours)):
                    for p in contours[i]:
                        if (0 in p) or ((semantic_map.shape[1] - 1) in p):
                            boundary_objects.append(i)

                if len(boundary_objects) > 1:
                    h = np.zeros_like(object_mask)
                    for i in range(len(boundary_objects)):
                        hull = cv2.convexHull(contours[boundary_objects[i]])
                        h = cv2.fillConvexPoly(h, hull, (1))
                    area = np.sum(h)
                    if (area > self._min_mask_area * image_area) and (
                        area < self._max_mask_area * image_area
                    ):
                        return h
                    else:
                        candidate_objects_for_removal.remove(chosen_id)
                        continue
                elif len(contours) >= 1:
                    max_area, max_id = -1, -1
                    for i in range(len(contours)):
                        hull = cv2.convexHull(contours[i])
                        h = np.zeros_like(object_mask)
                        h = cv2.fillConvexPoly(h, hull, (1))
                        area = np.sum(h)
                        if (
                            (area > max_area)
                            and (area > self._min_mask_area * image_area)
                            and (area < self._max_mask_area * image_area)
                        ):
                            max_area, max_id = area, i
                            mask = h
                    if max_id != -1:
                        return mask
                    else:  # suitable object not found
                        candidate_objects_for_removal.remove(chosen_id)
                        continue
            else:
                chosen_ids = []
                masks = []
                ratios = []
                for i in range(self.n_choices):
                    if len(candidate_objects_for_removal):
                        chosen_id = self.rng.choice(candidate_objects_for_removal)
                        candidate_objects_for_removal.remove(chosen_id)
                        chosen_ids.append(chosen_id)
                    else:
                        break
                for i in chosen_ids:
                    h = np.isin(semantic_map, i).astype(np.uint8)
                    m = h.squeeze(0)
                    masks.append(m)
                    ratios.append(np.sum(m))

                # pick initial mask
                def argmax(lst):
                    return lst.index(max(lst))

                kernel = np.ones((3, 3), np.uint8)
                while len(chosen_ids):
                    index_max = argmax(ratios)
                    mask = masks[index_max]
                    masks.pop(index_max)
                    ratios.pop(index_max)
                    chosen_ids.pop(index_max)
                    if self.dilate:
                        mask = cv2.dilate(mask, kernel, iterations=2, borderValue=0)
                    area = np.sum(mask)
                    if area <= self._max_mask_area * image_area:
                        break
                # while other clutter available, add if < max ratio
                while len(chosen_ids):
                    index_max = argmax(ratios)
                    mask2 = masks[index_max]
                    masks.pop(index_max)
                    ratios.pop(index_max)
                    chosen_ids.pop(index_max)
                    mask2 = np.logical_or(mask2, mask).astype(np.uint8)
                    if self.dilate:
                        mask2 = cv2.dilate(mask2, kernel, iterations=2, borderValue=0)
                    area2 = np.sum(mask2)
                    if area2 <= self._max_mask_area * image_area:
                        mask = mask2

                #if in ratio interval return mask
                area = np.sum(mask)
                if (area > self._min_mask_area * image_area) and (
                    area <= self._max_mask_area * image_area
                ):
                    return mask
                # else try dilate until in ratio
                while area <= self._max_mask_area * image_area:
                    mask = cv2.dilate(mask, kernel, iterations=1, borderValue=0)
                    area = np.sum(mask)
                    if (area > self._min_mask_area * image_area) and (
                        area <= self._max_mask_area * image_area
                    ):
                        return mask
            return self._center_mask(semantic_map)
        return self._center_mask(semantic_map)


class EveryOtherLineMaskGenerator:
    def __call__(self, img, **kwargs):
        height, width = img.shape[1:]
        mask = np.zeros((height, width), np.float32)
        mask[::2, :] = 1
        return mask[None, ...]


class EveryOtherColumnMaskGenerator:
    def __call__(self, img, **kwargs):
        height, width = img.shape[1:]
        mask = np.zeros((height, width), np.float32)
        mask[:, ::2] = 1
        return mask[None, ...]


class SR2XMaskGenerator:
    def __call__(self, img, **kwargs):
        height, width = img.shape[1:]

        lines = np.zeros((height, width), dtype=np.float32)
        lines[::2, :] = 1

        columns = np.zeros((height, width), dtype=np.float32)
        columns[:, ::2] = 1

        mask = np.logical_or(lines, columns).astype(np.float32)

        return mask[None, ...]


class MixedMaskGenerator:
    def __init__(
        self,
        irregular_mask_prob=0.3,
        rectangular_mask_prob=0.3,
        semantic_mask_prob=0.4,
        outpainting_mask_prob=0.3,
        quadrant_mask_prob=0.3,
        lines_mask_prob=0.1,
        columns_mask_prob=0.1,
        sr2x_mask_prob=0.1,
        irregular_kwargs=None,
        rectangular_kwargs=None,
        semantic_kwargs=None,
        outpainting_kwargs=None,
        quadrant_kwargs=None,
    ):
        self.probas = []
        self.gens = []

        if irregular_mask_prob > 0:
            self.probas.append(irregular_mask_prob)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs["draw_method"] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if rectangular_mask_prob > 0:
            self.probas.append(rectangular_mask_prob)
            if rectangular_kwargs is None:
                rectangular_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**rectangular_kwargs))

        if semantic_mask_prob > 0:
            self.probas.append(semantic_mask_prob)
            if semantic_kwargs is None:
                semantic_kwargs = {}
            self.gens.append(ObjectSegmentationMaskGenerator(**semantic_kwargs))

        if outpainting_mask_prob > 0:
            self.probas.append(outpainting_mask_prob)
            if outpainting_kwargs is None:
                outpainting_kwargs = {}
            self.gens.append(OutpaintingMaskGenerator(**outpainting_kwargs))

        if quadrant_mask_prob > 0:
            self.probas.append(quadrant_mask_prob)
            if quadrant_kwargs is None:
                quadrant_kwargs = {}
            self.gens.append(QuadrantsMaskGenerator(**quadrant_kwargs))

        if lines_mask_prob > 0:
            self.probas.append(lines_mask_prob)
            self.gens.append(EveryOtherLineMaskGenerator())

        if columns_mask_prob > 0:
            self.probas.append(columns_mask_prob)
            self.gens.append(EveryOtherColumnMaskGenerator())

        if sr2x_mask_prob > 0:
            self.probas.append(sr2x_mask_prob)
            self.gens.append(SR2XMaskGenerator())

        self.probas = np.array(self.probas, dtype="float32")
        self.probas /= self.probas.sum()

    def __call__(
        self, img, empty_semantic_map, full_semantic_map, iter_i=None, raw_image=None
    ) -> Any:
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(
            img=img,
            empty_semantic_map=empty_semantic_map,
            full_semantic_map=full_semantic_map,
            iter_i=iter_i,
            raw_image=raw_image,
        )
        return result


def GeneratorFactory_get(generator_type, mask_generator_kwargs={}):
    """
    Generator types:
    - 1 : Mixed
    - 2 : SemanticOnly
    - 3 : RandomIrregularOnly
    - 4 : RandomRectangularOnly
    - 5 : OutpaintingOnly
    - 6 : QuadrantOnly
    - 7 : EveryOtherLine
    - 8 : EveryOtherColumn
    - 9 : SR2X
    """
    if generator_type == 1:
        return MixedMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 2:
        return ObjectSegmentationMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 3:
        return RandomIrregularMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 4:
        return RandomRectangleMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 5:
        return OutpaintingMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 6:
        return QuadrantsMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 7:
        return EveryOtherLineMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 8:
        return EveryOtherColumnMaskGenerator(**mask_generator_kwargs)
    elif generator_type == 9:
        return SR2XMaskGenerator(**mask_generator_kwargs)
