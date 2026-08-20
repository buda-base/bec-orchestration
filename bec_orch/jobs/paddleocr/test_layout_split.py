"""Unit tests for two-column split from layout_detection_v1 text-areas.

Run (no GPU)::

    python -m unittest bec_orch.jobs.paddleocr.test_layout_split
"""

from __future__ import annotations

import unittest

from PIL import Image

from bec_orch.jobs.paddleocr.config import PaddleOCRConfig
from bec_orch.jobs.paddleocr.layout_split import (
    crop_body_regions,
    crop_labeled_regions,
    crop_with_background_margin,
    drop_fully_contained,
    fully_inside,
    horiz_overlap_frac,
    join_column_texts,
    reading_order_xycut,
    region_margin_px,
    select_column_rects,
    split_page_columns,
    text_area_rects,
    vert_overlap_frac,
)
from bec_orch.jobs.paddleocr.worker import _OCRResult, PaddleOCRJobWorker


def _xyxy(x1, y1, x2, y2):
    return (x1, y1, x2, y2)


def _box(label, x, y, w, h):
    return {"label": label, "x": x, "y": y, "w": w, "h": h, "conf": 0.9}


class TestOverlapHeuristic(unittest.TestCase):
    def test_side_by_side_columns_match(self):
        # Two 40%-wide columns, 10% gutter, same vertical span.
        left = _xyxy(50, 50, 250, 450)
        right = _xyxy(300, 50, 500, 450)
        self.assertGreaterEqual(vert_overlap_frac(left, right), 0.60)
        self.assertLess(horiz_overlap_frac(left, right), 0.05)
        cols = select_column_rects([left, right])
        self.assertEqual(cols, [left, right])

    def test_stacked_areas_do_not_match(self):
        top = _xyxy(50, 50, 450, 200)
        bot = _xyxy(50, 250, 450, 400)
        self.assertLess(vert_overlap_frac(top, bot), 0.60)
        self.assertIsNone(select_column_rects([top, bot]))

    def test_heavily_overlapping_horizontally_do_not_match(self):
        a = _xyxy(50, 50, 400, 450)
        b = _xyxy(200, 50, 550, 450)
        self.assertGreaterEqual(horiz_overlap_frac(a, b), 0.05)
        self.assertIsNone(select_column_rects([a, b]))

    def test_full_width_plus_two_columns_keeps_the_pair(self):
        left = _xyxy(40, 40, 240, 460)
        right = _xyxy(280, 40, 480, 460)
        full = _xyxy(30, 40, 490, 460)  # overlaps both horizontally
        cols = select_column_rects([left, right, full])
        self.assertEqual(cols, [left, right])

    def test_left_to_right_order(self):
        left = _xyxy(50, 50, 200, 400)
        right = _xyxy(250, 50, 400, 400)
        cols = select_column_rects([right, left])
        self.assertEqual(cols, [left, right])


def _from_xywhn(cid_x_y_w_h, pw, ph):
    x, y, w, h = cid_x_y_w_h
    return (
        int(round((x - w / 2) * pw)),
        int(round((y - h / 2) * ph)),
        int(round((x + w / 2) * pw)),
        int(round((y + h / 2) * ph)),
    )


class TestReadingOrderXYCut(unittest.TestCase):
    def test_single_and_pair(self):
        r = _xyxy(10, 10, 20, 20)
        self.assertEqual(reading_order_xycut([r], 100, 100), [r])
        left = _xyxy(10, 10, 40, 90)
        right = _xyxy(60, 10, 90, 90)
        self.assertEqual(reading_order_xycut([right, left], 100, 100), [left, right])

    def test_two_columns_read_top_to_bottom(self):
        # A clean 2x3 grid with a full-height gutter -> column-major order.
        l1 = _xyxy(10, 10, 40, 30)
        l2 = _xyxy(10, 40, 40, 60)
        l3 = _xyxy(10, 70, 40, 90)
        r1 = _xyxy(60, 10, 90, 30)
        r2 = _xyxy(60, 40, 90, 60)
        r3 = _xyxy(60, 70, 90, 90)
        order = reading_order_xycut([r2, l3, r1, l1, r3, l2], 100, 100)
        self.assertEqual(order, [l1, l2, l3, r1, r2, r3])

    def test_b3_title_between_two_bands(self):
        pw, ph = 1000, 1400
        A = _from_xywhn((0.282720, 0.320990, 0.413730, 0.431080), pw, ph)
        B = _from_xywhn((0.706110, 0.309860, 0.403340, 0.423670), pw, ph)
        C = _from_xywhn((0.494800, 0.600600, 0.271370, 0.043400), pw, ph)
        D = _from_xywhn((0.288670, 0.784940, 0.407780, 0.297600), pw, ph)
        E = _from_xywhn((0.708310, 0.785440, 0.401830, 0.302910), pw, ph)
        order = reading_order_xycut([E, C, A, D, B], pw, ph)
        self.assertEqual(order, [A, B, C, D, E])

    def test_v10_two_headings_then_two_columns(self):
        pw, ph = 760, 1050
        h1 = _from_xywhn((0.485880, 0.217170, 0.236350, 0.039190), pw, ph)
        h2 = _from_xywhn((0.738570, 0.286560, 0.352300, 0.038090), pw, ph)
        l1 = _from_xywhn((0.245780, 0.410480, 0.392450, 0.146170), pw, ph)
        l2 = _from_xywhn((0.246510, 0.583660, 0.388010, 0.145070), pw, ph)
        l3 = _from_xywhn((0.238360, 0.762650, 0.383520, 0.155690), pw, ph)
        r1 = _from_xywhn((0.687300, 0.413640, 0.386500, 0.137700), pw, ph)
        r2 = _from_xywhn((0.679150, 0.585760, 0.379080, 0.140860), pw, ph)
        r3 = _from_xywhn((0.694760, 0.760040, 0.404350, 0.139800), pw, ph)
        order = reading_order_xycut([r1, l2, h2, r3, l1, h1, r2, l3], pw, ph)
        self.assertEqual(order, [h1, h2, l1, l2, l3, r1, r2, r3])


class TestCropAndJoin(unittest.TestCase):
    def test_edge_crop_pads_background(self):
        img = Image.new("RGB", (100, 80), (10, 10, 10))
        bg = (200, 190, 180)
        out = crop_with_background_margin(
            img, (0, 0, 40, 40), margin_px=10, background=bg
        )
        self.assertEqual(out.size, (60, 60))
        # top-left of the canvas is outside the page → background
        self.assertEqual(out.getpixel((0, 0)), bg)
        # a pixel that came from the page
        self.assertEqual(out.getpixel((15, 15)), (10, 10, 10))

    def test_does_not_copy_sibling_column(self):
        img = Image.new("RGB", (200, 100), (240, 240, 240))
        px = img.load()
        bg = (1, 2, 3)
        for y in range(100):
            for x in range(0, 90):
                px[x, y] = (255, 0, 0)  # left column ink
            for x in range(110, 200):
                px[x, y] = (0, 0, 255)  # right column ink
        left = crop_with_background_margin(
            img, (0, 0, 90, 100), margin_px=20, background=bg
        )
        # Right-hand margin is synthetic, never the sibling column.
        w, h = left.size
        self.assertEqual(left.getpixel((w - 1, h // 2)), bg)
        self.assertEqual(left.getpixel((w - 5, 10)), bg)

    def test_split_page_columns_end_to_end(self):
        img = Image.new("RGB", (200, 100), (240, 240, 240))
        boxes = [
            _box("text-area", 0.25, 0.5, 0.4, 0.9),
            _box("text-area", 0.75, 0.5, 0.4, 0.9),
            _box("header", 0.5, 0.05, 0.8, 0.08),
        ]
        crops = split_page_columns(img, boxes, margin_frac=0.0)
        self.assertIsNotNone(crops)
        self.assertEqual(len(crops), 2)
        self.assertLess(crops[0].width, img.width)
        self.assertLess(crops[1].width, img.width)

    def test_join_two_line_breaks(self):
        self.assertEqual(join_column_texts(["left\n", "right"]), "left\n\nright")
        self.assertEqual(join_column_texts(["", "only"]), "only")

    def test_crop_footnotes_top_to_bottom(self):
        img = Image.new("RGB", (200, 400), (240, 240, 240))
        px = img.load()
        for y in range(320, 345):
            for x in range(20, 180):
                px[x, y] = (10, 10, 10)
        for y in range(355, 380):
            for x in range(20, 180):
                px[x, y] = (20, 20, 20)
        boxes = [
            _box("text-area", 0.5, 0.4, 0.8, 0.7),
            _box("footnote", 0.5, 0.831, 0.8, 0.062),
            _box("footnote", 0.5, 0.919, 0.8, 0.062),
        ]
        crops = crop_labeled_regions(
            img, boxes, label="footnote", margin_frac=0.0, min_px=0, box_margin_frac=0.0
        )
        self.assertEqual(len(crops), 2)
        # first crop is the upper footnote
        self.assertLess(crops[0].height, img.height)

    def test_margin_is_fully_synthetic(self):
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        px = img.load()
        bg = (1, 2, 3)
        # ink sitting just above a tight detection box must NOT be copied
        for y in range(88, 100):
            for x in range(20, 180):
                px[x, y] = (9, 9, 9)
        out = crop_with_background_margin(
            img, (20, 100, 180, 150), margin_px=20, background=bg
        )
        self.assertEqual(out.size, (200, 90))  # 160+40 x 50+40
        # canvas_y1 = 80; page y=88 is in the margin → synthetic
        self.assertEqual(out.getpixel((50, 8)), bg)
        # inside the box is still page pixels
        self.assertEqual(out.getpixel((50, 20)), (240, 240, 240))

    def test_text_area_overlap_does_not_eat_footnote_box(self):
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        px = img.load()
        for y in range(100, 110):
            for x in range(20, 180):
                px[x, y] = (9, 9, 9)
        out = crop_with_background_margin(
            img, (20, 100, 180, 160), margin_px=20, background=(1, 2, 3)
        )
        # box top y=100 is canvas y=20 and must still be ink
        self.assertEqual(out.getpixel((50, 20)), (9, 9, 9))

    def test_other_box_in_margin_is_synthetic_background(self):
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        px = img.load()
        bg = (1, 2, 3)
        for y in range(50, 80):
            for x in range(20, 180):
                px[x, y] = (255, 0, 0)
        out = crop_with_background_margin(
            img, (20, 100, 180, 150), margin_px=30, background=bg
        )
        # neighbouring text-area in the margin is not copied
        self.assertEqual(out.getpixel((50 - (-10), 75 - 70)), bg)

    def test_margin_extends_past_page_edge(self):
        img = Image.new("RGB", (100, 100), (10, 10, 10))
        bg = (1, 2, 3)
        out = crop_with_background_margin(
            img, (10, 80, 90, 100), margin_px=30, background=bg
        )
        # canvas is (10-30, 80-30) .. (90+30, 100+30) → 140 x 80
        self.assertEqual(out.size, (140, 80))
        self.assertEqual(out.getpixel((out.width - 1, out.height - 1)), bg)
        self.assertEqual(out.getpixel((0, 0)), bg)
        # box pixel page (20, 90) → canvas (40, 40)
        self.assertEqual(out.getpixel((20 - (10 - 30), 90 - (80 - 30))), (10, 10, 10))

    def test_region_margin_uses_box_height(self):
        # page-relative 2% of 200 = 4; min_px=8; 50% of 40px box = 20 → 20
        m = region_margin_px(
            200, 200, (10, 100, 190, 140), margin_frac=0.02, min_px=8, box_margin_frac=0.5
        )
        self.assertEqual(m, 20)

    def test_crop_labeled_regions_default_margin_is_generous(self):
        img = Image.new("RGB", (200, 400), (240, 240, 240))
        boxes = [
            _box("text-area", 0.5, 0.4, 0.8, 0.7),
            _box("footnote", 0.5, 0.9, 0.8, 0.1),
        ]
        crops = crop_labeled_regions(img, boxes, label="footnote")
        self.assertEqual(len(crops), 1)
        # box is 0.1 * 400 = 40px tall; half of that is 20, min_px is 32,
        # 5% of 200 is 10 → margin 32, canvas height 40+64
        self.assertEqual(crops[0].height, 40 + 2 * 32)

    def test_nested_footnote_is_dropped(self):
        img = Image.new("RGB", (200, 400), (240, 240, 240))
        boxes = [
            _box("footnote", 0.5, 0.85, 0.8, 0.2),
            _box("footnote", 0.5, 0.85, 0.4, 0.08),  # fully inside the first
        ]
        crops = crop_labeled_regions(
            img, boxes, label="footnote", margin_frac=0.0, min_px=0, box_margin_frac=0.0
        )
        self.assertEqual(len(crops), 1)

    def test_nested_text_area_is_dropped_when_not_columns(self):
        inner = _xyxy(80, 80, 120, 120)
        outer = _xyxy(50, 50, 150, 150)
        self.assertTrue(fully_inside(inner, outer))
        self.assertEqual(drop_fully_contained([outer, inner]), [outer])

    def test_near_nested_footnote_with_small_overhang_is_dropped(self):
        # Small note overhangs the big one by a few px on the left but is
        # otherwise inside it (the I1KG1760009 case) -> drop the small one.
        big = _xyxy(74, 802, 944, 986)
        small = _xyxy(71, 807, 931, 898)
        self.assertFalse(fully_inside(small, big))
        self.assertEqual(drop_fully_contained([big, small]), [big])

    def test_two_columns_inside_envelope_are_kept(self):
        left = _xyxy(40, 40, 240, 460)
        right = _xyxy(280, 40, 480, 460)
        full = _xyxy(30, 40, 490, 460)
        cols = select_column_rects([left, right, full])
        self.assertEqual(cols, [left, right])

    def test_footer_inside_footnote_is_blanked(self):
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        px = img.load()
        bg = (1, 2, 3)
        for y in range(100, 160):
            for x in range(20, 180):
                px[x, y] = (9, 9, 9)
        for y in range(140, 155):
            for x in range(80, 120):
                px[x, y] = (255, 0, 0)  # footer ink inside the footnote
        boxes = [
            _box("footnote", 0.5, 0.65, 0.8, 0.3),
            _box("footer", 0.5, 0.7375, 0.2, 0.075),
        ]
        crops = crop_labeled_regions(
            img,
            boxes,
            label="footnote",
            margin_frac=0.0,
            min_px=0,
            box_margin_frac=0.0,
            background=bg,
        )
        self.assertEqual(len(crops), 1)
        # footer centre ~ page (100, 147.5); footnote canvas origin = box
        fn = text_area_rects(boxes, 200, 200, label="footnote")[0]
        canvas_x1, canvas_y1 = fn[0], fn[1]
        self.assertEqual(crops[0].getpixel((100 - canvas_x1, 147 - canvas_y1)), bg)

    def test_footnote_overlap_is_blanked_on_text_area_crop(self):
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        px = img.load()
        bg = (1, 2, 3)
        for y in range(20, 160):
            for x in range(20, 180):
                px[x, y] = (9, 9, 9)
        for y in range(130, 155):
            for x in range(20, 180):
                px[x, y] = (0, 0, 255)  # footnote ink overlapping the text-area
        boxes = [
            _box("text-area", 0.5, 0.45, 0.8, 0.7),
            _box("footnote", 0.5, 0.7125, 0.8, 0.125),
        ]
        crops = crop_body_regions(img, boxes, margin_frac=0.0, min_px=0, background=bg)
        self.assertEqual(len(crops), 1)
        ta = text_area_rects(boxes, 200, 200, label="text-area")[0]
        self.assertEqual(crops[0].getpixel((100 - ta[0], 140 - ta[1])), bg)
        # a pixel of the text-area above the footnote stays ink
        self.assertEqual(crops[0].getpixel((100 - ta[0], 50 - ta[1])), (9, 9, 9))


class TestMergePageResults(unittest.TestCase):
    def test_joins_two_crops(self):
        cfg = PaddleOCRConfig(layout_split_columns=True)
        # Don't load the GPU engine — only call the merge helper.
        worker = object.__new__(PaddleOCRJobWorker)
        worker.cfg = cfg
        a = _OCRResult(raw_text="LEFT COL", output_tokens=3, truncated=False, finish_reason="stop")
        b = _OCRResult(raw_text="RIGHT COL", output_tokens=4, truncated=False, finish_reason="stop")
        merged = worker._merge_page_results([(0, "body"), (0, "body")], [a, b], n_pages=1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].raw_text, "LEFT COL\n\nRIGHT COL")
        self.assertEqual(merged[0].output_tokens, 7)

    def test_footnotes_go_to_separate_field(self):
        cfg = PaddleOCRConfig(layout_isolate_footnotes=True)
        worker = object.__new__(PaddleOCRJobWorker)
        worker.cfg = cfg
        body = _OCRResult(raw_text="BODY", output_tokens=2, truncated=False, finish_reason="stop")
        fn1 = _OCRResult(raw_text="note a", output_tokens=2, truncated=False, finish_reason="stop")
        fn2 = _OCRResult(raw_text="note b", output_tokens=2, truncated=False, finish_reason="stop")
        merged = worker._merge_page_results(
            [(0, "body"), (0, "footnote"), (0, "footnote")],
            [body, fn1, fn2],
            n_pages=1,
        )
        self.assertEqual(merged[0].raw_text, "BODY")
        self.assertEqual(merged[0].footnote_text, "note a\n\nnote b")
        self.assertEqual(merged[0].n_footnotes, 2)
        self.assertNotIn("note a", merged[0].raw_text)

    def test_v1_defaults_leave_split_off(self):
        cfg = PaddleOCRConfig()
        self.assertFalse(cfg.layout_split_columns)
        self.assertFalse(cfg.layout_isolate_footnotes)
        self.assertAlmostEqual(cfg.layout_column_min_vert_overlap, 0.60)
        self.assertAlmostEqual(cfg.layout_column_max_horiz_overlap, 0.05)
        self.assertAlmostEqual(cfg.layout_footnote_margin_frac, 0.05)
        self.assertEqual(cfg.layout_footnote_margin_min_px, 32)
        self.assertAlmostEqual(cfg.layout_footnote_box_margin_frac, 0.5)


if __name__ == "__main__":
    unittest.main()
