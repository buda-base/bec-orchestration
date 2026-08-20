"""Unit tests for paddleocr header/footer background fill.

Run (no GPU)::

    python -m unittest bec_orch.jobs.paddleocr.test_layout_mask
"""

from __future__ import annotations

import unittest

from PIL import Image

from bec_orch.jobs.paddleocr.config import PaddleOCRConfig
from bec_orch.jobs.paddleocr.layout_mask import (
    apply_header_footer_mask,
    estimate_background_rgb,
    removed_code,
    xywhn_to_xyxy,
)


def _box(label: str, x: float, y: float, w: float, h: float, cls: int | None = None) -> dict:
    b: dict = {"label": label, "x": x, "y": y, "w": w, "h": h, "conf": 0.9}
    if cls is not None:
        b["cls"] = cls
    return b


class TestXywhn(unittest.TestCase):
    def test_full_image(self):
        self.assertEqual(xywhn_to_xyxy(0.5, 0.5, 1.0, 1.0, 100, 200), (0, 0, 100, 200))

    def test_top_header(self):
        # centre y=0.05, h=0.10 → [0, 0.10] of 200px → [0, 20]
        xyxy = xywhn_to_xyxy(0.5, 0.05, 1.0, 0.10, 100, 200)
        self.assertIsNotNone(xyxy)
        x1, y1, x2, y2 = xyxy
        self.assertEqual((x1, y1), (0, 0))
        self.assertEqual(x2, 100)
        self.assertAlmostEqual(y2, 20, delta=1)

    def test_invalid(self):
        self.assertIsNone(xywhn_to_xyxy(0.5, 0.5, 0.0, 0.1, 100, 100))
        self.assertIsNone(xywhn_to_xyxy(0.5, 0.5, 0.1, 0.1, 0, 100))


class TestMask(unittest.TestCase):
    def setUp(self):
        # Cream page, dark "ink" in header / footer / footnote / body.
        self.bg = (240, 235, 220)
        img = Image.new("RGB", (200, 400), self.bg)
        px = img.load()
        # header band y=0..20: dark pixels
        for y in range(20):
            for x in range(40, 160):
                px[x, y] = (20, 20, 20)
        # body band y=80..300
        for y in range(80, 300):
            for x in range(20, 180):
                if (x + y) % 7 == 0:
                    px[x, y] = (10, 10, 10)
        # footnote band y=320..350
        for y in range(320, 350):
            for x in range(30, 170):
                px[x, y] = (15, 15, 15)
        # footer band y=380..400
        for y in range(380, 400):
            for x in range(80, 120):
                px[x, y] = (25, 25, 25)
        self.img = img
        self.boxes = [
            _box("header", 0.5, 0.025, 0.8, 0.05, cls=0),       # y 0..20
            _box("text-area", 0.5, 0.475, 0.85, 0.55, cls=1),   # y ~80..300
            _box("footnote", 0.5, 0.8375, 0.7, 0.075, cls=2),   # y 320..350
            _box("footer", 0.5, 0.975, 0.3, 0.05, cls=3),       # y 380..400
        ]

    def test_paints_header_and_footer_not_footnote(self):
        out, n = apply_header_footer_mask(self.img, self.boxes, pad_px=0)
        self.assertEqual(n, 2)
        self.assertNotEqual(list(self.img.getpixel((100, 10))), list(out.getpixel((100, 10))))
        self.assertNotEqual(list(self.img.getpixel((100, 390))), list(out.getpixel((100, 390))))
        # footnote ink must survive
        self.assertEqual(self.img.getpixel((100, 330)), out.getpixel((100, 330)))
        # body ink must survive
        self.assertEqual(self.img.getpixel((20, 80)), out.getpixel((20, 80)))
        # painted pixels should be close to the cream background
        hdr = out.getpixel((100, 10))
        for c_out, c_bg in zip(hdr, self.bg):
            self.assertLess(abs(c_out - c_bg), 25)

    def test_protects_overlapping_text_area(self):
        # Header box that incorrectly covers the body.
        boxes = [
            _box("header", 0.5, 0.4, 0.9, 0.8, cls=0),
            _box("text-area", 0.5, 0.475, 0.85, 0.55, cls=1),
        ]
        out, n = apply_header_footer_mask(self.img, boxes, pad_px=0)
        self.assertEqual(n, 1)
        # a body pixel inside the text-area must be restored
        self.assertEqual(self.img.getpixel((21, 81)), out.getpixel((21, 81)))

    def test_empty_is_noop(self):
        out, n = apply_header_footer_mask(self.img, [])
        self.assertEqual(n, 0)
        self.assertIs(out, self.img)

    def test_cls_id_fallback(self):
        boxes = [
            {"cls": 0, "x": 0.5, "y": 0.025, "w": 0.8, "h": 0.05},
            {"cls": 3, "x": 0.5, "y": 0.975, "w": 0.3, "h": 0.05},
            {"cls": 2, "x": 0.5, "y": 0.8375, "w": 0.7, "h": 0.075},
        ]
        out, n = apply_header_footer_mask(self.img, boxes, pad_px=0)
        self.assertEqual(n, 2)
        self.assertEqual(self.img.getpixel((100, 330)), out.getpixel((100, 330)))

    def test_background_is_light_on_cream_page(self):
        r, g, b = estimate_background_rgb(self.img)
        self.assertGreater(r, 180)
        self.assertGreater(g, 180)
        self.assertGreater(b, 160)


class TestRemovedCode(unittest.TestCase):
    def test_none_h_f_hf(self):
        self.assertEqual(removed_code([]), "none")
        self.assertEqual(removed_code([_box("text-area", 0.5, 0.5, 0.8, 0.8)]), "none")
        self.assertEqual(removed_code([_box("header", 0.5, 0.05, 0.8, 0.1)]), "h")
        self.assertEqual(removed_code([_box("footer", 0.5, 0.95, 0.2, 0.05)]), "f")
        self.assertEqual(
            removed_code(
                [_box("header", 0.5, 0.05, 0.8, 0.1), _box("footer", 0.5, 0.95, 0.2, 0.05)]
            ),
            "hf",
        )
        # footnotes do not affect removed
        self.assertEqual(
            removed_code(
                [_box("header", 0.5, 0.05, 0.8, 0.1), _box("footnote", 0.5, 0.9, 0.7, 0.08)]
            ),
            "h",
        )


class TestConfig(unittest.TestCase):
    def test_v1_defaults_leave_mask_off(self):
        cfg = PaddleOCRConfig()
        self.assertFalse(cfg.layout_mask_enabled)
        self.assertEqual(cfg.layout_mask_labels, ("header", "footer"))
        self.assertIn("footnote", cfg.layout_mask_protect_labels)
        self.assertFalse(cfg.layout_isolate_footnotes)


if __name__ == "__main__":
    unittest.main()
