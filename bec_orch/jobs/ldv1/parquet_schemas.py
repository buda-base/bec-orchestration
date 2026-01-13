import pyarrow as pa

def ld_build_schema():
    """Build a PyArrow schema for Parquet output of lines detection.

    - tps_points: list<list<float32>>
    - lines_contours: list<list<struct<x:int16,y:int16>>>
    Returns None if PyArrow is unavailable so the skeleton can run.

    """
    if pa is None:
        return None
    pt = pa.list_(pa.list_(pa.float32()))
    point_struct = pa.struct([("x", pa.int16()), ("y", pa.int16())])
    bbox_struct = pa.struct([("x", pa.int16()), ("y", pa.int16()), ("w", pa.int16()), ("h", pa.int16())])
    contour = pa.list_(point_struct)
    contours = pa.list_(contour)
    bboxes = pa.list_(bbox_struct)
    schema = pa.schema([
        ("img_file_name", pa.string()),
        ("source_etag", pa.string()),
        ("rotation_angle", pa.float32()),
        ("tps_points", pt),
        ("tps_alpha", pa.float16()),
        ("contours", contours),
        ("nb_contours", pa.int32()),
        ("contours_bboxes", bboxes),
         # Hybrid error summary (always present / nullable as appropriate)
        ("ok", pa.bool_()),
        ("error_stage", pa.string()),     # null when ok=True
        ("error_type", pa.string()),      # null when ok=True
        ("error_message", pa.string())    # null when ok=True (writer may truncate)
    ])
    return schema
