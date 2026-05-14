import json
import os
import pickle

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap
from ScanModels import (
    fov_location_records,
    load_fov_locations,
    load_sample_centers,
    sample_center_records,
)


def populate_sample_selector(ui, sample_centers):
    ui.sampleSelector.clear()
    if not sample_centers:
        ui.sampleSelector.addItem("No Samples Found")
        return
    for i in range(len(sample_centers)):
        ui.sampleSelector.addItem(f"Sample {i+1}")


def save_usb_training_data(folder_path, raw_img, pixel_polygons, sample_centers, fov_locations):
    if raw_img is None or pixel_polygons is None:
        return
    if isinstance(raw_img, list) and len(raw_img) == 0:
        return

    image_path = os.path.join(folder_path, 'usb_raw_image.png')
    cv2.imwrite(image_path, raw_img)

    def clean_points(points):
        return [[float(x), float(y)] for x, y in points]

    def clean_records(records):
        cleaned = []
        for item in records:
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            cleaned.append(
                {
                    key: (
                        int(value)
                        if isinstance(value, (np.integer,))
                        else float(value)
                        if isinstance(value, (np.floating,))
                        else value
                    )
                    for key, value in item.items()
                }
            )
        return cleaned

    roi_data = {
        'image_file': os.path.basename(image_path),
        'coordinate_system': {
            'roi_vertices': 'USB displayed image pixel coordinates',
            'image_vertical_axis': 'stage X',
            'image_horizontal_axis': 'stage Y',
            'image_right_direction': 'smaller stage Y',
            'stage_units': 'mm',
        },
        'pixel_polygons': [
            {
                'sample_id': idx + 1,
                'points': clean_points(poly),
            }
            for idx, poly in enumerate(pixel_polygons)
        ],
        'sample_centers': clean_records(sample_centers),
        'fov_locations': clean_records(fov_locations),
    }

    json_path = os.path.join(folder_path, 'usb_rois.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(roi_data, f, indent=2)


def save_session_data(
    folder_path,
    fov_locations,
    sample_centers,
    overlay_images,
    raw_img=None,
    pixel_polygons=None,
    render_overlay=None,
):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data_to_save = {
        'FOV_locations': fov_location_records(fov_locations),
        'sample_centers': sample_center_records(sample_centers),
    }
    with open(os.path.join(folder_path, 'scan_metadata.pkl'), 'wb') as f:
        pickle.dump(data_to_save, f)

    save_usb_training_data(folder_path, raw_img, pixel_polygons, sample_centers, fov_locations)

    overlay_sources = {
        sample_id: source
        for sample_id, source in overlay_images.items()
        if isinstance(source, dict)
    }
    with open(os.path.join(folder_path, 'overlay_sources.pkl'), 'wb') as f:
        pickle.dump(overlay_sources, f)

    image_dir = os.path.join(folder_path, 'overlays')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for sample_id, source in overlay_images.items():
        file_path = os.path.join(image_dir, f'sample_{sample_id}_overlay.png')
        if isinstance(source, QPixmap):
            source.save(file_path, "PNG")
        elif render_overlay is not None:
            pixmap = render_overlay(sample_id)
            if pixmap is not None:
                pixmap.save(file_path, "PNG")


def load_session_data(folder_path):
    fov_locations = []
    sample_centers = []
    overlay_images = {}

    metadata_path = os.path.join(folder_path, 'scan_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            fov_locations = load_fov_locations(data.get('FOV_locations', []))
            sample_centers = load_sample_centers(data.get('sample_centers', []))

    overlay_sources_path = os.path.join(folder_path, 'overlay_sources.pkl')
    if os.path.exists(overlay_sources_path):
        with open(overlay_sources_path, 'rb') as f:
            overlay_images = pickle.load(f)
        return fov_locations, sample_centers, overlay_images

    image_dir = os.path.join(folder_path, 'overlays')
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.startswith('sample_') and filename.endswith('.png'):
                try:
                    sample_id = int(filename.split('_')[1])
                    pixmap = QPixmap()
                    pixmap.load(os.path.join(image_dir, filename))
                    overlay_images[sample_id] = pixmap
                except ValueError:
                    continue
    return fov_locations, sample_centers, overlay_images
