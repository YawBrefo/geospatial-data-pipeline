[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
# geospatial-data-pipeline
Satellite imagery acquisition and processing pipeline for large-scale remote sensing — powering the Alberta Wells Dataset (ICLR 2025) and agricultural monitoring workflows using Planet, Sentinel, and drone data.

[![Paper](https://img.shields.io/badge/arXiv-2410.09032-b31b1b.svg)](https://arxiv.org/abs/2410.09032)
[![ICLR 2025](https://img.shields.io/badge/ICLR%202025-Climate%20Change%20AI-brightgreen)](https://www.climatechange.ai/papers/iclr2025/32)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-87CEEB)](https://zenodo.org/records/13743323)

This repository contains the geospatial data acquisition and processing
pipeline developed for the
[Alberta Wells Dataset](https://arxiv.org/abs/2410.09032) project and extended
to general-purpose remote sensing workflows.

The planet data pipeline in this repository was used to acquire and process
the satellite imagery for over **213,000 oil and gas wells** across Alberta,
Canada — the data backbone of the benchmark published at the
**ICLR 2025 Workshop on Tackling Climate Change with Machine Learning**.

> **Seth, P.\*, Lin, M.\*, Brefo Dwamena, Y., Boutot, J., Kang, M., & Rolnick, D.** (2025).
> *Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery.*
> ICLR 2025 Workshop on Tackling Climate Change with Machine Learning.
> [arXiv:2410.09032](https://arxiv.org/abs/2410.09032) &nbsp;|&nbsp;
> [Climate Change AI](https://www.climatechange.ai/papers/iclr2025/32) &nbsp;|&nbsp;
> [Official Repo](https://github.com/RolnickLab/Alberta_Wells_Dataset)

## Project Overview

The repository is a portfolio-ready showcase for geospatial data engineering
and remote sensing automation. It includes scripts, and utilities demonstrating:

- **Satellite data acquisition at scale** — The planet data pipeline built here
  powered the batch acquisition of 213,000+ geolocated well sites from Planet
  Labs multi-spectral imagery (PSScene `ortho_analytic_4b_sr`), forming the
  data foundation of the Alberta Wells benchmark dataset.
- **AOI-based imagery search** using GeoJSON polygons derived from Alberta
  Energy Regulator well coordinates.
- **Object segmentation** pixel-based binary classification
  monitoring projects (palm plantations, banana estates).
- **Automation of geospatial preprocessing and export** with structured,
  reproducible pipelines.

## Research Context: Alberta Wells Dataset

Millions of abandoned oil and gas wells worldwide leak methane into the
atmosphere and toxic compounds into groundwater. Many of these locations are
undocumented, preventing remediation. The Alberta Wells Dataset addresses this
gap by pairing high-resolution Planet Labs satellite imagery with verified well
locations from the Alberta Energy Regulator to create the first large-scale
benchmark for well detection via remote sensing.

**This repository contains the data acquisition layer** of that pipeline:

```
Alberta Energy Regulator       Planet Labs API
  (well coordinates)           (satellite imagery)
        │                             │
        ▼                             ▼
   GeoJSON AOIs  ──────────►  planet_data_pipeline.py
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                   PSScene        Asset           Clipped
                   Search       Activation       Ordering
                        │             │             │
                        ▼             ▼             ▼
                  data_item.csv  status_item.csv  GeoTIFF
                                                 Downloads
                                                    │
                                                    ▼
                                          Alberta Wells Dataset
                                          (213K+ well images)
                                                    │
                                                    ▼
                                          Detection & Segmentation
                                          (see RolnickLab/Alberta_Wells_Dataset)
```

The downloaded imagery was then used for well detection and segmentation
experiments using computer vision models. For the model training, evaluation,
and benchmark code, see the
[official Alberta Wells Dataset repository](https://github.com/RolnickLab/Alberta_Wells_Dataset).

## Key Script

`planet_data_pipeline.py`

A consolidated Planet API pipeline that demonstrates a clean, production-ready
workflow for:

1. Scanning a directory of GeoJSON AOI files (e.g., well site polygons)
2. Searching Planet for matching `PSScene` assets
3. Activating `ortho_analytic_4b_sr` assets
4. Creating orders for clipped scenes
5. Downloading the resulting assets locally

This pipeline was designed to handle the scale of the Alberta Wells project
(213K+ sites) with structured batching and status tracking.

## Getting Started

### Requirements

- Python 3.9+
- `pandas`
- `requests`
- `planet` Python SDK
- `shapely`

Install dependencies with pip:

```bash
python3 -m pip install pandas requests planet shapely
```

### Environment

Set your Planet API key in the environment before running the pipeline:

```bash
export PLANET_API_KEY="your_planet_api_key"
```

### Run the Pipeline

```bash
python3 planet_data_pipeline.py \
  --geojson-dir /path/to/geojsons \
  --output-dir ./output
```

Optional flags:

- `--skip-activation`: perform search only, skip asset activation
- `--skip-download`: generate CSV output only, skip order download
- `--start-date` / `--end-date`: adjust the Planet acquisition date range
- `--max-cloud-cover`: set the cloud cover filter threshold

## Output

The pipeline saves:

- `data_item.csv`: Planet item IDs found for each GeoJSON AOI
- `status_item.csv`: asset activation status per item

Downloaded scene assets are stored in the working directory created by the
Planet Orders client.

## Repository Structure

This repository also includes several notebooks and scripts for specific
geospatial workflows:

| Script / Notebook | Domain |
|---|---|
| `Planet_API.py`, `Planet_data_pipeline.py`, `satellite_image_masking_pipeline.py`, `vector_file_batch_processing.py`| Alberta Wells — satellite acquisition, ordering, raster reprojection, pixel masking, & vector clipping, buffering and conversion |


## Citation

If you use this pipeline or the Alberta Wells Dataset in your research, please
cite:

```bibtex
@misc{seth2024albertawellsdatasetpinpointing,
      title={Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery},
      author={Pratinav Seth and Michelle Lin and Brefo Dwamena Yaw and Jade Boutot and Mary Kang and David Rolnick},
      year={2024},
      eprint={2410.09032},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.09032},
}
```

## Related Links

- **Paper**: [arXiv:2410.09032](https://arxiv.org/abs/2410.09032)
- **Workshop**: [ICLR 2025 — Tackling Climate Change with ML](https://www.climatechange.ai/papers/iclr2025/32)
- **Benchmark Code**: [RolnickLab/Alberta_Wells_Dataset](https://github.com/RolnickLab/Alberta_Wells_Dataset)
- **Dataset**: [Zenodo (213K+ wells)](https://zenodo.org/records/13743323)
- **Project Page**: [pratinavseth.github.io/alberta_wells_dataset](https://pratinavseth.github.io/alberta_wells_dataset/)

## Notes

- This project is intended as a portfolio sample of geospatial data science
  and remote sensing automation.
- It uses GeoJSON AOIs and Planet imagery filters to produce reproducible
  geospatial monitoring outputs.
- The pipeline is written with structured functions and clear CLI usage for
  demonstration and extension.
