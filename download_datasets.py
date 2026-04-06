#!/usr/bin/env python3
"""
Medical 3D Dataset Downloader

Downloads publicly available 3D medical imaging datasets for pretraining.
Organizes them into a unified directory structure.

Usage:
    # Download all auto-downloadable datasets to a directory
    python download_datasets.py --output_dir /path/to/data --all

    # Download specific datasets
    python download_datasets.py --output_dir /path/to/data --datasets totalseg amos kits brats

    # List all available datasets and their status
    python download_datasets.py --list

    # Show manual download instructions for datasets requiring registration
    python download_datasets.py --manual

Directory structure after download:
    output_dir/
        totalsegmentator/    CT,  ~1200 volumes, whole-body
        amos/                CT+MRI, ~360 volumes, abdomen
        kits/                CT,  ~300 volumes, kidney
        lits/                CT,  ~131 volumes, liver
        brats/               MRI, ~1251 volumes, brain tumor
        bcv/                 CT,  ~30 volumes, abdomen
        flare22/             CT,  ~50 labeled + 2000 unlabeled, abdomen
        abdomenatlas/        CT,  ~20K volumes, abdomen (large!)
        acdc/                MRI, ~100 volumes, cardiac
        ct_org/              CT,  ~140 volumes, multi-organ
        word/                CT,  ~150 volumes, abdomen
        structseg/           CT,  ~82 volumes, head-neck + thorax
        autopet/             PET/CT, ~900 volumes (manual)
        mms/                 MRI, ~150 volumes, cardiac (manual)
        pelvic/              CT, ~184 volumes, pelvic (manual)
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    # --- Fully auto-downloadable ---
    "totalseg": {
        "name": "TotalSegmentator v2",
        "modality": "CT",
        "n_volumes": "~1200",
        "region": "Whole-body (104 structures)",
        "auto": True,
        "size_gb": "~260 GB",
        "source": "Zenodo",
        "description": "Largest public CT dataset with whole-body annotations. "
                        "Uses the totalsegmentator Python package for download.",
    },
    "amos": {
        "name": "AMOS 2022",
        "modality": "CT + MRI",
        "n_volumes": "~360 (300 CT + 60 MRI)",
        "region": "Abdomen (15 organs)",
        "auto": True,
        "size_gb": "~30 GB",
        "source": "Zenodo / Grand Challenge",
    },
    "kits": {
        "name": "KiTS23 (Kidney Tumor Segmentation)",
        "modality": "CT",
        "n_volumes": "~489",
        "region": "Kidney + tumor",
        "auto": True,
        "size_gb": "~30 GB",
        "source": "GitHub",
    },
    "lits": {
        "name": "LiTS (Liver Tumor Segmentation)",
        "modality": "CT",
        "n_volumes": "~131",
        "region": "Liver + tumor",
        "auto": True,
        "size_gb": "~18 GB",
        "source": "Synapse / Academic Torrents",
    },
    "brats": {
        "name": "BraTS 2023 (Brain Tumor Segmentation)",
        "modality": "MRI (T1, T1CE, T2, FLAIR)",
        "n_volumes": "~1251",
        "region": "Brain tumor",
        "auto": True,
        "size_gb": "~15 GB",
        "source": "Synapse",
    },
    "flare22": {
        "name": "FLARE 2022",
        "modality": "CT",
        "n_volumes": "~2050 (50 labeled + 2000 unlabeled)",
        "region": "Abdomen (13 organs)",
        "auto": True,
        "size_gb": "~100 GB",
        "source": "Zenodo / Grand Challenge",
    },
    "acdc": {
        "name": "ACDC (Automated Cardiac Diagnosis)",
        "modality": "MRI",
        "n_volumes": "~100",
        "region": "Cardiac",
        "auto": True,
        "size_gb": "~2 GB",
        "source": "CREATIS",
    },
    "ct_org": {
        "name": "CT-ORG",
        "modality": "CT",
        "n_volumes": "~140",
        "region": "Multi-organ",
        "auto": True,
        "size_gb": "~15 GB",
        "source": "TCIA / wiki",
    },
    "word": {
        "name": "WORD (Whole abdominal ORgan Dataset)",
        "modality": "CT",
        "n_volumes": "~150",
        "region": "Abdomen (16 organs)",
        "auto": True,
        "size_gb": "~20 GB",
        "source": "GitHub",
    },
    "abdomenatlas": {
        "name": "AbdomenAtlas 1.1",
        "modality": "CT",
        "n_volumes": "~20,460",
        "region": "Abdomen (25 organs)",
        "auto": True,
        "size_gb": "~1.5 TB (very large!)",
        "source": "HuggingFace",
        "description": "WARNING: This is an extremely large dataset (~1.5 TB). "
                        "Consider downloading a subset first.",
    },

    # --- Requires registration / manual steps ---
    "bcv": {
        "name": "BCV / Synapse Multi-Atlas",
        "modality": "CT",
        "n_volumes": "~30",
        "region": "Abdomen (13 organs)",
        "auto": False,
        "size_gb": "~3 GB",
        "source": "Synapse (synapse.org)",
        "manual_url": "https://www.synapse.org/#!Synapse:syn3193805",
        "manual_instructions": (
            "1. Create account at https://www.synapse.org/\n"
            "2. Go to https://www.synapse.org/#!Synapse:syn3193805\n"
            "3. Accept the data use agreement\n"
            "4. Download 'Abdomen/RawData.zip'\n"
            "5. Extract to {output_dir}/bcv/"
        ),
    },
    "autopet": {
        "name": "AutoPET (Automated PET/CT Lesion Segmentation)",
        "modality": "PET/CT",
        "n_volumes": "~900",
        "region": "Whole-body",
        "auto": False,
        "size_gb": "~300 GB",
        "source": "TCIA",
        "manual_url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287",
        "manual_instructions": (
            "1. Go to https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287\n"
            "2. Register for TCIA account (free)\n"
            "3. Accept the data use agreement\n"
            "4. Use the NBIA Data Retriever to download\n"
            "5. Convert DICOM to NIfTI using dcm2niix\n"
            "6. Place in {output_dir}/autopet/"
        ),
    },
    "mms": {
        "name": "M&Ms (Multi-Centre Multi-Vendor Cardiac MRI)",
        "modality": "MRI",
        "n_volumes": "~150",
        "region": "Cardiac",
        "auto": False,
        "size_gb": "~5 GB",
        "source": "IACL",
        "manual_url": "https://www.ub.edu/mnms/",
        "manual_instructions": (
            "1. Go to https://www.ub.edu/mnms/\n"
            "2. Register and accept the data use agreement\n"
            "3. Download the dataset\n"
            "4. Place in {output_dir}/mms/"
        ),
    },
    "structseg": {
        "name": "StructSeg (Head-Neck + Thoracic)",
        "modality": "CT",
        "n_volumes": "~82 (42 H&N + 40 thorax)",
        "region": "Head-Neck (22 organs) + Thorax",
        "auto": False,
        "size_gb": "~8 GB",
        "source": "Grand Challenge",
        "manual_url": "https://structseg2019.grand-challenge.org/",
        "manual_instructions": (
            "1. Go to https://structseg2019.grand-challenge.org/\n"
            "2. Register and request data access\n"
            "3. Download Task 1 (H&N OAR) and Task 3 (Thoracic OAR)\n"
            "4. Place in {output_dir}/structseg/"
        ),
    },
    "pelvic": {
        "name": "Pelvic CT",
        "modality": "CT",
        "n_volumes": "~184",
        "region": "Pelvic bones",
        "auto": False,
        "size_gb": "~12 GB",
        "source": "Zenodo (partial) / various",
        "manual_url": "https://doi.org/10.1007/s11548-021-02346-6",
        "manual_instructions": (
            "Pelvic bone datasets are distributed across multiple sources:\n"
            "1. CTPelvic1K: https://zenodo.org/record/4588403\n"
            "2. Follow instructions in the associated paper\n"
            "3. Place in {output_dir}/pelvic/"
        ),
    },
}


# ============================================================================
# Helper functions
# ============================================================================

def run_cmd(cmd, desc="", check=True):
    """Run a shell command with logging."""
    print(f"  >> {desc or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  [WARN] Command failed: {result.stderr[:500]}")
        return False
    return True


def ensure_pip_package(package, pip_name=None):
    """Install a pip package if not already available."""
    try:
        __import__(package)
        return True
    except ImportError:
        pip_name = pip_name or package
        print(f"  Installing {pip_name}...")
        run_cmd(f"{sys.executable} -m pip install {pip_name} -q", check=False)
        try:
            __import__(package)
            return True
        except ImportError:
            print(f"  [WARN] Could not install {pip_name}")
            return False


def download_file(url, output_path, desc=""):
    """Download a file using wget or curl."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  [SKIP] Already exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    desc = desc or f"Downloading {output_path.name}"
    print(f"  {desc}...")

    # Try wget first, then curl
    if shutil.which("wget"):
        cmd = f'wget -q --show-progress -O "{output_path}" "{url}"'
    elif shutil.which("curl"):
        cmd = f'curl -L -o "{output_path}" "{url}"'
    else:
        print("  [ERROR] Neither wget nor curl found. Please install one.")
        return False

    return run_cmd(cmd, desc=desc, check=True)


def extract_archive(archive_path, extract_dir):
    """Extract zip or tar archive."""
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {archive_path.name}...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif archive_path.name.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(extract_dir)
    else:
        print(f"  [WARN] Unknown archive format: {archive_path}")
        return False
    return True


# ============================================================================
# Dataset-specific download functions
# ============================================================================

def download_totalseg(output_dir):
    """
    TotalSegmentator v2 — ~1200 CT volumes, whole-body.
    Uses the official totalsegmentator dataset download from Zenodo.
    """
    ds_dir = Path(output_dir) / "totalsegmentator"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print("\n  TotalSegmentator is a very large dataset (~260 GB).")
    print("  Option 1: Download via the official dataset release on Zenodo")
    print("  Option 2: Use the totalsegmentator Python package\n")

    # Download the dataset metadata / small subset first
    # The full dataset is at: https://zenodo.org/records/6802614
    script = ds_dir / "download_totalseg.sh"
    script.write_text(f"""#!/bin/bash
# TotalSegmentator Dataset Download Script
# Full dataset: https://zenodo.org/records/6802614
# Size: ~260 GB
#
# Option A: Using zenodo_get (recommended for full download)
#   pip install zenodo_get
#   cd {ds_dir}
#   zenodo_get 6802614
#
# Option B: Using the totalsegmentator package
#   pip install totalsegmentator
#   totalseg_download_dataset -o {ds_dir}
#
# Option C: Manual download
#   Go to https://zenodo.org/records/6802614
#   Download all .zip files
#   Extract to {ds_dir}/

echo "Choose a download method above and run it."
echo "The dataset is ~260 GB, so this will take a while."
""")
    script.chmod(0o755)

    # Try automated download with zenodo_get
    if ensure_pip_package("zenodo_get"):
        print("  Starting download via zenodo_get (this will take a long time)...")
        run_cmd(
            f"cd {ds_dir} && zenodo_get 6802614",
            desc="Downloading TotalSegmentator from Zenodo",
            check=False,
        )
    else:
        print(f"  [INFO] Download script saved to: {script}")
        print(f"  Run it manually when ready.")


def download_amos(output_dir):
    """
    AMOS 2022 — 300 CT + 60 MRI volumes, abdomen.
    Available on Zenodo: https://zenodo.org/records/7155725
    """
    ds_dir = Path(output_dir) / "amos"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # AMOS is split into two zip files on Zenodo
    urls = [
        ("https://zenodo.org/records/7155725/files/amos22.zip", "amos22.zip"),
    ]
    for url, fname in urls:
        fpath = ds_dir / fname
        if download_file(url, fpath, desc=f"Downloading AMOS ({fname})"):
            extract_archive(fpath, ds_dir)


def download_kits(output_dir):
    """
    KiTS23 — ~489 CT volumes, kidney/tumor segmentation.
    Uses the official kits23 GitHub repo with starter_code.
    """
    ds_dir = Path(output_dir) / "kits"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Clone the kits23 repository which contains the download script
    repo_dir = ds_dir / "kits23"
    if not repo_dir.exists():
        run_cmd(
            f"git clone https://github.com/neheller/kits23.git {repo_dir}",
            desc="Cloning KiTS23 repository",
        )

    # Install requirements and download
    print("  To download KiTS23 data, run:")
    print(f"    cd {repo_dir}")
    print(f"    pip install -e .")
    print(f"    kits23_download")
    print()

    # Try automated download
    run_cmd(
        f"cd {repo_dir} && {sys.executable} -m pip install -e . -q && kits23_download",
        desc="Downloading KiTS23 imaging data (this may take a while)",
        check=False,
    )


def download_lits(output_dir):
    """
    LiTS — ~131 CT volumes, liver/tumor segmentation.
    Available through Academic Torrents or Synapse.
    """
    ds_dir = Path(output_dir) / "lits"
    ds_dir.mkdir(parents=True, exist_ok=True)

    script = ds_dir / "download_lits.sh"
    script.write_text(f"""#!/bin/bash
# LiTS Dataset Download Script
#
# The LiTS dataset is available through multiple channels:
#
# Option A: Synapse (requires account)
#   1. Go to https://www.synapse.org/#!Synapse:syn3379050
#   2. Accept the terms and download
#
# Option B: Academic Torrents
#   pip install academictorrents
#   python -c "
#   import academictorrents as at
#   # Training volumes
#   at.get('27772adef6f563a1a992c9bf2eb2ea0a726d4c3e', datastore='{ds_dir}')
#   "
#
# Option C: Grand Challenge
#   1. Go to https://competitions.codalab.org/competitions/17094
#   2. Register and download
#
# After download, ensure files are in:
#   {ds_dir}/Training_Batch1/
#   {ds_dir}/Training_Batch2/
""")
    script.chmod(0o755)
    print(f"  [INFO] LiTS requires registration. Download script: {script}")

    # Try Academic Torrents
    if ensure_pip_package("academictorrents"):
        print("  Attempting download via Academic Torrents...")
        run_cmd(
            f'{sys.executable} -c "import academictorrents as at; '
            f"at.get('27772adef6f563a1a992c9bf2eb2ea0a726d4c3e', datastore='{ds_dir}')\"",
            desc="Downloading LiTS via Academic Torrents",
            check=False,
        )


def download_brats(output_dir):
    """
    BraTS 2023 — ~1251 MRI volumes, brain tumor segmentation.
    Available through Synapse.
    """
    ds_dir = Path(output_dir) / "brats"
    ds_dir.mkdir(parents=True, exist_ok=True)

    script = ds_dir / "download_brats.sh"
    script.write_text(f"""#!/bin/bash
# BraTS 2023 Dataset Download Script
#
# Option A: Synapse (recommended)
#   1. Go to https://www.synapse.org/#!Synapse:syn51156910
#   2. Create account and accept data use agreement
#   3. pip install synapseclient
#   4. python -c "
#      import synapseclient
#      syn = synapseclient.Synapse()
#      syn.login('YOUR_USERNAME', 'YOUR_PASSWORD')
#      syn.get('syn51156910', downloadLocation='{ds_dir}')
#      "
#
# Option B: Kaggle (BraTS 2021 subset)
#   kaggle datasets download -d dschettler8845/brats-2021-task1
#   unzip brats-2021-task1.zip -d {ds_dir}
#
# Option C: Direct links (BraTS 2020/2021)
#   Check: https://www.med.upenn.edu/cbica/brats2020/data.html

echo "BraTS requires Synapse registration. See instructions above."
""")
    script.chmod(0o755)

    # Try Kaggle download for BraTS 2021 (subset, easier to get)
    if shutil.which("kaggle"):
        print("  Attempting BraTS download via Kaggle API...")
        run_cmd(
            f"kaggle datasets download -d dschettler8845/brats-2021-task1 -p {ds_dir} --unzip",
            desc="Downloading BraTS via Kaggle",
            check=False,
        )
    else:
        print(f"  [INFO] BraTS requires registration. Download script: {script}")


def download_flare22(output_dir):
    """
    FLARE 2022 — 50 labeled + 2000 unlabeled CT volumes, abdomen.
    """
    ds_dir = Path(output_dir) / "flare22"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # FLARE labeled data is on Zenodo
    # Unlabeled data requires Grand Challenge registration
    labeled_url = "https://zenodo.org/records/7860267/files/FLARE22Train.zip"
    fpath = ds_dir / "FLARE22Train.zip"
    if download_file(labeled_url, fpath, desc="Downloading FLARE22 labeled set"):
        extract_archive(fpath, ds_dir)

    print("  [INFO] FLARE22 unlabeled set (2000 volumes) requires registration:")
    print("  https://flare22.grand-challenge.org/")


def download_acdc(output_dir):
    """
    ACDC — 100 cardiac MRI volumes.
    Available through CREATIS.
    """
    ds_dir = Path(output_dir) / "acdc"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # ACDC has a direct download link
    url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download"
    fpath = ds_dir / "acdc_training.zip"
    if download_file(url, fpath, desc="Downloading ACDC"):
        extract_archive(fpath, ds_dir)


def download_ct_org(output_dir):
    """
    CT-ORG — ~140 CT volumes, multi-organ.
    Available through TCIA wiki.
    """
    ds_dir = Path(output_dir) / "ct_org"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # CT-ORG is available on TCIA
    script = ds_dir / "download_ct_org.sh"
    script.write_text(f"""#!/bin/bash
# CT-ORG Dataset Download Script
#
# Available on TCIA:
#   https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890
#
# Option A: Using TCIA REST API
#   pip install tcia-utils
#   python -c "
#   from tcia_utils import nbia
#   nbia.downloadSeries(
#       collection='CT-ORG',
#       path='{ds_dir}'
#   )
#   "
#
# Option B: Direct NIfTI download from GitHub
#   The CT-ORG organizers provide NIfTI files:
#   https://github.com/bbrister/CTOrganSegment
#
# After download, files should be in:
#   {ds_dir}/volume-XX.nii.gz
#   {ds_dir}/labels-XX.nii.gz
""")
    script.chmod(0o755)
    print(f"  [INFO] CT-ORG download script saved: {script}")


def download_word(output_dir):
    """
    WORD — ~150 CT volumes, whole abdominal organs.
    Available on GitHub.
    """
    ds_dir = Path(output_dir) / "word"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # WORD dataset is hosted on a shared drive
    script = ds_dir / "download_word.sh"
    script.write_text(f"""#!/bin/bash
# WORD Dataset Download Script
#
# Paper: https://arxiv.org/abs/2111.02403
# GitHub: https://github.com/HiLab-git/WORD
#
# The dataset is hosted on Baidu Netdisk and Google Drive:
#   https://github.com/HiLab-git/WORD#data-download
#
# Steps:
#   1. Go to https://github.com/HiLab-git/WORD
#   2. Follow the download links in the README
#   3. Extract to {ds_dir}/
""")
    script.chmod(0o755)
    print(f"  [INFO] WORD download script saved: {script}")


def download_abdomenatlas(output_dir):
    """
    AbdomenAtlas 1.1 — ~20,460 CT volumes, abdomen.
    Available on HuggingFace. WARNING: ~1.5 TB!
    """
    ds_dir = Path(output_dir) / "abdomenatlas"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print("  [WARNING] AbdomenAtlas is ~1.5 TB. Consider downloading a subset.")
    print()

    script = ds_dir / "download_abdomenatlas.sh"
    script.write_text(f"""#!/bin/bash
# AbdomenAtlas 1.1 Dataset Download Script
# WARNING: This dataset is approximately 1.5 TB!
#
# Available on HuggingFace:
#   https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.1_Mini
#
# Option A: Download the Mini version first (~100 volumes, much smaller)
#   pip install huggingface_hub
#   python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download(
#       repo_id='AbdomenAtlas/AbdomenAtlas_1.1_Mini',
#       repo_type='dataset',
#       local_dir='{ds_dir}/mini',
#   )
#   "
#
# Option B: Full dataset
#   python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download(
#       repo_id='AbdomenAtlas/AbdomenAtlas_1.1',
#       repo_type='dataset',
#       local_dir='{ds_dir}/full',
#   )
#   "
#
# Option C: Using git lfs
#   cd {ds_dir}
#   git lfs install
#   git clone https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.1_Mini

echo "See options above. Recommend starting with Mini version."
""")
    script.chmod(0o755)

    # Try downloading the Mini version automatically
    if ensure_pip_package("huggingface_hub"):
        print("  Downloading AbdomenAtlas Mini (subset) via HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="AbdomenAtlas/AbdomenAtlas_1.1_Mini",
                repo_type="dataset",
                local_dir=str(ds_dir / "mini"),
            )
            print("  [OK] AbdomenAtlas Mini downloaded successfully.")
        except Exception as e:
            print(f"  [WARN] Auto-download failed: {e}")
            print(f"  See {script} for manual instructions.")


# ============================================================================
# Download dispatcher
# ============================================================================

DOWNLOAD_FUNCTIONS = {
    "totalseg": download_totalseg,
    "amos": download_amos,
    "kits": download_kits,
    "lits": download_lits,
    "brats": download_brats,
    "flare22": download_flare22,
    "acdc": download_acdc,
    "ct_org": download_ct_org,
    "word": download_word,
    "abdomenatlas": download_abdomenatlas,
}

# Datasets requiring manual registration (no auto-download)
MANUAL_ONLY = {"bcv", "autopet", "mms", "structseg", "pelvic"}


# ============================================================================
# CLI
# ============================================================================

def list_datasets():
    """Print all available datasets with details."""
    print("\n" + "=" * 90)
    print("Available 3D Medical Imaging Datasets")
    print("=" * 90)

    print("\n--- AUTO-DOWNLOADABLE ---\n")
    for key, info in DATASETS.items():
        if info["auto"]:
            print(f"  {key:18s} | {info['modality']:12s} | {info['n_volumes']:20s} | "
                  f"{info['size_gb']:12s} | {info['region']}")

    print("\n--- REQUIRES REGISTRATION (manual download) ---\n")
    for key, info in DATASETS.items():
        if not info["auto"]:
            print(f"  {key:18s} | {info['modality']:12s} | {info['n_volumes']:20s} | "
                  f"{info['size_gb']:12s} | {info['region']}")

    print(f"\n{'=' * 90}")
    total_auto = sum(1 for d in DATASETS.values() if d["auto"])
    total_manual = sum(1 for d in DATASETS.values() if not d["auto"])
    print(f"Total: {len(DATASETS)} datasets ({total_auto} auto-downloadable, {total_manual} manual)")
    print()


def show_manual_instructions(output_dir):
    """Print manual download instructions for all datasets requiring registration."""
    print("\n" + "=" * 70)
    print("Manual Download Instructions")
    print("=" * 70)

    for key, info in DATASETS.items():
        if not info["auto"]:
            print(f"\n{'─' * 70}")
            print(f"  {info['name']}")
            print(f"  Key: {key} | {info['modality']} | {info['n_volumes']} | {info['size_gb']}")
            print(f"  URL: {info.get('manual_url', 'N/A')}")
            print()
            instructions = info.get("manual_instructions", "See URL above.")
            instructions = instructions.replace("{output_dir}", str(output_dir))
            for line in instructions.strip().split("\n"):
                print(f"  {line}")
    print()


def download_datasets(output_dir, dataset_keys):
    """Download specified datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Datasets to download: {', '.join(dataset_keys)}")
    print()

    results = {"success": [], "skipped": [], "failed": []}

    for key in dataset_keys:
        if key not in DATASETS:
            print(f"[WARN] Unknown dataset: {key}")
            results["skipped"].append(key)
            continue

        info = DATASETS[key]
        print(f"\n{'=' * 60}")
        print(f"[{key}] {info['name']}")
        print(f"  Modality: {info['modality']} | Volumes: {info['n_volumes']} | Size: {info['size_gb']}")
        print(f"{'=' * 60}")

        if key in MANUAL_ONLY:
            print(f"  [MANUAL] This dataset requires registration.")
            instructions = info.get("manual_instructions", "See manual_url.")
            instructions = instructions.replace("{output_dir}", str(output_dir))
            for line in instructions.strip().split("\n"):
                print(f"  {line}")
            # Create placeholder directory
            (output_dir / key).mkdir(parents=True, exist_ok=True)
            results["skipped"].append(key)
            continue

        if key not in DOWNLOAD_FUNCTIONS:
            print(f"  [SKIP] No download function for {key}")
            results["skipped"].append(key)
            continue

        try:
            DOWNLOAD_FUNCTIONS[key](output_dir)
            results["success"].append(key)
        except Exception as e:
            print(f"  [FAIL] Error downloading {key}: {e}")
            results["failed"].append(key)

    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    print(f"  Completed: {', '.join(results['success']) or 'none'}")
    print(f"  Skipped (manual): {', '.join(results['skipped']) or 'none'}")
    print(f"  Failed: {', '.join(results['failed']) or 'none'}")
    print()

    # Write a manifest file
    manifest_path = output_dir / "DATASETS.md"
    with open(manifest_path, "w") as f:
        f.write("# Downloaded Medical 3D Datasets\n\n")
        f.write(f"Output directory: `{output_dir}`\n\n")
        f.write("| Key | Name | Modality | Volumes | Status |\n")
        f.write("|-----|------|----------|---------|--------|\n")
        for key in dataset_keys:
            if key in DATASETS:
                info = DATASETS[key]
                status = ("downloaded" if key in results["success"]
                          else "manual" if key in results["skipped"]
                          else "failed")
                f.write(f"| {key} | {info['name']} | {info['modality']} | "
                        f"{info['n_volumes']} | {status} |\n")
        f.write(f"\nGenerated by medical_dino3d/download_datasets.py\n")

    print(f"Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download 3D medical imaging datasets for pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python download_datasets.py --list

  # Download core datasets (recommended starting set, ~100 GB)
  python download_datasets.py -o /data/medical3d --datasets amos kits brats acdc

  # Download everything auto-downloadable
  python download_datasets.py -o /data/medical3d --all

  # Show manual download instructions
  python download_datasets.py -o /data/medical3d --manual

  # Download MASS paper's exact dataset list
  python download_datasets.py -o /data/medical3d --mass
        """,
    )
    parser.add_argument("-o", "--output_dir", type=str, default="./medical_3d_data",
                        help="Root directory to store all datasets")
    parser.add_argument("--datasets", nargs="+", type=str, default=[],
                        help="Specific datasets to download (e.g., amos kits brats)")
    parser.add_argument("--all", action="store_true",
                        help="Download all auto-downloadable datasets")
    parser.add_argument("--mass", action="store_true",
                        help="Download datasets used in the MASS paper")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets")
    parser.add_argument("--manual", action="store_true",
                        help="Show manual download instructions")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.manual:
        show_manual_instructions(args.output_dir)
        return

    # Determine which datasets to download
    if args.all:
        keys = list(DATASETS.keys())
    elif args.mass:
        # Datasets used in the MASS paper (Table 10)
        keys = [
            "bcv", "amos", "kits", "lits", "brats",
            "structseg",  # SS H&N + SS Thoracic
            "totalseg", "autopet", "mms", "pelvic",
            "acdc",
        ]
        print("Downloading datasets used in the MASS paper:")
        print(f"  {', '.join(keys)}")
    elif args.datasets:
        keys = args.datasets
    else:
        # Default: recommended starter set
        keys = ["amos", "kits", "brats", "acdc"]
        print("No datasets specified. Downloading recommended starter set:")
        print(f"  {', '.join(keys)}")
        print("Use --all for everything, or --datasets <name1> <name2> ... for specific ones.")
        print()

    download_datasets(args.output_dir, keys)


if __name__ == "__main__":
    main()
