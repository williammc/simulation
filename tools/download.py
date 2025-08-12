"""
Download command implementation.
Downloads and prepares public datasets for SLAM evaluation.
"""

import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

# Dataset configurations
DATASET_CONFIGS = {
    "tum-vie": {
        "mocap-desk": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 1500,
            "description": "Desktop scene with motion capture ground truth",
            "format": "h5",
            "calibration": "calibration_a",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "mocap-1d-trans": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 800,
            "description": "1D translation with motion capture",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "mocap-3d-trans": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 900,
            "description": "3D translation with motion capture",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "mocap-6dof": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 1200,
            "description": "6DOF motion with motion capture",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "loop-floor0": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 2000,
            "description": "Loop closure scenario on floor 0",
            "format": "h5",
            "calibration": "calibration_a",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "skate-easy": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 1800,
            "description": "Skateboarding sequence - easy difficulty",
            "format": "h5",
            "calibration": "calibration_a",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "skate-hard": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 2200,
            "description": "Skateboarding sequence - hard difficulty",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "running-easy": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 1600,
            "description": "Running sequence - easy difficulty",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        },
        "bike-dark": {
            "url": "https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset",
            "size_mb": 2500,
            "description": "Biking in dark environment",
            "format": "h5",
            "calibration": "calibration_b",
            "has_events": True,
            "has_frames": True,
            "has_imu": True
        }
    },
    "euroc": {
        "mh-01": {
            "url": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
            "size_mb": 900,
            "description": "Machine Hall 01 - Easy difficulty",
            "format": "zip"
        },
        "v1-01": {
            "url": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
            "size_mb": 600,
            "description": "Vicon Room 1-01 - Easy difficulty",
            "format": "zip"
        }
    }
}


def download_mocap_calibration(output_dir: Path) -> int:
    """
    Download TUM-VIE mocap-IMU calibration files.
    
    Args:
        output_dir: Directory to save calibration files
    
    Returns:
        Exit code (0 for success)
    """
    import json as json_module
    
    urls = {
        "A": "https://tumevent-vi.vision.in.tum.de/mocap-imu-calibrationA.json",
        "B": "https://tumevent-vi.vision.in.tum.de/mocap-imu-calibrationB.json"
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold green]Downloading TUM-VIE Mocap-IMU Calibrations[/bold green]\n")
    
    for calib_type, url in urls.items():
        output_file = output_dir / f"mocap-imu-calibration{calib_type}.json"
        
        console.print(f"Downloading calibration {calib_type}...")
        console.print(f"  URL: [cyan]{url}[/cyan]")
        console.print(f"  Output: [cyan]{output_file}[/cyan]")
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, output_file)
            
            # Verify it's valid JSON and contains expected data
            with open(output_file, 'r') as f:
                data = json_module.load(f)
                # Check for TUM-VIE format with value0
                if "value0" in data and "T_imu_marker" in data["value0"]:
                    console.print(f"[green]✓[/green] Downloaded calibration{calib_type} successfully")
                    # Show transformation info
                    t_imu = data["value0"]["T_imu_marker"]
                    console.print(f"    T_imu_marker: p=[{t_imu['px']:.3f}, {t_imu['py']:.3f}, {t_imu['pz']:.3f}]")
                else:
                    console.print(f"[yellow]⚠[/yellow] Calibration{calib_type} format unexpected")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to download calibration{calib_type}: {e}")
            return 1
    
    console.print("\n[bold]Usage Instructions:[/bold]")
    console.print("• CalibrationA: loop-floor0:3, mocap-desk, mocap-desk2, skate-easy")
    console.print("• CalibrationB: all other sequences")
    console.print("\nUse with TUM-VIE reader:")
    console.print(f"  mocap_calib_path = Path('{output_dir}/mocap-imu-calibrationA.json')  # or B")
    
    return 0


def download_dataset(
    dataset: str,
    sequence: str,
    output: Optional[Path] = None,
) -> int:
    """
    Download public dataset for SLAM evaluation.
    
    Args:
        dataset: Dataset name (e.g., 'tum-vie', 'euroc', 'tum-vie-calib')
        sequence: Sequence name (e.g., 'mocap-desk', 'mh-01', or 'mocap' for calibration)
        output: Output directory for downloaded data
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Special case for mocap calibration download
    if dataset.lower() == "tum-vie-calib" or (dataset.lower() == "tum-vie" and sequence.lower() == "mocap-calib"):
        output_dir = output or Path("data/TUM-VIE/calibration")
        return download_mocap_calibration(output_dir)
    # Validate dataset and sequence
    dataset_lower = dataset.lower()
    sequence_lower = sequence.lower()
    
    if dataset_lower not in DATASET_CONFIGS:
        console.print(f"[red]✗ Unknown dataset: {dataset}[/red]")
        console.print(f"  Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        return 1
    
    if sequence_lower not in DATASET_CONFIGS[dataset_lower]:
        console.print(f"[red]✗ Unknown sequence: {sequence}[/red]")
        console.print(f"  Available sequences for {dataset}: {', '.join(DATASET_CONFIGS[dataset_lower].keys())}")
        return 1
    
    config = DATASET_CONFIGS[dataset_lower][sequence_lower]
    
    console.print(f"[bold green]Downloading Dataset[/bold green]")
    console.print(f"  Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"  Sequence: [cyan]{sequence}[/cyan]")
    console.print(f"  Description: {config['description']}")
    console.print(f"  Size: ~{config['size_mb']} MB")
    
    # Prepare output directory
    output_dir = output or Path(f"data/{dataset.upper()}/{sequence}")
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"  Output: [cyan]{output_dir}[/cyan]")
    
    # Check if already downloaded
    marker_file = output_dir / ".download_complete"
    if marker_file.exists():
        console.print("[yellow]⚠ Dataset already downloaded[/yellow]")
        console.print("  Delete the directory to re-download")
        return 0
    
    # For now, create placeholder files
    # TODO: Implement actual download with proper error handling
    console.print("\n[yellow]Note: Actual download not yet implemented[/yellow]")
    console.print("[yellow]Creating placeholder files...[/yellow]")
    
    # Create placeholder structure
    create_placeholder_dataset(output_dir, dataset_lower, sequence_lower)
    
    # Mark as complete
    marker_file.touch()
    
    console.print(f"\n[green]✓[/green] Dataset prepared: [cyan]{output_dir}[/cyan]")
    display_dataset_info(output_dir)
    
    return 0


def download_with_progress(url: str, output_file: Path) -> bool:
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        output_file: Local file path to save to
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get file size
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('Content-Length', 0))
        
        with Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "download",
                filename=output_file.name,
                total=total_size
            )
            
            with open(output_file, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
        
        return True
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        return False


def extract_archive(archive_file: Path, output_dir: Path, format: str) -> bool:
    """
    Extract downloaded archive.
    
    Args:
        archive_file: Path to archive file
        output_dir: Directory to extract to
        format: Archive format ('tgz', 'zip')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if format == 'tgz':
            with tarfile.open(archive_file, 'r:gz') as tar:
                tar.extractall(output_dir)
        elif format == 'zip':
            with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        else:
            console.print(f"[red]✗ Unknown archive format: {format}[/red]")
            return False
        
        return True
    except Exception as e:
        console.print(f"[red]✗ Extraction failed: {e}[/red]")
        return False


def create_placeholder_dataset(output_dir: Path, dataset: str, sequence: str) -> None:
    """Create placeholder dataset structure for testing."""
    
    if dataset == "tum-vie":
        # TUM-VIE dataset structure
        # Determine which calibration to use
        config = DATASET_CONFIGS["tum-vie"].get(sequence, {})
        calibration_type = config.get("calibration", "calibration_b")
        
        # Create directory structure
        (output_dir / "events").mkdir(exist_ok=True)
        (output_dir / "frames").mkdir(exist_ok=True)
        (output_dir / "imu").mkdir(exist_ok=True)
        (output_dir / "mocap").mkdir(exist_ok=True)
        
        # Create calibration JSON based on type
        calib_json = output_dir / f"{calibration_type}.json"
        
        if calibration_type == "calibration_a":
            # Calibration A: for loop-floor0:3, mocap-desk, mocap-desk2, skate-easy
            calib_data = {
                "comment": "Calibration A - Valid for loop-floor0:3, mocap-desk, mocap-desk2, skate-easy",
                "cameras": {
                    "cam0": {
                        "model": "kannala-brandt",
                        "intrinsics": {
                            "fx": 419.9,
                            "fy": 419.6,
                            "cx": 512.0,
                            "cy": 512.0,
                            "k1": 0.899,
                            "k2": -2.42,
                            "k3": 1.59,
                            "k4": -0.315
                        },
                        "resolution": [1024, 1024],
                        "T_cam_imu": {
                            "translation": [0.045, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "cam1": {
                        "model": "kannala-brandt",
                        "intrinsics": {
                            "fx": 419.5,
                            "fy": 419.2,
                            "cx": 512.0,
                            "cy": 512.0,
                            "k1": 0.907,
                            "k2": -2.45,
                            "k3": 1.61,
                            "k4": -0.321
                        },
                        "resolution": [1024, 1024],
                        "T_cam_imu": {
                            "translation": [-0.045, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "event0": {
                        "model": "pinhole",
                        "intrinsics": {
                            "fx": 656.8,
                            "fy": 656.8,
                            "cx": 640.0,
                            "cy": 360.0
                        },
                        "resolution": [1280, 720],
                        "T_cam_imu": {
                            "translation": [0.045, 0.02, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "event1": {
                        "model": "pinhole",
                        "intrinsics": {
                            "fx": 656.8,
                            "fy": 656.8,
                            "cx": 640.0,
                            "cy": 360.0
                        },
                        "resolution": [1280, 720],
                        "T_cam_imu": {
                            "translation": [-0.045, 0.02, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    }
                },
                "imu": {
                    "accelerometer": {
                        "noise_density": 0.00018,
                        "random_walk": 0.001,
                        "bias_stability": 0.0001
                    },
                    "gyroscope": {
                        "noise_density": 0.00026,
                        "random_walk": 0.0001,
                        "bias_stability": 0.0001
                    },
                    "update_rate": 200.0
                }
            }
        else:
            # Calibration B: for all other sequences
            calib_data = {
                "comment": "Calibration B - Valid for all sequences except loop-floor0:3, mocap-desk, mocap-desk2, skate-easy",
                "cameras": {
                    "cam0": {
                        "model": "kannala-brandt",
                        "intrinsics": {
                            "fx": 420.5,
                            "fy": 420.1,
                            "cx": 512.0,
                            "cy": 512.0,
                            "k1": 0.912,
                            "k2": -2.48,
                            "k3": 1.63,
                            "k4": -0.329
                        },
                        "resolution": [1024, 1024],
                        "T_cam_imu": {
                            "translation": [0.045, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "cam1": {
                        "model": "kannala-brandt",
                        "intrinsics": {
                            "fx": 420.2,
                            "fy": 419.8,
                            "cx": 512.0,
                            "cy": 512.0,
                            "k1": 0.918,
                            "k2": -2.51,
                            "k3": 1.65,
                            "k4": -0.335
                        },
                        "resolution": [1024, 1024],
                        "T_cam_imu": {
                            "translation": [-0.045, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "event0": {
                        "model": "pinhole",
                        "intrinsics": {
                            "fx": 656.8,
                            "fy": 656.8,
                            "cx": 640.0,
                            "cy": 360.0
                        },
                        "resolution": [1280, 720],
                        "T_cam_imu": {
                            "translation": [0.045, 0.02, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    },
                    "event1": {
                        "model": "pinhole",
                        "intrinsics": {
                            "fx": 656.8,
                            "fy": 656.8,
                            "cx": 640.0,
                            "cy": 360.0
                        },
                        "resolution": [1280, 720],
                        "T_cam_imu": {
                            "translation": [-0.045, 0.02, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0]
                        }
                    }
                },
                "imu": {
                    "accelerometer": {
                        "noise_density": 0.00018,
                        "random_walk": 0.001,
                        "bias_stability": 0.0001
                    },
                    "gyroscope": {
                        "noise_density": 0.00026,
                        "random_walk": 0.0001,
                        "bias_stability": 0.0001
                    },
                    "update_rate": 200.0
                }
            }
        
        # Write calibration JSON
        import json
        with open(calib_json, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        # Create sample data files
        (output_dir / "events" / "left_events.h5").touch()
        (output_dir / "events" / "right_events.h5").touch()
        (output_dir / "frames" / "left_frames.h5").touch()
        (output_dir / "frames" / "right_frames.h5").touch()
        (output_dir / "imu" / "imu_data.csv").write_text(
            "#timestamp [ns],w_x [rad/s],w_y [rad/s],w_z [rad/s],a_x [m/s^2],a_y [m/s^2],a_z [m/s^2]\n"
        )
        (output_dir / "mocap" / "gt_data.csv").write_text(
            "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n"
        )
        
    elif dataset == "euroc":
        # EuRoC dataset structure
        (output_dir / "mav0").mkdir(exist_ok=True)
        (output_dir / "mav0" / "cam0").mkdir(exist_ok=True)
        (output_dir / "mav0" / "cam1").mkdir(exist_ok=True)
        (output_dir / "mav0" / "imu0").mkdir(exist_ok=True)
        (output_dir / "mav0" / "state_groundtruth_estimate0").mkdir(exist_ok=True)
        
        # Similar structure to TUM-VI
        console.print("  Created EuRoC dataset structure")
    
    # Create README
    readme = output_dir / "README.md"
    readme.write_text(f"""
# {dataset.upper()} Dataset - {sequence}

This is a placeholder dataset structure for testing.
To download the actual dataset, the download functionality needs to be implemented.

## Structure
- `mav0/cam0/` - Left camera data
- `mav0/cam1/` - Right camera data  
- `mav0/imu0/` - IMU data
- `mav0/state_groundtruth_estimate0/` - Ground truth trajectory

## Usage
Use the SLAM commands to process this dataset:
```bash
./run.sh slam ekf --input {output_dir}/processed.json
```
""")


def display_dataset_info(output_dir: Path) -> None:
    """Display information about the downloaded dataset."""
    # Count files and calculate size
    total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    
    console.print("\n[bold]Dataset Info:[/bold]")
    console.print(f"  Location: {output_dir}")
    console.print(f"  Files: {total_files}")
    console.print(f"  Size: {total_size / 1024:.1f} KB")
    
    # List main directories
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if subdirs:
        console.print("  Structure:")
        for d in subdirs[:5]:  # Show first 5 directories
            console.print(f"    - {d.name}/")
        if len(subdirs) > 5:
            console.print(f"    ... and {len(subdirs) - 5} more")


def list_available_datasets() -> None:
    """List all available datasets and sequences."""
    console.print("[bold]Available Datasets:[/bold]\n")
    
    for dataset_name, sequences in DATASET_CONFIGS.items():
        console.print(f"[cyan]{dataset_name}[/cyan]")
        for seq_name, seq_config in sequences.items():
            console.print(f"  • {seq_name}: {seq_config['description']}")
            console.print(f"    Size: ~{seq_config['size_mb']} MB")
        console.print()