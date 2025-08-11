Implement a cohesive SLAM synthetic data simulation including: trajectory of device, imu sensors data, 3D points, 2d measurements of 3p points on camera frames. 
Device can have multiple cameras, multiple IMUs. Device origin is at first IMU. 

Simulation data can be exported to json files for a typical SLAM system to process. 
Simulation data can be from TUM VIE (ex: mocap desk dataset with all relevant calibration data).

Also implement a GeneralEstimator which can extend to support Sliding Window BA or BA, EKF, SRIF estimation processes.

Implementation is in python including plotting for simulation data and SLAM system output data. 

Configuration management must use Pydantic for loading, validation, and type safety of all YAML configuration files. 

Plotting must be using plotly and save to html files. Plot types: trajectory & points, 2d measurements points of 3D points in camera frames, 2d scatters of IMU data. 

Implementation must have run.sh and python tooling scripts (using typer) under tools folder for easy setup, config, build, run and expand to support more commands later ( simulate, slam, dashboard, download).

- simulate: command for simulation; ex.: 'run.sh simulate circle'
- slam: for running one of slam techniques given simulation data input; ex. 'run.sh slam swba' for running sliding window bundle adjustment. Each slam run would store kpis as json file with date in file name under data/SLAM/ folder. This kpis json file will then be used for kpis dashboard visualization with plotly. 
- dashboard: for generating dashboard html static files for showing slam estimation runs kpis
- download: for downloading public datasets; for instance, 'run.sh download tum-vie mocap-desk' should download relevant data and calibration data and put in data/TUM-VIE/mocap-desk/ folder.


Proposed folder structure
slam_simulation/
├── run.sh                     # Main execution script
├── requirements.txt           # Dependencies
├── setup.py                  # Package configuration
├── config/                   # YAML configurations
├── src/                      # Core implementation
│   ├── common/ 
│   ├── simulation/
│   ├── plotting/
│   └── utils/
├── tools/
│   └── cli.py               # Command-line interface
|-- data/                    # data generated or download from TUM VIE
└── output/                  # Generated data & plots