# Implementation of paper - Five-point Fundamental Matrix Estimation for Uncalibrated Cameras

## TODO:
- project setup
- dataset handling
- evaluation (7 pt.)
- algorithm implementation

## Prerequisites

- Bash

- cmake 3.5.1 +

- C++ 14


## Setup

1. Download data, run
  ```
  ./download_data
  ```

2. Install requirements, run
  ```
  ./install_reqs
  ```

3. Build project dependencies, run (under constsruction)
  ```
  ./build_deps
  ```
  time to grab a coffee or do something else more productive than staring at a loading screen.


## Compile

Build project, run
  ```
  ./build
  ```
Which runs 'cmake' and 'make' for all the code in 'src/'. Target location will be in 'bin/'.


## Cleaning (under construction)

Different level of clean scripts, please take note, otherwise you might have to rebuild everything.
However none of them will clean the downloaded datasets, re-running `download_data` will handle that.

Run
```
./clean
```
to just remove the built artifacts for the project. This does not clean any
dependencies.

Run
```
./clean_built_deps
```
to just remove the build artifacts for the project (calls ./clean as well) and 
dependencies. Downloaded content will not be deleted.

Run
```
./clean_all
```
to remove all downloaded content, cleans all built artifacts (`clean_build_deps`). 
After this, `build_deps` will have to be re-run again before `build` to compile 
the project.


## OpenCV Libraries

Currenly only builds

- features2d
- xfeatures2d
- highgui
- imgcodecs
- imgprocs
- calib3d

For other extra libraries, edit `scripts/build_opencv.bash` 
under the flag `BUILD_LIST`. Remember to exclude spaces when adding extra 
libraries, otherwise they will be whitelisted by `cmake`.