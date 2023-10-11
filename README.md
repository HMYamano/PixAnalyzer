# PixAnalyzer
<!-- ![Logo](img/logo.png) -->
<img src="img/logo.png" width="40%">

PixAnalyzer is a tool for quantifying the motion of objects.
It calculates pixel changes and quantifies the overall motion, which cannot be quantified by tracking or object recognition.

## Installation

#### Step1: Clone this repository

#### Step2: Install [Anaconda](https://www.anaconda.com)

#### Step3: Build an environment using the yaml file

In Anacnda prompt
```
  conda env create -f <path/to/environment.yml>
```

#### Step4: Install PixAnalyzer

In Anacnda prompt
```
  conda acitate pixanalyzer
  pip install pixanalyzer
```

#### Step5: Start PixAnalyzer

In Anacnda prompt
```
  conda acitate pixanalyzer
  pixanalyzer
```
## How to use

#### Step1: Select video files.(*.mp4, *.avi files)

#### Step2: Select crop area using GUI.
<img src="sampledata/sample_jumping_spyder_video_croparea_draw.png" width="40%">

#### Step3: Select threshold for analyzing motion/deformation of objects.
It is recommended that the threshold be set so that the contours of the object are clearly observed and the background is eliminated as much as possible.
<img src="sampledata/sample_jumping_spyder_video_crop_threshold.png" width="40%">

#### Step4: Analyzing.
After analysis, graphs showing the time series of pixel change results, a heatmap, an exel file of the results, and a json file of the configuration values are output.
<img src="sampledata/sample_jumping_spyder_video_au_graph.png" width="40%">
<img src="sampledata/sample_jumping_spyder_video_heatmap.png" width="40%">
    
## Demo
In the "sampledata" folder, there are analyze sample_jumping_spyder_video.mp4 and other figures.


## Acknowledgements
 We thanks for PhD. Ryoya Tanaka for discussing.


## Release Note
- 2023/10/XX First release (Ver. 1.0)
## Feedback

If you have any feedback, please reach out to us by:

- Opening an issue in the repository

- E-mail: haya.m.yamano.neuro@gmail.com


## License

[MIT](https://choosealicense.com/licenses/mit/)

