# mind-the-chl-gap-2024

This is the repo from the Geo-smart HackWeek 2024. The material is used in "mind-the-chl-gap" which is a series of tutorials on CNNs for the gap-filling and prediction problem.

#### Neural network models for Chloraphyll-a gap-filling for remote-sensing products

### 2024 GeoSMART Hackweek team:

* Shridhar Sinha
* Ares
* Gabe
* Qi Ge
* Andy Barrett
* Eli Holmes
* Robin Clancy

### Hackweek notes

Image to use for GPU if you need tensorflow `quay.io/pangeo/ml-notebook:2024.08.18`

[Pitch slide](https://docs.google.com/presentation/d/1YfBLkspba2hRz5pTHG9OF3o9WHv-yNemZDq2QKFCme0/edit?usp=sharing)
[Zotero library](https://www.zotero.org/groups/5595561/safs-interns-/library)
[Google doc](https://docs.google.com/document/d/1ADjtPFMy5mDxWJ_jhFhUWaBvjSd54YAfcc3d6araPCs/edit?usp=sharing)

The IO dataset is an analysis ready gridded zarr file for the Indian Ocean. It is in the Cryo hub. Load with
```
import xarray as xr
ds = xr.open_zarr("~/shared-public/mind_the_chl_gap/IO.zarr")
```
- Copernicus level 3 Chl-a (globColour) `ds["CHL_cmes-level3"]`
- Copernicus level 4 Chl-a (globColour) `ds["CHL_cmes-gapfree"]`
- Environmental variables: SST, E-W & N-S surface wind, surface air temp, E-W & N-S currents, sea surface anomaly, mean ocean mixed layer thickness, salinity, bathymetry, 
- 0.25 degree grid. So large but that makes it easier to work with.
- 1972 to 2022 but Chl is 1997-2022


### SAFS Varanasi Interns on this project

* 2024 [Shridhar Sinha](https://www.linkedin.com/in/shridhar-sinha-5b7125184/), University of Washington, Paul G. Allen School of Computer Science & Engineering, ssinha19@uw.edu
* 2024 Yifei Hang, University of Washington, Applied & Computational Mathematical Sciences, yhang2@uw.edu
* 2023 [Jiarui Yu](https://www.linkedin.com/in/jiarui-yu-0b0ab522b/), University of Washington, Applied & Computational Mathematical Sciences
* 2023 [Minh Phan](https://www.linkedin.com/in/minhphan03/), University of Washington, Applied & Computational Mathematical Sciences
* mentor [Elizabeth Eli Holmes](https://eeholmes.github.io/), NOAA Fisheries, UW SAFS


## Problem

Chlorophyll is a key indicator of plankton abundance and thus marine productivity and the health of ocean ecosystems. By estimating chlorophyll levels, scientists gauge the abundance of phytoplankton, which in turn reflects the overall productivity and food availability for many types of forage fish. Estimation of plankton abundance is also critical for study of the earth carbon cycle since the ocean cover ca 70% of the earth and phytoplankton are the "forests" of the ocean---capturing and releasing carbon as they bloom and die. Remote-sensing via ocean-color sensors allow one to derive an index of concentration of chlorophyll-a in the water based on the reflectance characteristics of plankton and allows us to use remote-sensing to monitor chlorophyll on a global scale. However clouds, especially, are a major problem for ocean color instruments. Clouds obstruct the view of the ocean surface, leading to gaps in the chlorophyll-a data. Close to 70% of the earth is covered by clouds at any time. Other problems, like sun-glint, also cause gaps but clouds are a very pervasive cause of gaps.  Gaps in ocean color measurements significantly hinder our ability to monitor ocean productivity and to estimate the plankton contribution to the carbon cycle.

| <img width="906" alt="image" src="https://github.com/user-attachments/assets/f8a098a0-ef7a-447d-b468-f00954e95094"> | 
|:--:| 
| *Raw gappy chl-a estimates and gap-filled chl-a map.* |

Some parts of the world are more impacted by clouds than others. The Arabian Sea between East Africa and India is one of these regions. This region is home to two important and dynamic coast upwelling zones that drive large seasonal chlorophyll blooms that in turn drive the fisheries in this region and are critical for regional food security. The chlorophyll blooms happen during the summer monsoon when clouds are thick and winds drive coastal upwelling. Lack of data on chlorophyll intensity means poor spatial predictions of where fish, especially oil sardines, will be. Much of the fishing in this region is artisanal, on small boats that cannot go far. The Indian government produces fishing predictions based, in part, on Chl-a estimates. When it is cloudy, as it typically is during the monsoon, the Chl-a data are missing and fishing predictions are more uncertain. *Also EE Holmes is part of [a team](https://hackweek-itcoocean.github.io/2023-Hackbook/team.html) working on using machine-learning for fisheries forecasting in the Arabian Sea*.

Convolutional neural networks are able to learn local patterns and features in images and use this information to create predictions. This type of model has been used successfully for other types of earth data gap-filling problems. Our work ([slideshow](https://docs.google.com/presentation/d/1etA0GIpuRJrahZnKkC36YrCN8A0dmnXPTETdv5v1m0E/edit?usp=sharing)) is testing a variety of different machine-learning models and comparing them to existing Chl-a gap-filling algorithms.  **For UWHackWeek 2024, we would like to add a new, experimental, type of NN,  physics informed NNs (PINNs) and LSTM PINNs, to our set of models. We also want to add visualizations and notebooks/tutorials to help others apply PINNs to their work.**

| <img width="623" alt="image" src="https://github.com/user-attachments/assets/4ea38c95-67e9-44ed-939f-3fb8ac6bcf54"> | 
|:--:| 
| *Results from our U-Net model. Top left panel is the Copernicus Level-4 (science grade) Gap-filled Chl-a globColour product and Top right panel is our U-Net gap-filled product using the Copernicus Level-3 (gappy) data plus co-located environmental variables. The gap-filling algorithms are very different. Our model's ability to match a science-grade product is very promising. Note, we do not know that the globColour product doing better at gap-filling since we have no way to produce estimates from the globColour algorithm and compare to non-missing pixels, i.e. we cannot do our 'fake' clouds tests.* |

### Overarching hackweek goals

The overarching goal for this hackweek is to compare physics informed neural network (PINN) to out standard convolutional neural nets (or U-Nets) with co-located environmental data. The main goal is to understand how adding physical constraints through a neural network can affect or improve the performance of gap-filling algorithms. We want to use the week to do some experimentation and tuning to better understand PINNs (they are pretty new!).

#### Methods

* Our plan is to use the DeepXDE library for PINN support along with a standard ConvLSTM architecture in Pytorch. The ConvLSTM is trained on the Level 3 data and the predictions are passed to a DeepONet from DeepXDE, this fits the predictions according to the PDE and conditions(subject to tuning) defined in the PINN. We have a basic notebook for this and a model that works "ok" but not better than our U-Net model.
* We have a process for testing model performance using "fake" clouds (obtained from actual cloud cover elsewhere in the data) in order to get an estimate of the gap-filling error of the model. This allows us to compare performance to other models or to other gap-filled products (like the Copernicus Level-4 globColour product). See below for the 'fake' cloud figure.
* Data: We have a zarr file for the Indian Ocean and Bay of Bengal with 26 years of gridded data on Chl-a and many co-located environmental variables. It is ready for machine-learning models! 

#### General goals 
- improve hyperparameters, model architecture, and input selection for a PINN
- perform model validation and testing
- learn about different physical constraints that can be applied and try them for this problem. We are particularly interested in coastal constraints and what to do with river deltas.
- improve visualizations
- write tutorials on PINNs to help others apply this new approach to their problems. PINNs are especially useful in fluid environments like the ocean.

#### Other goals
- right now our zarr file does not have turbidity or particulate matter or biogeochemical properties. It would be good to add those.
- we would like to add more Chl-a gapfree products to our dataset
- try other gap-filling models
- perform sensitivity analysis to quantify the importance of each co-located environmental variable. We have not done that yet for any of our models.
- use a model without Chl-a as a predictor variable and create Chl-a predictions for 1972 to present.

### Notebooks

TBD. Will add to the notebooks folder before the hackweek starts.

### Additional resources or background reading
- [DeepNetO](https://arxiv.org/abs/1910.03193)
- [2023 machine-learning and upwelling paper](https://github.com/SAFS-Varanasi-Internship/Summer-2023/blob/main/Internship_Report.pdf)
- [talk on 2024 summer results to date](https://docs.google.com/presentation/d/1etA0GIpuRJrahZnKkC36YrCN8A0dmnXPTETdv5v1m0E/present?usp=sharing)

### Testing with fake clouds

We use fake clouds to create a test set for each image. Each image has missing values from the real clouds and from our fake clouds. However the fake clouds have a real observation so we have a way to make predictions and then compare to the true observation. In order to create realistic clouds, we use the clouds from 10 days after the image as the additional fake clouds. Random missing pixels would not make good clouds since clouds are bigger than 1 pixel and the clouds have geometric shapes and gaps that would be hard to replicate with circles or squares. This shows an image with the real clouds in dark purple, fake clouds in green and the pixels we use as our observations in yellow.

| <img width="934" alt="image" src="https://github.com/user-attachments/assets/004f26bd-205e-4dac-bc9f-5d9fbad38f91"> | 
|:--:| 
| *Both green + yellow have been observed for this image. For predicting, we treat the green as missing (fake clouds). We use the yellow as our observations that help inform our estimates. We predict the green and compare our predictions to the actual observations for the green pixels.* |

## Building the book

See the `README.md` in the `book` directory.

