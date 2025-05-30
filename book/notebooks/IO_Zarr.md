# Indian Ocean Dataset

**Author:** Minh Phan (UW Varanasi intern 2023), Eli Holmes (NOAA/UW)

Our Indian Ocean zarr dataset `INDIAN_OCEAN_025GRID_DAILY.zarr` or `IO.zarr` is a 1972-2022 blended dataset for the Arabian Sea and Bay of Bengal formated as a `.zarr` file, containing daily cleaned and interpolated data from variables across multiple sources, mostly from processed NASA/NOAA and Copernicus collections and the ERA5 reanalysis products.

### Variables

* `adt`: sea surface height above geoid (m) (SL_TAC)
* `air_temp`: air temperature at 2 meters above the surface (K), from 1979 (ERA5)
* `mlotst`: mean ocean mixed layer thickness (m) (GLORY)
* `sla`: sea level anomaly (m) (SL_TAC)
* `so`: sea salinity concentration (m**-3 or PSL) (GLORY)
* `sst`: sea surface temperature (K), from 1979 (ERA5)
* `topo`: topography (m) (SRTM30+)
* `u_curr`: u-component of total surface currents (m/s) (OSCAR)
* `v_curr`: v-component of total surface currents (m/s) (OSCAR)
* `ug_curr`: u-component of geostrophic surface currents (m/s) (OSCAR)
* `vg_curr`: v-component of geostrophic surface currents (m/s)
* `u_wind`: u-component of surface wind (m/s), from 1979 (ERA5)
* `v_wind`: v-component of surface wind (m/s), from 1979 (ERA5)

### Derived variables

* `curr_speed`: total current speed (m/s), computed from `u_curr` and `v_curr`
* `curr_dir`: total current direction (degrees) (OSCAR), computed from `u_curr` and `v_curr`
* `wind_speed`: surface wind speed (m/s), computed from `u_wind` and `v_wind`
* `wind_dir`: surface wind direction (degrees), computed from `u_wind` and `v_wind`

### Chlorophyll variables

Part of the purpose of the dataset was to study ocean color gap-filling algorithms. Thus we include a variety of different comparison CHL datasets.

* `CHL_cmes-level3` Multi-sensor chlorophyll-a concentration (mg/m**3) estimates, from Oct 1997. Level 3 (means has gaps from clouds etc). `CHL_cmes-gapfree` is created from this product. (GlobColour)
* `CHL_cmes_flags-level3` 0=land, 1=observed, 2=NA (GlobColour)
* `CHL_cmes_uncertainty-level3` chlorophyll-a concentration uncertainty (%)  (GlobColour)
* `CHL_cmes-gapfree`: Gap-filled chlorophyll-a concentration (mg/m**3), from Oct 1997 (GlobColour)
* `CHL_cmes_flags-gapfree`: 0=land, 1=observed, 2=interpolated, 3=NA (GlobColour)
* `CHL_cmes_uncertainty-gapfree`: Chlorophyll-a concentration uncertainty (%)  (GlobColour)
* TO DO `CHL_cci`: multi-sensor chlorophyll-a concentration (mg/m**3), from Dec 1998 (CCI)
* TO DO `CHL_cci_uncertainty`: chlorophyll-a concentration uncertainty (rmsd)  (CCI)
* TO DO `CHL_dinoef`: Gap-free chlorophyll-a concentration (mg/m**3), from 2018 (DINOEF)
* TO DO `CHL_dinoef_uncertainty`: chlorophyll-a concentration uncertainty (rmsd)  (DINOEF)
* TO DO `CHL_dinoef_flag`: flag  (DINOEF)

All variables have been broadcasted to fit into the temporal range we have. Therefore, not all variable data are available at all times. Examine each individual variable before use.

### Sources

* ERA5: These are hourly data that have been averaged to daily data, with the addition of some additional hourly wind layers.
* GlobColour: CHL from the [GlobColour project](https://www.globcolour.info/) and accessed from Copernicus. There are two products. A Level 4 gap-filled product which is derived from a gappy Level 3 multi-sensor product. Gappy means still has cloud (etc) NaNs.
* GLORY: Global Ocean Physics Reanalysis (product code name GLORYS12V1) from Copernicus Marine Environment Monitoring Service (CMEMS).
* OSCAR: Ocean Surface Current Analyses Real-time (OSCAR) Version 2.0.
* SL_TAC:  Sea Level TAC product Global Ocean Gridded L 4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing
* SRTM30+: SRTM30+ Global 1-km Digital Elevation Model Version 11
* CCI: Ocean Color CCI product that merges multiple sensors.
* DINEOF: NOAA MSL12 Ocean Color, science quality, VIIRS multi-sensor (SNPP + NOAA-20), chlorophyll DINEOF gap-filled analysis
