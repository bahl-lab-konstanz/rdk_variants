import sys
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.interpolate import CubicSpline
from dotenv import dotenv_values

try:
    env_path = sys.argv[1]
except IndexError:
    env_path = "config.env"
env = dotenv_values(env_path)
PATH_DATA = Path(env["PATH_DIR"]) / "data" / "videos" / "results" / "motion_summary.csv"
PATH_SAVE = Path(env["PATH_DIR"]) / "data" / "videos" / "results"
noise_column = "noise_metrics"  # "median_spatial_sd"
threshold = 5e-4
STIMULUS_STRENGTH_ADD = np.array([12.5])
THETA_DEG = 90
STIMULUS_LABEL_REFERENCE = "CoherentDotsFrameBasedRedraw"

df = pd.read_csv(PATH_DATA)
stimulus_label_list = df["stimulus_label"].unique()
stimulus_strength_array = np.asarray(df["stimulus_strength"].unique())
motion_strength_ref = df[(df["stimulus_label"] == STIMULUS_LABEL_REFERENCE) & (df["stimulus_strength"] == 100)]["median_strength"].iloc[0]
noise_strength_ref = df[(df["stimulus_label"] == STIMULUS_LABEL_REFERENCE) & (df["stimulus_strength"] == 0)]["noise_metrics"].iloc[0]

is_stimulus_strength_add_list = np.array([s not in stimulus_strength_array for s in STIMULUS_STRENGTH_ADD], dtype=bool)
stimulus_strength_add_list = STIMULUS_STRENGTH_ADD[is_stimulus_strength_add_list]

stimulus_lookup = []
for i_s, stimulus_label in enumerate(stimulus_label_list):
    df_stim = df[df["stimulus_label"] == stimulus_label]
    median_strength_array = np.asarray(df_stim["median_strength"])
    std_strength_array = np.asarray(df_stim[noise_column])

    
    cs_motion = CubicSpline(stimulus_strength_array, df_stim["median_strength"])
    cs_noise = CubicSpline(stimulus_strength_array, df_stim["noise_metrics"])
    for s in stimulus_strength_add_list:
        df_stim = pd.concat([df_stim, pd.DataFrame([{"stimulus_label": stimulus_label,
                                                     "stimulus_strength": s,
                                                     "theta_deg": THETA_DEG,
                                                     "median_strength": cs_motion(s),
                                                     # "median_spatial_sd": None, 
                                                     "noise_metrics": cs_noise(s)}])], ignore_index=True)

    df_stim.sort_values(["stimulus_label", "stimulus_strength"], inplace=True, ignore_index=True)
    
    stimulus_strength_array_ = np.sort(np.concatenate([stimulus_strength_array, stimulus_strength_add_list]))
    for stimulus_strength in stimulus_strength_array_:
          df_stim_strength = df_stim[df_stim["stimulus_strength"] == stimulus_strength]
          
          motion_strength = np.abs(df_stim_strength["median_strength"].iloc[0])
          motion_strength = motion_strength if motion_strength > threshold else 0
          motion_strength /= motion_strength_ref
          
          noise_strength = np.abs(df_stim_strength["noise_metrics"].iloc[0])
          noise_strength = noise_strength if noise_strength > threshold else 0
          noise_strength /= noise_strength_ref

          stimulus_lookup.append({"stimulus_label": stimulus_label,
                                  "stimulus_strength": stimulus_strength,
                                  "motion_strength": motion_strength,
                                  "noise_strength": noise_strength})

df_stimulus_lookup = pd.DataFrame(stimulus_lookup)
df_stimulus_lookup.to_csv(PATH_SAVE / "stimulus_lookup.csv")