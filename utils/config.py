from pathlib import Path
from dotenv import dotenv_values

class ConfigurationLocalPath():
    env_path = Path(__file__).parent.parent / "config.env"
    env = dotenv_values(env_path)

    PATH_DATA_ALL = Path(env["PATH_DATA_ALL"])  # before preprocessing
    PATH_DIR = Path(env["PATH_DIR"])            # preprocessed data, starting point for all analysis
    PATH_SAVE = Path(env["PATH_SAVE"])          # location where to save results 
    PATH_MODEL = Path(env["PATH_MODEL"])

class ConfigurationExperiment():
    TIME_REST_PRE  = 5   # seconds (duration pre-stimulus phase)
    TIME_STIMULUS  = 20  # seconds (duration stimulus phase)
    TIME_REST_POST = 5   # seconds (duration post-stimulus phase)
    TIME_BUFFER    = 2      # seconds (time after stimulus onset when behavioral transition happens)

    COLUMN_FISH_ID             = "folder_name"  # column to identify single fish in the df in output from experiment, before preprocess.py
    COLUMN_TIME_START          = "start_time"
    COLUMN_TIME_END            = "end_time"
    COLUMN_TIME_START_ABSOLUTE = "start_time_absolute"
    COLUMN_TIME_END_ABSOLUTE   = "end_time_absolute"
    COLUMN_TURN_ANGLE          = "estimated_orientation_change"
    COLUMN_IBI                 = "interbout_interval"
    COLUMN_END_X               = "end_x_position"
    COLUMN_END_Y               = "end_y_position"
    COLUMN_TRIAL               = "trial"
    COLUMN_TRIAL_FROM_START    = "trial_count_since_experiment_start"

    LABEL_LEFT_ANGLE  = "left"
    LABEL_RIGHT_ANGLE = "right"

    COHERENCE_LIST = [0, 12.5, 25, 50, 100]  # tested coherence levels (N.B.: their interpretation changes across stimulus variants)

class ConfigurationModel():
    # naming standard used by the DDM during fitting and simulation
    LABEL_LEFT_ANGLE  = -90  # degrees
    LABEL_RIGHT_ANGLE = +90  # degrees
    LABEL_STRAIGHT = -1

    COLUMN_FISH_ID             = "fish_id"
    COLUMN_DIR                 = "dir"
    COLUMN_COH                 = "coh"
    COLUMN_CORRECT_BOUT        = "correct_bout"
    COLUMN_TIME_START          = "start_time"
    COLUMN_TIME_END            = "end_time"
    COLUMN_TIME_START_ABSOLUTE = "start_time_absolute"
    COLUMN_TIME_END_ABSOLUTE   = "end_time_absolute"
    COLUMN_TURN_ANGLE          = "estimated_orientation_change"
    COLUMN_IBI                 = "interbout_interval"
    COLUMN_RADIUS              = "radius"
    COLUMN_STIM_VARIANT        = "stimulus_variant"
    COLUMN_TRIAL               = "trial"
    COLUMN_TRIAL_FROM_START    = "trial_count_since_experiment_start"

    MAX_SCORE = 999999

class ConfigurationAnalysis():
    NUMBER_RESAMPLING_BOOTSTRAPPING = 10000
    THRESHOLD_P_VALUE_SIGNIFICANT = 0.005
