# DEPENDENCIES
import pandas as pd
from utils.config import ConfigurationExperiment, ConfigurationLocalPath, ConfigurationModel
from service.behavior_processing_service import BehaviorProcessingService


# SCRIPT CONFIGURATIONS
# behavior
side_threshold = 3  # degrees if |turn angle|<side_threshold then it is considered straight bout
# paths
path_data_all = ConfigurationLocalPath.PATH_DATA_ALL
path_save = ConfigurationLocalPath.PATH_SAVE

# PREPROCESSING
# load combined dataset
df_all = pd.read_hdf(path_data_all, key="all_bout_data_pandas")
# find list of fish contained in the dataframe
fish_id_list = df_all.index.unique(ConfigurationExperiment.COLUMN_FISH_ID)

fish_n = 0
for fish_id in fish_id_list:
    # filter df by fish id
    df_fish = df_all.xs(fish_id, level=ConfigurationExperiment.COLUMN_FISH_ID)
    # add stimulus details to the df
    df_fish = BehaviorProcessingService.tag_stimulus_info(df_fish)
    # identify which stimulus variant this fish underwent
    variant = list(df_fish[ConfigurationModel.COLUMN_STIM_VARIANT])[0]
    # create dataframe
    df_out = pd.DataFrame({
        ConfigurationModel.COLUMN_FISH_ID: fish_id,
        ConfigurationModel.COLUMN_TRIAL: df_fish.index.get_level_values(ConfigurationExperiment.COLUMN_TRIAL_FROM_START),
        ConfigurationModel.COLUMN_TIME_START: df_fish[ConfigurationExperiment.COLUMN_TIME_START],
        ConfigurationModel.COLUMN_TIME_END: df_fish[ConfigurationExperiment.COLUMN_TIME_END],
        ConfigurationModel.COLUMN_TIME_START_ABSOLUTE: df_fish[ConfigurationExperiment.COLUMN_TIME_START_ABSOLUTE],
        ConfigurationModel.COLUMN_TIME_END_ABSOLUTE: df_fish[ConfigurationExperiment.COLUMN_TIME_END_ABSOLUTE],
        ConfigurationModel.COLUMN_IBI: df_fish[ConfigurationExperiment.COLUMN_IBI],
        ConfigurationModel.COLUMN_TURN_ANGLE: df_fish[ConfigurationExperiment.COLUMN_TURN_ANGLE],
        ConfigurationModel.COLUMN_CORRECT_BOUT: BehaviorProcessingService.compute_correct_bout_list(df_fish, side_threshold=side_threshold),
        ConfigurationModel.COLUMN_DIR: df_fish[ConfigurationModel.COLUMN_DIR],
        ConfigurationModel.COLUMN_COH: df_fish[ConfigurationModel.COLUMN_COH],
        ConfigurationModel.COLUMN_RADIUS: BehaviorProcessingService.compute_radius(df_fish)
    }).reset_index(drop=True)

    # create parent directory if needed
    (path_save / variant).mkdir(parents=True, exist_ok=True)
    # store single-fish dataframe
    df_out.to_hdf(str(path_save / str(variant) / f"data_fish_{fish_n:03}.hdf5"), key="all_events", complevel=9)

    fish_n += 1
    print(f"Fish {fish_n:03} saved | ID was {fish_id}")

print("\nPreprocessing completed.")
