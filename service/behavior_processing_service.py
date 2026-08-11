from typing import List

import numpy as np

from service.statistics_service import StatisticsService
from utils.config import ConfigurationExperiment, ConfigurationModel


class BehaviorProcessingService():

    @classmethod
    def tag_stimulus_info(cls, df):
        def _tag_group(g):
                name = g.index.get_level_values('stimulus_name')[0]
                variant, coherence, direction = name.rsplit("_", 2)
                if direction == ConfigurationExperiment.LABEL_LEFT_ANGLE:
                    direction = ConfigurationModel.LABEL_LEFT_ANGLE
                elif direction == ConfigurationExperiment.LABEL_RIGHT_ANGLE:
                    direction = ConfigurationModel.LABEL_RIGHT_ANGLE
                g = g.copy()
                g[ConfigurationModel.COLUMN_STIM_VARIANT] = variant
                g[ConfigurationModel.COLUMN_COH]  = coherence
                g[ConfigurationModel.COLUMN_DIR]  = direction
                return g
        
        return df.groupby(level='stimulus_name', group_keys=False).apply(_tag_group)
    
    @staticmethod
    def compute_quantities_per_parameters(df, column_name=ConfigurationModel.COLUMN_CORRECT_BOUT, analysed_parameter=ConfigurationModel.COLUMN_COH):
        parameter_list = np.sort(df[analysed_parameter].unique())

        if column_name == ConfigurationModel.COLUMN_CORRECT_BOUT:
            df = df[df[column_name] != -1]

        quantity_list = np.array([df[df[analysed_parameter] == parameter][column_name].mean()
                             for parameter in parameter_list])
        std_quantity_list = np.array([df[df[analysed_parameter] == parameter][column_name].std() for parameter in parameter_list])
        return parameter_list, quantity_list, std_quantity_list


    @classmethod
    def compute_correct_bout_list(cls, df, side_threshold=0, inverted_stimulus=False, simmetric=False):
        # if simmetric is set to True
        #    - label_corr = 1
        #    - label_straight = 0
        #    - label_err = -1
        # else
        #    - label_corr = 1
        #    - label_straight = -1
        #    - label_err = 0

        if ConfigurationModel.COLUMN_DIR not in df.columns:
             df = cls.tag_stimulus_info(df)

        sign_correction = -1 if inverted_stimulus else 1
        correct_bout_list = np.sign(np.array(df[ConfigurationExperiment.COLUMN_TURN_ANGLE]) * np.array(df[ConfigurationModel.COLUMN_DIR])) * sign_correction
        if not simmetric:
            correct_bout_list = np.sign(1+correct_bout_list)

        index_straight_bout = np.argwhere(abs(df[ConfigurationExperiment.COLUMN_TURN_ANGLE]) <= side_threshold)
        correct_bout_list[index_straight_bout] = 0 if simmetric else -1
        return correct_bout_list

    def compute_radius(df):
        return np.sqrt(np.array(df[ConfigurationExperiment.COLUMN_END_X])**2 + np.array(df[ConfigurationExperiment.COLUMN_END_Y])**2)
    
    @staticmethod
    def remove_border_bout(df, border_threshold):
        return df[np.abs(df["radius"]) < border_threshold]

    @staticmethod
    def transform_arena_measure(measure_in_cm, radius_arena_in_cm=6):
        return measure_in_cm / radius_arena_in_cm

    @staticmethod
    def remove_fast_straight_bout(df, threshold_response_time=0.1):
        invalid_condition = np.array(StatisticsService.and_list(df[ConfigurationModel.COLUMN_CORRECT_BOUT] == ConfigurationModel.LABEL_STRAIGHT,
                                                                df[ConfigurationModel.COLUMN_IBI] < threshold_response_time))

        return df.iloc[~invalid_condition]

    @classmethod
    def get_duration_trials_in_df(cls, df, properties_to_group_by: List[str] = None, time_column="end_time", fixed_time_trial: float = None, verbose=False):
        if properties_to_group_by is None:
            properties_to_group_by = ["folder_name", "trial_count_since_experiment_start", "trial"]
        # exclude terms that are not in index (maybe dataframe has already be filtered by index)
        properties_to_group_by_in_df = [p for p in properties_to_group_by if p in df.index.names]
        diff_list = list(set(properties_to_group_by) - set(properties_to_group_by_in_df))
        if len(diff_list) > 0 and verbose:
            print(f"WARNING | df does not contain the following indices: {', '.join(diff_list)}")
        if len(properties_to_group_by_in_df) == 0:
            time_list = len(df["trial"].unique()) * fixed_time_trial
        elif fixed_time_trial is None:
            time_list = df.groupby(level=properties_to_group_by_in_df).max()[time_column]
        else:
            time_list = df.groupby(level=properties_to_group_by_in_df).count().shape[0] * [fixed_time_trial]
        return time_list
