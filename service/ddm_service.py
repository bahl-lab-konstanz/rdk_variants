from matplotlib import pyplot as plt
import numpy as np

from scipy.integrate import simpson
from scipy.special import rel_entr

from service.behavior_processing_service import BehaviorProcessingService
from service.statistics_service import StatisticsService
from utils.config import ConfigurationModel


class DDMService():
    @ staticmethod
    # test if I want to check simulation against data coming from a specific time window within the trial
    def is_window(key: str):
        try:
            return key.startswith("window")
        except AttributeError:
            return False
    
    @classmethod
    def dklstar_ibi_dist(cls, response_time_list_0, response_time_list_1, duration_0=None, duration_1=None, resolution=10, focus_scope=None, order_sum_result=False, order_max_result=False, correct_by_area=False, plot_distributions=False, perturbation=None):
        if focus_scope is None:
            focus_scope = cls.compute_focus_scope(response_time_list_0, response_time_list_1)
        if len(response_time_list_0) == 0:
            distribution_response_time_list_0 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_0, bin_list_0 = StatisticsService.get_hist(
                response_time_list_0, bins=resolution, hist_range=focus_scope,
                duration=duration_0, allow_zero=False, center_bin=True)
            
        if len(response_time_list_1) == 0:
            distribution_response_time_list_1 = np.ones(resolution) * np.finfo(float).eps
        else:
            distribution_response_time_list_1, bin_list_1 = StatisticsService.get_hist(
                response_time_list_1, bins=resolution, hist_range=focus_scope,
                duration=duration_1, allow_zero=False, center_bin=True)
            
        if perturbation is not None:
            distribution_response_time_list_0 = distribution_response_time_list_0 + perturbation
        distribution_response_time_list_0[distribution_response_time_list_0 <= 0] = np.finfo(float).eps

        weight_array = distribution_response_time_list_0 / np.sum(distribution_response_time_list_0)

        KL_divergence_response = sum(np.abs(rel_entr(distribution_response_time_list_0, distribution_response_time_list_1)) * weight_array)
        if correct_by_area:
            area_0 = simpson(distribution_response_time_list_0, dx=(focus_scope[1]-focus_scope[0])/resolution)
            KL_divergence_response = KL_divergence_response / np.sqrt(area_0)
        if order_max_result:
            KL_divergence_response_1 = sum(np.abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)) * weight_array)
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = max([KL_divergence_response_1, KL_divergence_response])

        elif order_sum_result:
            KL_divergence_response_1 = sum(np.abs(rel_entr(distribution_response_time_list_1, distribution_response_time_list_0)) * weight_array)
            if correct_by_area:
                area_1 = simpson(distribution_response_time_list_1, dx=(focus_scope[1] - focus_scope[0]) / resolution)
                KL_divergence_response_1 = KL_divergence_response_1 / np.sqrt(area_1)

            KL_divergence_response = KL_divergence_response_1 + KL_divergence_response

        if plot_distributions:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            try:
                ax.plot(bin_list_0, distribution_response_time_list_0, color="k", alpha=0.5)
            except UnboundLocalError:
                pass
            try:
                ax.plot(bin_list_1, distribution_response_time_list_1, "--", color="blue")
            except UnboundLocalError:
                pass
            fig.show()

        return KL_divergence_response

    @classmethod
    def compute_score_fit(cls, df_0, df_1, resolution=40, focus_scope=(0, 2), time_selection=None,
                          missing_data_score=100, time_experimental_trial=30, plot_distributions=False):
        score_dict = {}
        score = 0
        try:
            if time_selection is not None:
                if "start" in time_selection.keys():
                    df_0 = df_0.query(f"start_time > {time_selection['start']}")
                    df_1 = df_1.query(f"start_time > {time_selection['start']}")
                if "end" in time_selection.keys():
                    df_0 = df_0.query(f"end_time < {time_selection['end']}")
                    df_1 = df_1.query(f"end_time < {time_selection['end']}")
            analysed_parameter_list = set(df_0[ConfigurationModel.COLUMN_COH].unique()) & set(df_1[ConfigurationModel.COLUMN_COH].unique())
            for key in analysed_parameter_list:
                score_dict[key] = {}
                df_0_key = df_0[df_0[ConfigurationModel.COLUMN_COH] == key]
                df_1_key = df_1[df_1[ConfigurationModel.COLUMN_COH] == key]

                df_0_correct = df_0_key[df_0_key[ConfigurationModel.COLUMN_CORRECT_BOUT] == 1]
                df_1_correct = df_1_key[df_1_key[ConfigurationModel.COLUMN_CORRECT_BOUT] == 1]
                df_0_error = df_0_key[df_0_key[ConfigurationModel.COLUMN_CORRECT_BOUT] == 0]
                df_1_error = df_1_key[df_1_key[ConfigurationModel.COLUMN_CORRECT_BOUT] == 0]

                df_0_correct_response_time_list = np.array(df_0_correct[ConfigurationModel.COLUMN_IBI])
                df_1_correct_response_time_list = np.array(df_1_correct[ConfigurationModel.COLUMN_IBI])
                df_0_error_response_time_list = np.array(df_0_error[ConfigurationModel.COLUMN_IBI])
                df_1_error_response_time_list = np.array(df_1_error[ConfigurationModel.COLUMN_IBI])

                try:
                    duration_0 = np.sum(
                        BehaviorProcessingService.get_duration_trials_in_df(df_0_key,
                                                                       fixed_time_trial=time_experimental_trial))  # seconds
                except KeyError:
                    duration_0 = len(df_0_key['trial'].unique()) * time_experimental_trial # seconds

                try:
                    duration_1 = np.sum(BehaviorProcessingService.get_duration_trials_in_df(df_1_key,
                                                                       fixed_time_trial=time_experimental_trial))  # seconds
                except KeyError:
                    duration_1 = len(df_1_key['trial'].unique()) * time_experimental_trial # seconds

                KL_divergence_response_correct = cls.dklstar_ibi_dist(df_0_correct_response_time_list,
                                                                        df_1_correct_response_time_list,
                                                                        resolution=resolution,
                                                                        focus_scope=focus_scope,
                                                                        duration_0=duration_0,
                                                                        duration_1=duration_1,
                                                                        order_max_result=True,
                                                                        correct_by_area=False,
                                                                        plot_distributions=plot_distributions)
                score_dict[key]["corr"] = KL_divergence_response_correct
                KL_divergence_response_error = cls.dklstar_ibi_dist(df_0_error_response_time_list,
                                                                        df_1_error_response_time_list,
                                                                        resolution=resolution,
                                                                        focus_scope=focus_scope,
                                                                        duration_0=duration_0,
                                                                        duration_1=duration_1,
                                                                        order_max_result=True,
                                                                        correct_by_area=False,
                                                                        plot_distributions=plot_distributions)
                score_dict[key]["err"] = KL_divergence_response_error

                score = score + KL_divergence_response_correct + KL_divergence_response_error  # + np.abs(self.parameters.leak.value)**2  # + abs(np.log(area_target/area_simulation))  # + 10*(1-float(self.input_signal.label)/100) * self.parameters.scaling_factor.value  # + abs(area_simulation - area_target) + abs(median_rt_target - median_rt_simulation)

        except ValueError:
            score = missing_data_score + np.random.normal(scale=missing_data_score)
        score_dict["score"] = score

        return score_dict