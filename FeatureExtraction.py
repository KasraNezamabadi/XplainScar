import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from collections import Counter
from QTSegmentExtractor import QTSegmentExtractor
import GlobalPaths
from tslearn.preprocessing import TimeSeriesScalerMinMax
from Utility import Util, SignalProcessing
import heapq
from scipy.stats import linregress
from tqdm import tqdm


class Wave:
    def __init__(self, peak_index: int, prominence: float, width: float, amp: float = None):
        self.peak_index = peak_index
        self.prominence = prominence
        self.width = width
        self.amp = amp


class ECGUtility:

    @staticmethod
    def get_positive_waves(segment: [float], min_prominence: float, global_max: int) -> [Wave]:
        result = []
        all_peak_indexes, _ = signal.find_peaks(x=segment)
        all_peak_prominences = signal.peak_prominences(x=segment, peaks=all_peak_indexes)[0]
        all_peak_widths = signal.peak_widths(x=segment, peaks=all_peak_indexes, rel_height=1)[0]

        for i in range(len(all_peak_indexes)):
            if all_peak_indexes[i] != global_max and all_peak_prominences[i] >= min_prominence:
                wave = Wave(peak_index=all_peak_indexes[i], amp=segment[all_peak_indexes[i]],
                            prominence=all_peak_prominences[i], width=all_peak_widths[i])
                result.append(wave)
        return sorted(result, key=lambda x: x.peak_index)

    @staticmethod
    def find_global_extremum(segment_orig: [float]):
        peak_indexes, _ = signal.find_peaks(x=segment_orig)
        peak_prominences = signal.peak_prominences(x=segment_orig, peaks=peak_indexes)[0]
        peak_widths = signal.peak_widths(x=segment_orig, peaks=peak_indexes, rel_height=1)[0]

        segment_inverted = [-1 * x for x in segment_orig]
        valley_indexes, _ = signal.find_peaks(x=segment_inverted)
        valley_prominences = signal.peak_prominences(x=segment_inverted, peaks=valley_indexes)[0]
        valley_widths = signal.peak_widths(x=segment_inverted, peaks=valley_indexes)[0]

        extrema_idx = np.concatenate((peak_indexes, valley_indexes))
        extrema_prominences = np.concatenate((peak_prominences, valley_prominences))
        extrema_widths = np.concatenate((peak_widths, valley_widths))

        extrema = zip(extrema_idx, extrema_prominences, extrema_widths)
        extrema = [ext for ext in extrema if ext[0] > 2]
        if len(extrema) > 0:
            extremum_idx_with_max_prom = max(extrema, key=lambda x: x[1])
            return Wave(peak_index=extremum_idx_with_max_prom[0],
                        prominence=extremum_idx_with_max_prom[1],
                        width=extremum_idx_with_max_prom[2],
                        amp=segment_orig[extremum_idx_with_max_prom[0]])
        return None

    @staticmethod
    def identify_qs(qrs_segment: [float], r_index: int, base_amp: float, min_prom: float):
        result = {'Q': -1, 'S': -1}

        inverted_segment = [-1 * x for x in qrs_segment]
        valley_indexes, _ = signal.find_peaks(x=inverted_segment)
        valley_prominences = signal.peak_prominences(x=inverted_segment, peaks=valley_indexes)[0]
        valley_widths = signal.peak_widths(x=inverted_segment, peaks=valley_indexes, rel_height=1)[0]

        all_negative_waves = []
        for i in range(len(valley_indexes)):
            wave = Wave(peak_index=valley_indexes[i], prominence=valley_prominences[i], width=valley_widths[i],
                        amp=qrs_segment[valley_indexes[i]])
            all_negative_waves.append(wave)

        significant_negative_waves = []
        for wave in all_negative_waves:
            if wave.prominence > min_prom:
                significant_negative_waves.append(wave)
        significant_negative_waves = sorted(significant_negative_waves, key=lambda x: x.peak_index)
        left_valleys = [wave for wave in significant_negative_waves if wave.peak_index < r_index]
        right_valleys = [wave for wave in significant_negative_waves if wave.peak_index > r_index]

        if len(left_valleys) > 0:
            deepest_valley = max(left_valleys, key=lambda x: abs(x.amp))
            if deepest_valley.amp < min(base_amp, 0) and deepest_valley.prominence > max(50.0, min_prom):
                result['Q'] = deepest_valley.peak_index

        if len(right_valleys) > 0:
            deepest_valley = max(right_valleys, key=lambda x: abs(x.amp))
            if deepest_valley.amp < min(base_amp, 0):
                result['S'] = deepest_valley.peak_index

        return result

    @staticmethod
    def identify_qrs_offset(qt_segment: [float], base_prom, base_amp: float) -> int:
        qrs_segment = qt_segment[:round(len(qt_segment) / 2.5)]

        peak_indexes, _ = signal.find_peaks(x=qrs_segment)
        peak_prominences = signal.peak_prominences(x=qrs_segment, peaks=peak_indexes)[0]
        peak_widths = signal.peak_widths(x=qrs_segment, peaks=peak_indexes, rel_height=1)[0]

        valley_indexes, _ = signal.find_peaks(x=[-x for x in qrs_segment])
        valley_prominences = signal.peak_prominences(x=[-x for x in qrs_segment], peaks=valley_indexes)[0]
        valley_widths = signal.peak_widths(x=[-x for x in qrs_segment], peaks=valley_indexes)[0]

        peak_waves = [Wave(peak_index=peak_indexes[i],
                           prominence=peak_prominences[i],
                           width=peak_widths[i],
                           amp=qrs_segment[peak_indexes[i]]) for i in range(len(peak_indexes))]

        valley_waves = [Wave(peak_index=valley_indexes[i],
                             prominence=valley_prominences[i],
                             width=valley_widths[i],
                             amp=qrs_segment[valley_indexes[i]]) for i in range(len(valley_indexes))]

        waves = peak_waves + valley_waves
        waves = sorted(waves, key=lambda wave: wave.peak_index, reverse=True)

        last_significant_wave = None
        for wave in waves:
            if wave.prominence > max(50, 2 * base_prom) and abs(wave.amp - base_amp) > 50:
                last_significant_wave = wave
                break

        qrs_offset = -1
        max_dist = 0

        p_start = np.asarray((last_significant_wave.peak_index, qrs_segment[last_significant_wave.peak_index]))
        p_end = np.asarray((len(qrs_segment) - 1, qrs_segment[-1]))

        for i in range(last_significant_wave.peak_index + 1, len(qrs_segment)):
            point = np.asarray((i, qrs_segment[i]))
            distance = np.linalg.norm(np.cross(p_end - p_start, p_start - point)) / np.linalg.norm(p_end - p_start)
            if distance > max_dist:
                max_dist = distance
                qrs_offset = i

        return qrs_offset

    @staticmethod
    def parse_t_wave(qt_segment, qrs_offset: int):
        t_segment = qt_segment[qrs_offset + 1:]
        t_segment_norm = ECGUtility.normalize(segment=qt_segment)[qrs_offset + 1:]
        t_waves, st_line, t_onset = ECGUtility.find_t_peak(t_segment_norm=t_segment_norm, t_segment_orig=t_segment)
        return t_waves, st_line, t_onset

    @staticmethod
    def normalize(segment: [float]):
        segment_norm = np.array(TimeSeriesScalerMinMax(value_range=(-1, 1)).fit_transform([segment]))
        return np.reshape(segment_norm, (segment_norm.shape[0], segment_norm.shape[1])).ravel()

    @staticmethod
    def find_t_peak(t_segment_norm: [float], t_segment_orig: [float]):
        t_waves = []  # Single T-wave or biphasic T-wave.

        smooth_w_len = round(len(t_segment_norm) / 10)
        t_segment_norm_smooth = SignalProcessing.smooth(x=t_segment_norm, window_len=smooth_w_len, window='flat')
        t_segment_orig_smooth = SignalProcessing.smooth(x=t_segment_orig, window_len=smooth_w_len, window='flat')

        t_segment_norm = t_segment_norm_smooth[
                         round(smooth_w_len / 2) - 1: len(t_segment_norm_smooth) - round(smooth_w_len / 2)]
        t_segment_orig = t_segment_orig_smooth[
                         round(smooth_w_len / 2) - 1: len(t_segment_orig_smooth) - round(smooth_w_len / 2)]

        t_wave = ECGUtility.find_global_extremum(segment_orig=t_segment_orig)
        if t_wave is None:
            raise AssertionError('could not identify global T peak.')
        t_waves.append(t_wave)

        if t_wave.amp > 0:
            inverted_t_segment = [-1 * x for x in t_segment_norm]
        else:
            inverted_t_segment = t_segment_norm

        valley_indexes, _ = signal.find_peaks(x=inverted_t_segment)
        valley_prominences = signal.peak_prominences(x=inverted_t_segment, peaks=valley_indexes)[0]
        valley_widths = signal.peak_widths(x=inverted_t_segment, peaks=valley_indexes, rel_height=1)[0]
        if len(valley_indexes) > 0:
            valleys = zip(valley_indexes, valley_prominences, valley_widths)
            valley_max = max(valleys, key=lambda x: x[1])
            if t_wave.prominence - 0.2 * t_wave.prominence < valley_max[1] < t_wave.prominence + 0.2 * t_wave.prominence:
                t2_wave = Wave(peak_index=valley_max[0], prominence=valley_max[1], width=valley_max[2], amp=t_segment_orig[valley_max[0]])
                t_waves.append(t2_wave)

        left_most_t_peak = t_waves[0].peak_index
        if len(t_waves) > 1 and t_waves[1].peak_index < left_most_t_peak:
            left_most_t_peak = t_waves[1].peak_index

        p_start = np.asarray((0, t_segment_norm[0]))
        p_end = np.asarray((left_most_t_peak, t_segment_norm[left_most_t_peak]))

        max_dist = 0
        t_onset = -1
        for i in range(1, left_most_t_peak):
            point = np.asarray((i, t_segment_norm[i]))
            distance = np.linalg.norm(np.cross(p_end - p_start, p_start - point)) / np.linalg.norm(p_end - p_start)
            if distance > max_dist:
                max_dist = distance
                t_onset = i

        if t_onset == -1:
            plt.plot(t_segment_orig)
            plt.show()
            raise AssertionError('could not identify T onset.')

        t_width = len(t_segment_orig) - t_onset + 1
        t_waves[0].width = t_width

        x = np.array(list(range(t_onset)))
        y = t_segment_orig[x]
        st_line = linregress(x=x, y=y)
        return t_waves, st_line, t_onset


class QRSComplex:
    def __init__(self, segment: [float], frequency: int):
        self.segment = segment
        self.frequency = frequency
        self.q = -1
        self.r = -1
        self.s = -1
        self.onset = -1
        self.offset = -1
        self.notches = []

    def _get_last_wave_peak(self) -> int:
        if self.s != -1:
            return self.s
        if self.r != -1:
            return self.r
        return self.q

    def get_q_amp(self):
        if self.q == -1:
            return 0
        return self.segment[self.q]

    def get_r_amp(self):
        if self.r == -1:
            return 0
        return self.segment[self.r]

    def get_s_amp(self):
        if self.s == -1:
            return 0
        return self.segment[self.s]

    def get_energy(self) -> float:
        # Calculate QRS energy: https://matel.p.lodz.pl/wee/i12zet/Signal%20energy%20and%20power.pdf
        dx = 250 / self.frequency
        energy = np.trapz(y=[abs(y) ** 2 for y in self.segment], dx=dx)
        return energy

    def get_auc(self) -> float:
        # Area above X=0 is positive, while the area under in negative.
        dx = 250 / self.frequency
        return np.trapz(y=self.segment, dx=dx)

    def get_rs_slope(self):
        if self.r == -1:
            return 0

        if self.s == -1:
            delta_y = self.segment[-1] - self.segment[self.r]
            delta_x = (len(self.segment) - self.r) / self.frequency  # delta_x to seconds to handle different frequency.
        else:
            delta_y = self.segment[self.s] - self.segment[self.r]
            delta_x = (self.s - self.r) / self.frequency  # delta_x to seconds to deal with different sample bases.

        return delta_y / delta_x

    def get_q_slope(self):
        if self.q == -1:
            return 0

        delta_y = self.segment[self.q] - self.segment[0]
        delta_x = (self.q - 0) / self.frequency

        return delta_y / delta_x

    def get_r_upstroke_slope(self):
        if self.r == -1:
            return 0

        if self.q == -1:
            delta_y = self.segment[self.r] - self.segment[0]
            delta_x = (self.r - 0) / self.frequency
        else:
            delta_y = self.segment[self.r] - self.segment[self.q]
            delta_x = (self.r - self.q) / self.frequency

        return delta_y / delta_x

    def get_duration(self):
        return len(self.segment) / self.frequency

    def get_non_terminal_duration(self):
        end_point = self._get_last_wave_peak()
        return (end_point + 1) / self.frequency

    def get_terminal_duration(self):
        end_point = self._get_last_wave_peak()
        return (len(self.segment) - end_point) / self.frequency

    def get_number_notches(self):
        return len(self.notches)

    def get_number_notches_terminal(self):
        return len([notch for notch in self.notches if notch.peak_index > self._get_last_wave_peak()])

    def get_notch_max_prominence(self):
        if len(self.notches) == 0:
            return 0
        return max(self.notches, key=lambda x: x.prominence).prominence


class Twave:
    def __init__(self, segment: [float], frequency: int):
        self.segment = segment
        self.frequency = frequency
        self.peak = None
        self.onset = -1
        self.offset = -1

    def get_peak_amp(self) -> float:
        return self.segment[self.peak]

    def get_duration(self) -> float:
        return len(self.segment) / self.frequency

    def get_terminal_duration(self) -> float:
        return len(self.segment[self.peak:]) / self.frequency

    def get_energy(self) -> float:
        # Calculate QRS energy: https://matel.p.lodz.pl/wee/i12zet/Signal%20energy%20and%20power.pdf
        dx = 250 / self.frequency
        energy = np.trapz(y=[abs(y) ** 2 for y in self.segment], dx=dx)
        return energy

    def get_auc(self) -> float:
        # Area above X=0 is positive, while the area under in negative.
        dx = 250 / self.frequency
        return np.trapz(y=self.segment, dx=dx)


class TPSegment:
    def __init__(self):
        self.onset = -1
        self.offset = -1


class STsegment:
    def __init__(self, segment: [float], baseline_amp: float, frequency: int):
        self.segment = segment
        self.baseline_amp = baseline_amp
        self.frequency = frequency
        self.onset = -1
        self.offset = -1

    def get_slope(self) -> float:
        st_line = linregress(x=list(range(len(self.segment))), y=self.segment)
        return st_line.slope

    def get_onset_elevation(self):
        # Consider a small interval of 8ms after QRS offset to calculate the amp of QRS offset.
        wlen = round(self.frequency * 0.008)
        return statistics.mean(self.segment[:wlen]) - self.baseline_amp

    def get_offset_elevation(self):
        # Consider a small interval of 8ms around T onset to calculate the amp of T onset.
        wlen = round(self.frequency * 0.008)
        return statistics.mean(self.segment[-wlen:]) - self.baseline_amp


class Baseline:
    def __init__(self, amp, std, slope, prominence: float, segments: [[float]]):
        self.segments = segments
        self.amp = amp
        self.std = std
        self.slope = slope
        self.prominence = prominence


def plot_ecg(ecg: pd.DataFrame, pid: int = None, ecg_id: int = None, feature_vector: pd.DataFrame = None):
    fig, ax = plt.subplots(12, figsize=(14, 15))
    for i in range(12):
        ax[i].plot(ecg.values[:, i])
        ax[i].set_title(Util.get_lead_name(i))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

        if feature_vector is not None:
            lead_name = Util.get_lead_name(i)
            title = f'Q={round(feature_vector[f"({lead_name})_QRS_q"].values[0], 2)} | ' \
                    f'R={round(feature_vector[f"({lead_name})_QRS_r"].values[0], 2)} | ' \
                    f'S={round(feature_vector[f"({lead_name})_QRS_s"].values[0], 2)} | ' \
                    f'RS={round(feature_vector[f"({lead_name})_QRS_rs_slope"].values[0], 2)} | ' \
                    f'Notches={round(feature_vector[f"({lead_name})_QRS_notches"].values[0], 2)} | ' \
                    f'STon={round(feature_vector[f"({lead_name})_ST_on_elevation"].values[0], 2)} | ' \
                    f'STend={round(feature_vector[f"({lead_name})_ST_off_elevation"].values[0], 2)} | ' \
                    f'T={round(feature_vector[f"({lead_name})_T_amp"].values[0], 2)} | ' \
                    f'B={round(feature_vector[f"({lead_name})_Base_slope"].values[0], 2)}'
            ax[i].set_title(title)

    if pid is not None and ecg_id is not None:
        fig.suptitle(f'PID={pid} ECGID={ecg_id}')
    plt.show()


def plot_qt_segments(segments: [np.ndarray], pid: int = None, ecg_id: int = None, x_1: [int] = None, x_2: [int] = None):
    fig, ax = plt.subplots(nrows=12, ncols=10, figsize=(12, 15))
    n = min([10, len(segments)])
    for i in range(12):
        for j in range(n):
            qt_segment = segments[j][i]
            ax[i][j].plot(qt_segment)
            if x_1 is not None:
                ax[i][j].axvline(x=x_1[j], color='red')
            if x_2 is not None:
                ax[i][j].axvline(x=x_2[j], color='blue')
            if j == 0:
                ax[i][j].set_title(Util.get_lead_name(i))
    for i in range(12):
        for j in range(10):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['bottom'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)

    if pid is not None and ecg_id is not None:
        fig.suptitle(f'PID={pid} ECGID={ecg_id}')
    plt.show()


def get_ecg_scar_dataset() -> pd.DataFrame:
    mri_loc_df = pd.read_excel(GlobalPaths.scar_location)
    mri_meta_df = pd.read_excel(GlobalPaths.mri)
    mri_meta_df = mri_meta_df[['Record_ID', 'Scar tissue %']]
    mri_loc_df = mri_loc_df[['Record_ID', 'MRI Date'] + [col for col in mri_loc_df.columns if
                                                         'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]]
    dataset = pd.merge(left=mri_meta_df, right=mri_loc_df, how='inner', on=['Record_ID'])
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # Clean MRI Date.
    need_correction = 0
    for index, row in dataset.iterrows():
        try:
            dataset.iat[index, 2] = pd.to_datetime(row['MRI Date'])
        except:
            date_str = str(row['MRI Date'])
            if '.' in date_str:
                date_str = date_str.replace('.', '')
            if ',' in date_str:
                mri_date = pd.to_datetime(date_str.split(',')[0])
            else:
                mri_date = pd.to_datetime(date_str.split(' ')[0])
            need_correction += 1
            dataset.iat[index, 2] = mri_date
    dataset['MRI Date'] = pd.to_datetime(dataset['MRI Date'])

    ecg_df = pd.read_excel('Data/overall_ecg_feature_uncertain_scar_location.xlsx')

    dataset = pd.merge(left=ecg_df, right=dataset[['Record_ID', 'Scar tissue %']], how='inner', on=['Record_ID'])
    dataset.reset_index(drop=True, inplace=True)
    mri_diff_days = []
    entire_lv_scar = []
    for index, row in dataset.iterrows():
        ecg_date = pd.to_datetime(row['ECG Date'])
        mri_date = pd.to_datetime(row['MRI Date'])
        diff = ecg_date - mri_date
        mri_diff_days.append(diff.days)
        entire_lv_scar.append(sum(row[['Basal', 'Mid', 'Apical', 'Apex']].values))

    dataset['MRI Diff'] = mri_diff_days
    dataset['LV Scar'] = entire_lv_scar
    n_patients = len(dataset['Record_ID'].unique())
    return dataset


def average(data: list) -> float:
    if len(data) == 1:
        return data[0]
    return statistics.mean(data)


def identify_baseline(ecg_dict: dict):
    baseline_notches = []
    baseline_segments = []
    qt_onsets_offsets = [x[0] for x in ecg_dict['qt_onsets_offsets']]

    # Step 1: Extract `baseline_segments` as segments between T_end and next QRS_on in each lead .
    # For each segment, identify all positive waves as `baseline_notches`.
    for i in range(len(qt_onsets_offsets) - 1):
        t_offset = qt_onsets_offsets[i][1]
        next_qrs_onset = qt_onsets_offsets[i + 1][0]

        beat_baseline_notches = []
        beat_baseline_segment = []
        for lead in range(12):
            non_qt_segment = ecg_dict['ecg_denoised'].iloc[t_offset:next_qrs_onset + 1, lead].values
            baseline_segment = non_qt_segment[:len(non_qt_segment) // 2]
            beat_baseline_segment.append(baseline_segment)
            lead_baseline_notches = ECGUtility.get_positive_waves(segment=baseline_segment, min_prominence=0, global_max=-1)
            beat_baseline_notches.append(lead_baseline_notches)

        baseline_segments.append(beat_baseline_segment)
        baseline_notches.append(beat_baseline_notches)

    # Step 2: In 12-D lists, extract following information about the baseline:
    #   `baseline_prominence`: average of 10 highest baseline notches.
    #   `baseline_amp_mean`: average amp of all baseline segments in a lead.
    #   `baseline_amp_std`: std amp of all baseline segments in a lead.
    #   `baseline_slope`: average slope of fitted lines on all baseline segments in a lead.
    baseline_prominence = []
    for lead in range(12):
        lead_notches = []
        for notch in baseline_notches:
            lead_notches += [round(wave.prominence, 1) for wave in notch[lead]]
        baseline_prominence.append(sorted(lead_notches))

    for lead in range(12):
        lead_base_proms = baseline_prominence[lead]
        if len(lead_base_proms) >= 10:
            base_prom = statistics.mean(heapq.nlargest(n=10, iterable=lead_base_proms))
        elif len(lead_base_proms) > 0:
            base_prom = statistics.mean(lead_base_proms)
        else:
            base_prom = 0
        baseline_prominence[lead] = base_prom

    baseline = []
    for lead in range(12):
        lead_base_segments = []
        for base_segments in baseline_segments:
            lead_base_segments.append(list(base_segments[lead]))
        baseline.append(lead_base_segments)

    baseline_amp_mean = []
    baseline_amp_std = []
    baseline_slope = []
    for lead in range(12):
        amp_mean = statistics.mean([statistics.mean(lead_base) for lead_base in baseline[lead]])
        amp_std = statistics.stdev([statistics.mean(lead_base) for lead_base in baseline[lead]])
        slope = statistics.mean(
            [linregress(x=np.array(range(len(lead_base))), y=lead_base).slope for lead_base in baseline[lead]])
        baseline_amp_mean.append(amp_mean)
        baseline_amp_std.append(amp_std)
        baseline_slope.append(slope)

    # Step 3: Make `Baseline` objects and return a 12-D list.
    baseline_objects = []
    for i in range(12):
        base = Baseline(segments=baseline[i],
                        amp=baseline_amp_mean[i],
                        std=baseline_amp_std[i],
                        slope=baseline_slope[i],
                        prominence=baseline_prominence[i])
        baseline_objects.append(base)
    return baseline_objects


def identify_qrs(ecg_dict: dict, voted_qrs_offsets: [int], baseline_prominence: [float], frequency: int) -> np.ndarray:
    if len(baseline_prominence) != 12:
        raise ValueError('`baseline_prominence` must be 12-D')

    result_qrs = []
    for i, qrs_offset in enumerate(voted_qrs_offsets):
        if qrs_offset == 0:
            continue
        one_qrs_among_12_leads = []
        qrs_all_leads = ecg_dict['segments'][i][:, :qrs_offset]
        for lead, qrs_segment in enumerate(qrs_all_leads):

            gmax = ECGUtility.find_global_extremum(qrs_segment)
            if gmax is None:
                raise ValueError(f'no global extremum could be found in qrs segment at lead {lead}')

            qrs_base_amp = statistics.mean([qrs_segment[0], qrs_segment[-1]])
            positive_waves = ECGUtility.get_positive_waves(qrs_segment, min_prominence=0, global_max=gmax.peak_index)

            QRS = QRSComplex(segment=qrs_segment, frequency=frequency)

            # Step 1: Identify R-wave
            if qrs_segment[gmax.peak_index] > 0:
                QRS.r = gmax.peak_index
            else:
                if len(positive_waves) == 0 or max(positive_waves, key=lambda x: x.amp).amp <= qrs_base_amp:
                    QRS.q = gmax.peak_index
                else:
                    QRS.r = max(positive_waves, key=lambda x: x.amp).peak_index
                    if gmax.peak_index < QRS.r:
                        QRS.q = gmax.peak_index
                    else:
                        QRS.s = gmax.peak_index

            # Step 2: If R-wave is not present, Step 1 has identified Q and S waves. But, if R-wave is present, Step 1
            # has not necessarily identified Q and S waves.
            if QRS.r != -1:
                qs = ECGUtility.identify_qs(qrs_segment=qrs_segment, r_index=QRS.r, base_amp=qrs_base_amp, min_prom=baseline_prominence[lead])
                QRS.q = qs['Q']
                QRS.s = qs['S']

            # Step 3: Notches
            for notch in positive_waves:
                if notch.peak_index != QRS.r:

                    closest_qrs_peak = -1
                    min_dist_qrs_peak = 5000
                    if QRS.q != -1 and abs(QRS.q - notch.peak_index) < min_dist_qrs_peak:
                        closest_qrs_peak = QRS.q
                        min_dist_qrs_peak = abs(QRS.q - notch.peak_index)
                        if QRS.r != -1 and abs(qrs_segment[QRS.r] > abs(qrs_segment[QRS.q])):
                            ratio = 0.8
                        else:
                            ratio = 1 / 3
                    if QRS.r != -1 and abs(QRS.r - notch.peak_index) < min_dist_qrs_peak:
                        closest_qrs_peak = QRS.r
                        min_dist_qrs_peak = abs(QRS.r - notch.peak_index)
                        ratio = 1 / 3
                    if QRS.s != -1 and abs(QRS.s - notch.peak_index) < min_dist_qrs_peak:
                        closest_qrs_peak = QRS.s
                        ratio = 1 / 3

                    closest_peak_deviation = abs(qrs_segment[closest_qrs_peak] - qrs_base_amp)
                    notch_deviation = abs(qrs_segment[notch.peak_index] - qrs_base_amp)

                    if notch.prominence > baseline_prominence[lead] or notch_deviation > closest_peak_deviation * ratio:
                        QRS.notches.append(notch)

            one_qrs_among_12_leads.append(QRS)

        one_qrs_among_12_leads = np.array(one_qrs_among_12_leads)
        result_qrs.append(one_qrs_among_12_leads)

    return np.array(result_qrs).transpose()


def identify_st(ecg_dict: dict, voted_qrs_offsets: [int], voted_t_onsets: [int], baseline_amps: [float], frequency: int) -> np.ndarray:
    if len(voted_qrs_offsets) != len(voted_t_onsets):
        raise ValueError('voted_qrs_offsets len is not equal to voted_t_onsets')

    result_st = []
    for i, (qrs_offset, t_onset) in enumerate(zip(voted_qrs_offsets, voted_t_onsets)):
        if qrs_offset == 0:
            continue
        st_all_leads = ecg_dict['segments'][i][:, qrs_offset: t_onset+1]
        one_st_among_12_leads = []
        for lead, st_segment in enumerate(st_all_leads):
            ST = STsegment(segment=st_segment, baseline_amp=baseline_amps[lead], frequency=frequency)
            one_st_among_12_leads.append(ST)

        result_st.append(np.array(one_st_among_12_leads))

    return np.array(result_st).transpose()


def identify_t(ecg_dict: dict, voted_t_onsets: [int], frequency: int) -> np.ndarray:

    result_t = []
    for i, t_onset in enumerate(voted_t_onsets):
        if t_onset == 0:
            continue
        t_all_leads = ecg_dict['segments'][i][:, t_onset:]
        one_t_among_12_leads = []
        for lead, t_segment in enumerate(t_all_leads):
            gmax = ECGUtility.find_global_extremum(t_segment)
            if gmax is None:
                raise ValueError(f'no global extremum could be found in qrs segment at lead {lead}')
            T = Twave(segment=t_segment, frequency=frequency)
            T.peak = ECGUtility.find_global_extremum(t_segment).peak_index
            one_t_among_12_leads.append(T)
        result_t.append(np.array(one_t_among_12_leads))

    result_t = np.array(result_t).transpose()

    voted_t_peaks = [0] * len(result_t[0])
    k = 0
    for i in range(len(result_t[0])):
        t_among_12_leads = result_t[:, i]
        peaks = [t.peak for t in t_among_12_leads]

        # Vote among 12 t peaks -> threshold for vicinity = 8ms. Any two T peaks farther than 8ms from
        # each other are not in each other's consensus set.
        threshold = round(frequency * 0.008)
        consensus = []
        for i in range(len(peaks)):
            c = 0
            onset_current = peaks[i]
            for j in range(len(peaks)):
                if i != j:
                    onset_neighbor = peaks[j]
                    if abs(onset_current - onset_neighbor) <= threshold:
                        c += 1
            consensus.append(c)

        # Final T peak is the weighted average of all lead T peaks whose consensus has >= 7 members out of 12.
        voted_peaks = []
        weights = []
        for i in range(len(consensus)):
            if consensus[i] >= 7:
                weights.append(consensus[i])
                voted_peaks.append(peaks[i])
        if len(voted_peaks) > 1:
            t_peak = round(np.average(voted_peaks, weights=weights))
        else:
            t_peak = round(np.average(peaks[-4:])) + 1
        voted_t_peaks[k] = t_peak
        k += 1

    for i in range(len(result_t[0])):
        for t in result_t[:, i]:
            t.peak = voted_t_peaks[i]

    return result_t


def extract_feature_qrs(result_qrs: np.ndarray):
    result = {}
    for lead, qrs_per_lead in enumerate(result_qrs):
        vote_q = sum([1 if qrs.q != -1 else 0 for qrs in qrs_per_lead])
        vote_r = sum([1 if qrs.r != -1 else 0 for qrs in qrs_per_lead])
        vote_s = sum([1 if qrs.s != -1 else 0 for qrs in qrs_per_lead])

        voted_beats = [0] * len(qrs_per_lead)
        half_population = len(qrs_per_lead) // 2

        q_is_50_50 = False
        if vote_q >= half_population + 1:
            majority = 1
        elif vote_q <= half_population - 1:
            majority = 0
        else:
            # In this case, it is obvious that Q is very small. It's safe saying that Q is not present.
            #raise ValueError(f'Q-wave is 50-50 detected at lead {lead}')
            q_is_50_50 = True
        for i in range(len(qrs_per_lead)):
            if q_is_50_50 or (qrs_per_lead[i].q != -1 and majority == 1) or (qrs_per_lead[i].q == -1 and majority == 0):
                voted_beats[i] += 1

        r_is_50_50 = False
        if vote_r >= half_population + 1:
            majority = 1
        elif vote_r <= half_population - 1:
            majority = 0
        else:
            # raise ValueError(f'R-wave is 50-50 detected at lead {lead}')
            r_is_50_50 = True
        for i in range(len(qrs_per_lead)):
            if r_is_50_50 or (qrs_per_lead[i].r != -1 and majority == 1) or (qrs_per_lead[i].r == -1 and majority == 0):
                voted_beats[i] += 1

        s_is_50_50 = False
        if vote_s >= half_population + 1:
            majority = 1
        elif vote_s <= half_population - 1:
            majority = 0
        else:
            # raise ValueError(f'S-wave is 50-50 detected at lead {lead}')
            s_is_50_50 = True
        for i in range(len(qrs_per_lead)):
            if s_is_50_50 or (qrs_per_lead[i].s != -1 and majority == 1) or (qrs_per_lead[i].s == -1 and majority == 0):
                voted_beats[i] += 1

        selected_qrs_complexes = []
        for i, vote in enumerate(voted_beats):
            if vote == 3:
                selected_qrs_complexes.append(qrs_per_lead[i])

        if len(selected_qrs_complexes) == 0:
            raise ValueError(f'no QRS complex could be selected after voting in lead {lead}')

        lead_name = Util.get_lead_name(lead)

        if q_is_50_50:
            result[f'({lead_name})_QRS_q'] = 0
        else:
            result[f'({lead_name})_QRS_q'] = average([qrs.get_q_amp() for qrs in selected_qrs_complexes])

        if r_is_50_50:
            result[f'({lead_name})_QRS_r'] = 0
        else:
            result[f'({lead_name})_QRS_r'] = average([qrs.get_r_amp() for qrs in selected_qrs_complexes])

        if s_is_50_50:
            result[f'({lead_name})_QRS_s'] = 0
        else:
            result[f'({lead_name})_QRS_s'] = average([qrs.get_s_amp() for qrs in selected_qrs_complexes])

        result[f'({lead_name})_QRS_energy'] = average([qrs.get_energy() for qrs in selected_qrs_complexes])
        result[f'({lead_name})_QRS_auc'] = average([qrs.get_auc() for qrs in selected_qrs_complexes])

        result[f'({lead_name})_QRS_rs_slope'] = average([qrs.get_rs_slope() for qrs in selected_qrs_complexes])
        result[f'({lead_name})_QRS_q_slope'] = average([qrs.get_q_slope() for qrs in selected_qrs_complexes])
        result[f'({lead_name})_QRS_r_upstroke_slope'] = average([qrs.get_r_upstroke_slope() for qrs in selected_qrs_complexes])

        result[f'({lead_name})_QRS_duration'] = average([qrs.get_duration() for qrs in selected_qrs_complexes])
        result[f'({lead_name})_QRS_duration_non_terminal'] = average([qrs.get_non_terminal_duration() for qrs in selected_qrs_complexes])
        result[f'({lead_name})_QRS_duration_terminal'] = average([qrs.get_terminal_duration() for qrs in selected_qrs_complexes])

        voted_number_notch = Counter([qrs.get_number_notches() for qrs in selected_qrs_complexes]).most_common(1)
        if voted_number_notch[0][0] == 0:
            result[f'({lead_name})_QRS_notches'] = 0
            result[f'({lead_name})_QRS_notches_terminal'] = 0
            result[f'({lead_name})_QRS_notches_prom'] = 0
        else:
            result[f'({lead_name})_QRS_notches'] = voted_number_notch[0][0]
            result[f'({lead_name})_QRS_notches_terminal'] = Counter([qrs.get_number_notches_terminal() for qrs in selected_qrs_complexes if qrs.get_number_notches() == voted_number_notch[0][0]]).most_common(1)[0][0]
            result[f'({lead_name})_QRS_notches_prom'] = average([qrs.get_notch_max_prominence() for qrs in selected_qrs_complexes if qrs.get_number_notches() == voted_number_notch[0][0]])

    return pd.DataFrame(result, index=[0])


def extract_feature_t(result_t: np.ndarray) -> pd.DataFrame:
    result = {}
    for lead, t_per_lead in enumerate(result_t):
        lead_name = Util.get_lead_name(lead)

        result[f'({lead_name})_T_amp'] = average([t.get_peak_amp() for t in t_per_lead])
        result[f'({lead_name})_T_duration'] = average([t.get_duration() for t in t_per_lead])
        result[f'({lead_name})_T_duration_terminal'] = average([t.get_terminal_duration() for t in t_per_lead])
        result[f'({lead_name})_T_auc'] = average([t.get_auc() for t in t_per_lead])
        result[f'({lead_name})_T_energy'] = average([t.get_energy() for t in t_per_lead])

    return pd.DataFrame(result, index=[0])


def extract_feature_st(result_st: np.ndarray) -> pd.DataFrame:
    result = {}
    for lead, st_per_lead in enumerate(result_st):
        lead_name = Util.get_lead_name(lead)

        slopes = [st.get_slope() for st in st_per_lead]
        onset_elevations = [st.get_onset_elevation() for st in st_per_lead]
        offset_elevations = [st.get_offset_elevation() for st in st_per_lead]

        if len([x for x in slopes if x >= 0]) > len([x for x in slopes if x < 0]):
            slopes = [x for x in slopes if x >= 0]
        else:
            slopes = [x for x in slopes if x < 0]

        if len([x for x in onset_elevations if x >= 0]) > len([x for x in onset_elevations if x < 0]):
            onset_elevations = [x for x in onset_elevations if x >= 0]
        else:
            onset_elevations = [x for x in onset_elevations if x < 0]

        if len([x for x in offset_elevations if x >= 0]) > len([x for x in offset_elevations if x < 0]):
            offset_elevations = [x for x in offset_elevations if x >= 0]
        else:
            offset_elevations = [x for x in offset_elevations if x < 0]

        result[f'({lead_name})_ST_slope'] = average(slopes)
        result[f'({lead_name})_ST_on_elevation'] = average(onset_elevations)
        result[f'({lead_name})_ST_off_elevation'] = average(offset_elevations)

    return pd.DataFrame(result, index=[0])


def vote_qrs_offset(ecg_dict: dict, frequency: int, base_proms, base_amps: [float]):
    voted_qrs_offsets = [0] * len(ecg_dict['segments'])
    k = 0  # Index for `voted_qrs_offsets` array.

    for qt_segment_12_lead in ecg_dict['segments']:
        candidate_qrs_offsets = []
        for lead in range(12):
            qt_segment = qt_segment_12_lead[lead, :]
            base_prom = base_proms[lead]
            base_amp = base_amps[lead]
            try:
                qrs_offset = ECGUtility.identify_qrs_offset(qt_segment=qt_segment, base_prom=base_prom, base_amp=base_amp)
            except:
                continue
            candidate_qrs_offsets.append(qrs_offset)

        # Vote among 12 qrs offsets -> threshold for vicinity = 20ms. Any two QRS offsets farther than 20ms from
        # each other are not in each other's consensus set.
        threshold = round(frequency / 50)
        consensus = []
        for i in range(len(candidate_qrs_offsets)):
            c = 0
            offset_current = candidate_qrs_offsets[i]
            for j in range(len(candidate_qrs_offsets)):
                if i != j:
                    offset_neighbor = candidate_qrs_offsets[j]
                    if abs(offset_current - offset_neighbor) <= threshold:
                        c += 1
            consensus.append(c)

        # Final QRS offset is the weighted average of all lead QRS offsets whose consensus is more than 4.
        offsets = []
        weights = []
        for i in range(len(consensus)):
            if consensus[i] >= 4:
                weights.append(consensus[i])
                offsets.append(candidate_qrs_offsets[i])
        if len(weights) == 0:
            k += 1
            continue
            # raise ValueError(f'Consensus could not be reached for QRS offset (all < 4)')
        qrs_offset = round(np.average(offsets, weights=weights)) + 1
        voted_qrs_offsets[k] = qrs_offset
        k += 1
    if sum(voted_qrs_offsets) == 0:
        raise ValueError('no QRS offset could be identified after voting')
    return voted_qrs_offsets


def vote_t_onset(ecg_dict: dict, frequency: int, voted_qrs_offsets: [int]):
    # Part 3 -> Given QRS offsets, identify T onsets in QT segments by voting among 12 leads.
    voted_t_onsets = [0] * len(voted_qrs_offsets)
    k = 0  # Index for `voted_t_onsets` array.
    for qrs_offset, qt_segment_12_lead in zip(voted_qrs_offsets, ecg_dict['segments']):
        if qrs_offset == 0:
            k += 1
            continue
        candidate_t_onsets = []
        for lead in range(12):
            qt_segment = qt_segment_12_lead[lead, :]
            try:
                _, _, t_onset = ECGUtility.parse_t_wave(qt_segment, qrs_offset=qrs_offset)
            except AssertionError:
                continue
            candidate_t_onsets.append(t_onset)

        # Vote among 12 t onsets -> threshold for vicinity = 20ms. Any two T onsets farther than 20ms from
        # each other are not in each other's consensus set.
        threshold = round(frequency / 50)
        consensus = []
        for i in range(len(candidate_t_onsets)):
            c = 0
            onset_current = candidate_t_onsets[i]
            for j in range(len(candidate_t_onsets)):
                if i != j:
                    onset_neighbor = candidate_t_onsets[j]
                    if abs(onset_current - onset_neighbor) <= threshold:
                        c += 1
            consensus.append(c)
        # Final T onset is the weighted average of all lead T onsets whose consensus is more than 2.
        onsets = []
        weights = []
        for i in range(len(consensus)):
            if consensus[i] >= 3:
                weights.append(consensus[i])
                onsets.append(candidate_t_onsets[i])
        if len(onsets) > 1:
            t_onset = round(np.average(onsets, weights=weights)) + 1
        else:
            t_onset = round(np.average(candidate_t_onsets[-4:])) + 1
        voted_t_onsets[k] = t_onset
        k += 1
    voted_t_onsets = [a + b for a, b in zip(voted_t_onsets, voted_qrs_offsets)]
    return voted_t_onsets


def get_scar_dataset():
    mri_loc_df = pd.read_excel(GlobalPaths.scar_location)
    mri_meta_df = pd.read_excel(GlobalPaths.mri)
    mri_meta_df = mri_meta_df[['Record_ID', 'Scar tissue %']]
    mri_loc_df = mri_loc_df[['Record_ID', 'MRI Date'] + [col for col in mri_loc_df.columns if
                                                         'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]]
    dataset = pd.merge(left=mri_meta_df, right=mri_loc_df, how='inner', on=['Record_ID'])
    dataset.dropna(inplace=True)

    rv_insertion_pids = set(pd.read_excel(GlobalPaths.rv_insertion)['Record_ID'].values)
    dataset = dataset.loc[~dataset['Record_ID'].isin(rv_insertion_pids)]
    dataset.reset_index(drop=True, inplace=True)

    # Clean MRI Date.
    need_correction = 0
    for index, row in dataset.iterrows():
        try:
            dataset.iat[index, 2] = pd.to_datetime(row['MRI Date'])
        except:
            date_str = str(row['MRI Date'])
            if '.' in date_str:
                date_str = date_str.replace('.', '')
            if ',' in date_str:
                mri_date = pd.to_datetime(date_str.split(',')[0])
            else:
                mri_date = pd.to_datetime(date_str.split(' ')[0])
            need_correction += 1
            dataset.iat[index, 2] = mri_date
    dataset['MRI Date'] = pd.to_datetime(dataset['MRI Date'])
    return dataset


def run_feature_extraction():
    ecg_df = pd.read_excel(GlobalPaths.ecg_meta_loc)
    scar_df = get_scar_dataset()
    extract_dict = QTSegmentExtractor(ecg_dir_path=GlobalPaths.ecg_loc, ann_dir_path=GlobalPaths.pla_annotation_loc,
                                      metadata_path=GlobalPaths.ecg_meta_loc, verbose=True).extract_segments()
    pids_with_ecg = set(extract_dict.keys())

    scar_df = scar_df.loc[scar_df['Record_ID'].isin(pids_with_ecg)].reset_index(drop=True, inplace=False)

    ecg_skipped = {}
    dataset = pd.DataFrame()
    for _, scar in enumerate(pbar := tqdm(scar_df.iterrows(), total=scar_df.shape[0])):
        scar = scar[1]
        ecgs = extract_dict[scar['Record_ID']]
        pbar.set_description(f'Extract features from {len(ecgs)} ECGs for PID = {scar["Record_ID"]}')
        for ecg_dict in ecgs:
            if len(ecg_dict['segments']) < 3:
                ecg_skipped[ecg_dict['ecg_id']] = '< 3 segments were identified'
                continue
            frequency = ecg_dict['frequency']

            baselines = identify_baseline(ecg_dict=ecg_dict)
            base_proms = [base.prominence for base in baselines]
            base_amps = [base.amp for base in baselines]

            try:
                qrs_offsets = vote_qrs_offset(ecg_dict=ecg_dict, frequency=frequency, base_proms=base_proms, base_amps=base_amps)
                t_onsets = vote_t_onset(ecg_dict=ecg_dict, frequency=frequency, voted_qrs_offsets=qrs_offsets)

                all_qrs = identify_qrs(ecg_dict=ecg_dict, voted_qrs_offsets=qrs_offsets, baseline_prominence=base_proms, frequency=frequency)
                all_t = identify_t(ecg_dict=ecg_dict, voted_t_onsets=t_onsets, frequency=frequency)
                all_st = identify_st(ecg_dict=ecg_dict, voted_qrs_offsets=qrs_offsets, voted_t_onsets=t_onsets, baseline_amps=base_amps, frequency=frequency)

                qrs_features = extract_feature_qrs(all_qrs)
                t_features = extract_feature_t(all_t)
                st_features = extract_feature_st(all_st)

            except ValueError as e:
                ecg_skipped[ecg_dict['ecg_id']] = e
                continue

            # The following two features measure how variable the heartbeat interval is across a 10-second ECG.
            heartrate_variability = {'(all)_qt_dist_mean': statistics.mean(ecg_dict['qt_distances']) * (1 / frequency),
                                     '(all)_qt_dist_std': statistics.stdev(ecg_dict['qt_distances']) * (1 / frequency)}
            heartrate_variability = pd.DataFrame(data=heartrate_variability, index=[0])

            baseline_features = {}
            for lead in range(12):
                lead_name = Util.get_lead_name(lead)
                baseline_features[f'({lead_name})_Base_slope'] = baselines[lead].slope
            baseline_features = pd.DataFrame(data=baseline_features, index=[0])

            ground_truth = scar.to_frame().transpose().reset_index().drop(columns=['index'])
            ground_truth['ECG Date'] = ecg_df.loc[ecg_df['ECG_ID'] == ecg_dict['ecg_id']]['ECG Date'].values[0]
            ground_truth['ECG_ID'] = ecg_dict['ecg_id']
            feature_vector = pd.concat([ground_truth, heartrate_variability, qrs_features, st_features, t_features, baseline_features], axis=1)
            dataset = pd.concat([dataset, feature_vector], ignore_index=True, axis=0)
    return dataset


if __name__ == '__main__':
    dataset = run_feature_extraction()
    dataset['MRI_ECG_Diff'] = abs(dataset['MRI Date'] - dataset['ECG Date'])
    dataset['Scar'] = dataset[[col for col in dataset.columns if 'Basal' in col or 'Mid' in col or 'Apical' in col or 'Apex' in col]].sum(axis=1).values
    dataset['Basal'] = dataset[[col for col in dataset.columns if 'Basal' in col]].sum(axis=1).values
    dataset['Mid'] = dataset[[col for col in dataset.columns if 'Mid' in col]].sum(axis=1).values
    dataset['Apical'] = dataset[[col for col in dataset.columns if 'Apical' in col]].sum(axis=1).values
    dataset.to_excel('datasetV2.xlsx', index=False)

