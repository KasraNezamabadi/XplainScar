import os
import statistics
import GlobalPaths
import pandas as pd
from tqdm import tqdm
import xmltodict
from QTSegmentExtractor import QTSegmentExtractor
from FeatureExtraction import identify_baseline, vote_qrs_offset, vote_t_onset, identify_qrs, identify_t, identify_st,\
    extract_feature_qrs, extract_feature_t, extract_feature_st
from Utility import Util
from sklearn.preprocessing import StandardScaler
import math
from sklearn import mixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind

from scarf.loss import NTXent
from scarf.model import SCARF

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from scarf_utils.dataset import ScarfDataset
from scarf_utils.utils import dataset_embeddings, fix_seed, train_epoch

import shap
import numpy as np
import seaborn as sns
import scipy.stats as stats


def prepare_dataset(name: str, mri_ecg_days: int):
    """
    This function reads muse xml ECG files, keeps those ECGs taken within `mri_ecg_days` days from MRI, and extract ECG
    features from QRS complexes, T-waves, ST segments, and TP segments. The final dataset is saved as 'name'.xlsx
    :param name: dataset name to save on disk.
    :param mri_ecg_days: Maximum distance in days between MRI and ECG dates for an ECG to be considered in the dataset.
    :return: None (the dataset is saved on disk.)
    """
    ecg_scar_df = pd.read_excel('Data/ECG/recruitment_ecg_scar_location_after_myectomy_filter.xlsx')
    for region_name in ['Basal', 'Mid', 'Apical']:
        sum_ann = ecg_scar_df[[col for col in ecg_scar_df.columns if region_name in col]].sum(axis=1).values
        ecg_scar_df[region_name] = sum_ann
    ecg_scar_df['Scar'] = ecg_scar_df[['Basal', 'Mid', 'Apical']].sum(axis=1).values

    # Get ECG frequencies
    freq = []
    for _, row in ecg_scar_df.iterrows():
        ecg_id = int(row['ECG_ID'])
        xml_name = row['MUSE Name']
        try:
            frequency = int(xmltodict.parse(open(os.path.join(GlobalPaths.muse, xml_name), 'rb').read().decode('utf8'))['RestingECG']['Waveform'][1]['SampleBase'])
            freq.append([ecg_id, frequency])
        except KeyError:
            continue
    freq_df = pd.DataFrame(freq, columns=['ECG_ID', 'Frequency'])
    ecg_scar_df = ecg_scar_df.merge(freq_df, on='ECG_ID')

    # Get MRI parameters
    mri = pd.read_excel('Data/HCM_MRI_Kasra.xlsx')[['Record_ID', 'Scar tissue %', 'LV_Mass_ES/BSA', 'LV_Mass_ED/BSA', 'Hypertrophied segment']]
    ecg_scar_df = ecg_scar_df.merge(mri, on='Record_ID')

    ecg_scar_df['ECG_MRI_Diff'] = abs(ecg_scar_df['ECG Date'] - ecg_scar_df['MRI Date'])

    # Keep ECGs whose date difference between MRI and ECG is less than defined days.
    ecg_scar_df = ecg_scar_df[ecg_scar_df['ECG_MRI_Diff'] <= pd.Timedelta(days=mri_ecg_days)]

    # Extract QT segments
    ecg_scar_df[['Record_ID', 'ECG_ID', 'Frequency']].to_excel('temp_ecg_meta.xlsx', index=False)
    extract_dict = QTSegmentExtractor(ecg_dir_path='Data/ECG/HopkinsOneYearECGs',
                                      ann_dir_path='Data/ECG/HopkinsOneYearECGs_PLAAnnotation',
                                      metadata_path='temp_ecg_meta.xlsx',
                                      verbose=True).extract_segments()

    ecg_skipped = {}
    dataset = pd.DataFrame()
    for _, pid in enumerate(pbar := tqdm(set(ecg_scar_df['Record_ID']), total=len(set(ecg_scar_df['Record_ID'])))):
        if pid not in extract_dict:
            ecg_skipped[pid] = 'No ECG was found'
            continue
        ecgs = extract_dict[pid]
        pbar.set_description(f'Extract features from {len(ecgs)} ECGs for PID = {pid}')
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
                print(f'Error: {e} (PID = {pid}, ECG_ID = {ecg_dict["ecg_id"]})')
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

            df_target = ecg_scar_df.loc[ecg_scar_df['ECG_ID'] == ecg_dict['ecg_id']]
            df_target.reset_index(drop=True, inplace=True)
            feature_vector = pd.concat([df_target, heartrate_variability, qrs_features, st_features, t_features, baseline_features], axis=1)
            dataset = pd.concat([dataset, feature_vector], ignore_index=True, axis=0)

    # Keep only maximum first 3 closest ECGs to MRI for each patient
    def keep_first_rows_noscar(group):
        return group.head(3)

    def keep_first_rows_scar(group):
        return group.head(3)

    dataset_noscar = dataset.loc[dataset['Scar'] == 0]
    dataset_noscar = dataset_noscar.sort_values(by=['Record_ID', 'ECG_MRI_Diff'])
    dataset_noscar = dataset_noscar.groupby('Record_ID').apply(keep_first_rows_noscar)
    dataset_noscar.reset_index(drop=True, inplace=True)
    dataset_scar = dataset.loc[dataset['Scar'] > 0]
    dataset_scar = dataset_scar.sort_values(by=['Record_ID', 'ECG_MRI_Diff'])
    dataset_scar = dataset_scar.groupby('Record_ID').apply(keep_first_rows_scar)
    dataset_scar.reset_index(drop=True, inplace=True)

    dataset = pd.concat([dataset_scar, dataset_noscar], ignore_index=True)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset.to_excel(f'{name}.xlsx', index=False)


def get_significant_features(region_name: str):
    df = pd.read_excel(f'{dataset_name}.xlsx')
    selected_features = []
    for feature in df.columns:
        if '(' in feature and ')' in feature:
            population_zero = df.loc[df['Scar'] == 0][feature].values
            if region_name == 'NonApical':
                population_target = df.loc[((df['Basal'] > 0) | (df['Mid'] > 0) & (df['Apical'] == 0))][feature].values
            else:
                population_target = df.loc[df[region_name] > 0][feature].values
            _, p_value = ttest_ind(a=population_zero, b=population_target, equal_var=False)
            selected_features.append((feature, p_value))

    selected_features = sorted(selected_features, key=lambda x: x[1])
    return selected_features


def run_clustering(region_name: str, dataset: pd.DataFrame, recursive: int, remove_pids: list = None):
    """
    This function clusters patients based on their ECG features such that the resulting clusters have similar ECG traits
    but separable in terms of Scar vs. NoScar as much as possible. It uses the DPGMM algorithm to first partition
    patients into some initial clusters and then merges those clusters where Scar patients dominate NoScar or vice
    versa. Notably, this function is not implemented recursively. The user must recall it recursively to merge the
    resulting clusters until no patient is left.
    :param region_name: Region name to cluster patients based on. It can be 'Basal', 'Mid', 'Apical', or 'NonApical'.
    :param dataset: Dataset to cluster patients based on.
    :param recursive: Number of times to recursively merge clusters.
    :param remove_pids: List of patient IDs to remove from the dataset. It is the patients that have been merged and
     identified in iterations before [1, recursive-1].
    :return: None
    """
    print(f'****** Clustering {region_name} | Recursive = {recursive} ********')
    if region_name == 'NonApical':
        dataset = dataset.loc[(dataset['Scar'] == 0) | (((dataset['Basal'] > 0) | (dataset['Mid'] > 0)) & (dataset['Apical'] == 0))]
    else:
        dataset = dataset.loc[(dataset['Scar'] == 0) | (dataset[region_name] > 0)]
    if remove_pids is not None:
        dataset = dataset.loc[~dataset['Record_ID'].isin(remove_pids)]
    dataset['target'] = [0 if scar == 0 else 1 for scar in dataset['Scar'].values]
    dataset.reset_index(drop=True, inplace=True)

    n_0 = len(set(dataset.loc[dataset['Scar'] == 0]['Record_ID']))
    if region_name == 'NonApical':
        n_1 = len(set(dataset.loc[((dataset['Basal'] > 0) | (dataset['Mid'] > 0)) & (dataset['Apical'] == 0)]['Record_ID']))
    else:
        n_1 = len(set(dataset.loc[dataset[region_name] > 0]['Record_ID']))
    n_ecgs = len(dataset)
    print(f'# of NoScar Patients = {n_0}   |   # of {region_name} Scar Patients = {n_1}  |   # of ECGs = {n_ecgs}')

    lim = int(input('Enter number of features to use: '))
    n_components = int(input('Enter number of components to use: '))
    min_scar = int(input('Enter minimum scar size to collect: '))
    min_noscar = int(input('Enter minimum noscar size to collect: '))
    scar_ratio = float(input('Enter scar ratio to use: '))
    noscar_ratio = float(input('Enter noscar ratio to use: '))

    features = get_significant_features(region_name=region_name)

    for i, scar in enumerate(tqdm(range(100), total=100)):

        selected_features = [f[0] for f in features[:lim]]
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type="full")
        preds = list(dpgmm.fit_predict(StandardScaler().fit_transform(dataset[selected_features])))

        scar_dominant = set()
        noscar_dominant = set()
        for c in range(n_components):
            n_scar, n_noscar = set(), set()
            for i, pred in enumerate(preds):
                if pred == c:
                    [pid, scar] = dataset.iloc[i][['Record_ID', 'target']].values
                    if scar == 0:
                        n_noscar.add(pid)
                    else:
                        n_scar.add(pid)
            if len(n_noscar) == 0:
                ratio = math.inf
            else:
                ratio = round(len(n_scar) / len(n_noscar), 2)

            if ratio == math.inf or ratio > scar_ratio:
                scar_dominant = scar_dominant | n_scar | n_noscar

            if ratio <= noscar_ratio:
                noscar_dominant = noscar_dominant | n_scar | n_noscar

        scar_dominant_df = pd.merge(left=pd.DataFrame(data=list(scar_dominant), columns=['Record_ID']),
                                    right=dataset,
                                    on=['Record_ID'],
                                    how='inner')
        scar_dominant_meta = scar_dominant_df[['Record_ID', 'Scar tissue %', 'target']].drop_duplicates(ignore_index=True)
        n_scar_dominant = scar_dominant_meta.loc[scar_dominant_meta['target'] == 1].shape[0]

        noscar_dominant_df = pd.merge(left=pd.DataFrame(data=list(noscar_dominant), columns=['Record_ID']),
                                      right=dataset,
                                      on=['Record_ID'],
                                      how='inner')
        noscar_dominant_meta = noscar_dominant_df[['Record_ID', 'Scar tissue %', 'target']].drop_duplicates(ignore_index=True)
        n_noscar_dominant = noscar_dominant_meta.loc[(noscar_dominant_meta['target'] == 0)].shape[0]

        if n_scar_dominant < min_scar or n_noscar_dominant < min_noscar:
            continue

        _, _, f1_noscar_record, f1_scar_record = predict_logistic_regression(region_name, scar_dominant_meta, noscar_dominant_meta)

        if f1_noscar_record < 0.8 or f1_scar_record < 0.8:
            continue
        print(f'\n\n Dominant Scar Super Cluster so far:')
        print(f'Scar={n_scar_dominant} | NoScar={n_noscar_dominant} ({(n_scar_dominant + n_noscar_dominant) / (n_0 + n_1) * 100:.1f}%)')
        print(f'F1 Score (Patient): Scar={f1_scar_record:.2f} | NoScar={f1_noscar_record:.2f}')
        change_p_value = input('Change p-value? (Yes/No): ')
        p_value = 0.0001
        if change_p_value.lower() == 'yes':
            p_value = float(input('Enter new p-value: '))
            f1_noscar_ecg, f1_scar_ecg, f1_noscar_record, f1_scar_record = predict_logistic_regression(region_name,
                                                                                                       scar_dominant_meta,
                                                                                                       noscar_dominant_meta,
                                                                                                       max_p_value=p_value)
            print(f'F1 Score (Patient): Scar={f1_scar_record:.2f} | NoScar={f1_noscar_record:.2f}')

        what = input('What do you want to do? (Stay/Next/Skip)')
        if what == 'Stay':
            scar_dominant_meta.to_excel(f'Adjusted/gmm_{recursive}_{dataset_name}_scar_dominant_{region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx', index=False)
            noscar_dominant_meta.to_excel(f'Adjusted/gmm_{recursive}_{dataset_name}_noscar_dominant_{region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx', index=False)
            print(f'Results saved as {region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx')
        elif what == 'Next':
            scar_dominant_meta.to_excel(f'Adjusted/gmm_{recursive}_{dataset_name}_scar_dominant_{region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx', index=False)
            noscar_dominant_meta.to_excel(f'Adjusted/gmm_{recursive}_{dataset_name}_noscar_dominant_{region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx', index=False)
            print(f'Results saved as {region_name}_{n_scar_dominant}_{n_noscar_dominant}.xlsx')

            pids_scar = set(scar_dominant_meta.loc[scar_dominant_meta['target'] == 1]['Record_ID'])
            pids_noscar = set(noscar_dominant_meta.loc[noscar_dominant_meta['target'] == 0]['Record_ID'])
            pids_to_remove = list(pids_scar | pids_noscar)
            run_clustering(region_name=region_name, dataset=dataset, recursive=recursive + 1, remove_pids=pids_to_remove)
        else:
            change_anything = input('Do you want to change any parameters? (Yes/No)')
            if change_anything == 'Yes':
                change = input('Do you want to change noscar ratio? (Yes/No)')
                if change == 'Yes':
                    noscar_ratio = float(input('Enter new noscar ratio: '))

                change = input('Do you want to change scar ratio? (Yes/No)')
                if change == 'Yes':
                    scar_ratio = float(input('Enter new scar ratio: '))

                change = input('Do you want to change NoScar min size? (Yes/No)')
                if change == 'Yes':
                    min_noscar = float(input('Enter new NoScar min size: '))


def predict_logistic_regression(region_name: str, gmm_scar, gmm_noscar: pd.DataFrame, max_p_value=0.0001):
    pids_scar = set(gmm_scar.loc[gmm_scar['target'] == 1]['Record_ID'])
    pids_noscar = set(gmm_noscar.loc[gmm_noscar['target'] == 0]['Record_ID'])
    dataset = pd.read_excel(f'{dataset_name}.xlsx')
    dataset = dataset.loc[dataset['Record_ID'].isin(pids_scar | pids_noscar)]
    dataset['target'] = [0 if scar == 0 else 1 for scar in dataset['Scar'].values]

    selected_features = []
    for feature in dataset.columns:
        if '(' in feature and ')' in feature:
            population_zero = dataset.loc[dataset['Scar'] == 0][feature].values
            if region_name == 'NonApical':
                population_target = dataset.loc[((dataset['Basal'] > 0) | (dataset['Mid'] > 0) & (dataset['Apical'] == 0))][feature].values
            else:
                population_target = dataset.loc[dataset[region_name] > 0][feature].values
            _, p_value = ttest_ind(a=population_zero, b=population_target, equal_var=False)
            selected_features.append((feature, p_value))

    selected_features = sorted(selected_features, key=lambda x: x[1])
    selected_features = [f[0] for f in selected_features if f[1] < max_p_value]
    # print(f'Number of features: {len(selected_features)}')

    pids = list(set(dataset['Record_ID'].values))

    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    pred_df = pd.DataFrame(columns=['Record_ID', 'ECG_ID', 'target', 'pred'])
    for train_idx, test_idx in kf.split(pids):
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train = dataset.loc[dataset['Record_ID'].isin(train_pids)]
        test = dataset.loc[dataset['Record_ID'].isin(test_pids)]

        train_data = train[selected_features]
        train_target = train['target']

        test_data = test[selected_features]
        test_target = test['target']

        sm = SMOTE(random_state=42)
        train_data, train_target = sm.fit_resample(train_data, train_target)

        scaler = StandardScaler()
        train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
        test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

        clf = LogisticRegression()
        clf.fit(train_data, train_target)
        pred = clf.predict(test_data)
        pred_df = pred_df.append(pd.DataFrame(data={'Record_ID': test['Record_ID'].values,
                                                    'ECG_ID': test['ECG_ID'].values,
                                                    'target': test_target.values,
                                                    'pred': pred}))
    return compute_performance(pred_df)


def predict_scarf(region_name: str, gmm_scar, gmm_noscar: pd.DataFrame, max_p_value=0.0001):
    pids_scar = set(gmm_scar.loc[gmm_scar['target'] == 1]['Record_ID'])
    pids_noscar = set(gmm_noscar.loc[gmm_noscar['target'] == 0]['Record_ID'])
    features = get_significant_features(region_name)
    selected_features = [f[0] for f in features if f[1] < max_p_value]
    print(f'Number of features: {len(selected_features)}')

    dataset = pd.read_excel(f'{dataset_name}.xlsx')
    dataset = dataset.loc[dataset['Record_ID'].isin(pids_scar | pids_noscar)]
    dataset['target'] = [0 if scar == 0 else 1 for scar in dataset['Scar'].values]
    pids = list(set(dataset['Record_ID'].values))

    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    pred_df = pd.DataFrame(columns=['Record_ID', 'ECG_ID', 'target', 'pred'])
    for train_idx, test_idx in kf.split(pids):
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train = dataset.loc[dataset['Record_ID'].isin(train_pids)]
        test = dataset.loc[dataset['Record_ID'].isin(test_pids)]

        train_data = train[selected_features]
        train_target = train['target']

        test_data = test[selected_features]
        test_target = test['target']

        sm = SMOTE(random_state=42)
        train_data, train_target = sm.fit_resample(train_data, train_target)

        scaler = StandardScaler()
        train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
        test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

        clf = LogisticRegression()
        clf.fit(train_data, train_target)

        # to torch dataset
        train_ds = ScarfDataset(
            train_data.to_numpy(),
            train_target.to_numpy(),
            columns=train_data.columns
        )
        test_ds = ScarfDataset(
            test_data.to_numpy(),
            test_data.to_numpy(),
            columns=test_data.columns
        )

        batch_size = 64
        epochs = 10_000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = SCARF(
            input_dim=train_ds.shape[1],
            emb_dim=16,
            corruption_rate=0.6,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        ntxent_loss = NTXent()

        loss_history = []

        for epoch in range(1, epochs + 1):
            epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
            loss_history.append(epoch_loss)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # get embeddings for training and test set
        train_embeddings = dataset_embeddings(model, train_loader, device)
        test_embeddings = dataset_embeddings(model, test_loader, device)

        train_target = list(train_target.values)
        test_target = list(test_target.values)

        clf.fit(train_embeddings, train_target)
        pred = clf.predict(test_embeddings)

        pred_df = pred_df.append(pd.DataFrame(data={'Record_ID': test['Record_ID'].values,
                                                    'ECG_ID': test['ECG_ID'].values,
                                                    'target': test_target,
                                                    'pred': pred}))
    return compute_performance(pred_df)


def compute_performance(pred_df: pd.DataFrame):
    assert 'target' in pred_df.columns
    assert 'pred' in pred_df.columns
    assert 'ECG_ID' in pred_df.columns
    assert 'Record_ID' in pred_df.columns

    # ECG-based F1 score
    f1_noscar_ecg = f1_score(y_true=list(pred_df['target']), y_pred=list(pred_df['pred']), pos_label=0)
    f1_scar_ecg = f1_score(y_true=list(pred_df['target']), y_pred=list(pred_df['pred']), pos_label=1)

    # Record-based F1 score
    pred_df = pred_df.groupby(['Record_ID', 'target']).agg({'pred': ['mean']}).apply(lambda x: round(x)).reset_index()
    pred_df.columns = ['Record_ID', 'target', 'pred']
    f1_noscar_record = f1_score(y_true=list(pred_df['target']), y_pred=list(pred_df['pred']), pos_label=0)
    f1_scar_record = f1_score(y_true=list(pred_df['target']), y_pred=list(pred_df['pred']), pos_label=1)
    return f1_noscar_ecg, f1_scar_ecg, f1_noscar_record, f1_scar_record


def get_performance_all_groups(region_name: str, n_groups: int):
    result_patient = {}
    y_pred_original = []
    y_pred_scarf = []
    y_true = []
    for group in range(1, n_groups + 1):
        for fold in range(1, 6):
            result = pd.read_excel(f'Adjusted/Pred/{region_name}_g{group}_f{fold}.xlsx')
            pids = set(result['Record_ID'].values)
            for pid in pids:
                target = list(result.loc[result['Record_ID'] == pid]['target'].values)
                pred_scarf = list(result.loc[result['Record_ID'] == pid]['pred_scarf'].values)
                pred_orig = list(result.loc[result['Record_ID'] == pid]['pred_original'].values)
                if pid in result_patient:
                    result_patient[pid]['target'] += target
                    result_patient[pid]['pred_original'] += pred_orig
                    result_patient[pid]['pred_scarf'] += pred_scarf
                else:
                    result_patient[pid] = {'target': target, 'pred_original': pred_orig, 'pred_scarf': pred_scarf}

            y_true += list(result['target'].values)
            y_pred_scarf += list(result['pred_scarf'].values)
            y_pred_original += list(result['pred_original'].values)

    y_true_patient = []
    y_pred_patient_original = []
    y_pred_patient_scarf = []
    for pid in result_patient:
        if sum(result_patient[pid]['target']) == 0:
            y_true_patient.append(0)
        else:
            y_true_patient.append(1)

        if list(result_patient[pid]['pred_original']).count(1) > list(result_patient[pid]['pred_original']).count(0):
            y_pred_patient_original.append(1)
        else:
            y_pred_patient_original.append(0)

        if list(result_patient[pid]['pred_scarf']).count(1) > list(result_patient[pid]['pred_scarf']).count(0):
            y_pred_patient_scarf.append(1)
        else:
            y_pred_patient_scarf.append(0)

    print(f'\n--- {region_name} Patient-specific results ORIGINAL:')
    print(classification_report(y_true=y_true_patient, y_pred=y_pred_patient_original))

    print(f'\n--- {region_name} Patient-specific results SCARF:')
    print(classification_report(y_true=y_true_patient, y_pred=y_pred_patient_scarf))


def get_confidence_intervals():
    regions = {'Basal': 2, 'Mid': 2, 'Apical': 2}
    meta = pd.read_excel(f'datasetV3.xlsx')[['Record_ID', 'ECG_ID', 'Scar tissue %', 'MRI Date', 'ECG Date']]
    pred_df = pd.DataFrame(columns=['Record_ID', 'ECG_ID', 'Region', 'target', 'pred', 'prob_noscar', 'prob_scar'])
    for region_name, n_groups in regions.items():
        for group in range(1, n_groups + 1):
            # Step 1: Get the dataset for the group
            gmm_scar = pd.read_excel(f'Adjusted/Final/{region_name}_g{group}_scar.xlsx')
            gmm_noscar = pd.read_excel(f'Adjusted/Final/{region_name}_g{group}_noscar.xlsx')
            pids_scar = set(gmm_scar.loc[gmm_scar['target'] == 1]['Record_ID'])
            pids_noscar = set(gmm_noscar.loc[gmm_noscar['target'] == 0]['Record_ID'])

            dataset = pd.read_excel(f'{dataset_name}.xlsx')
            dataset = dataset.loc[dataset['Record_ID'].isin(pids_scar | pids_noscar)]

            features = []
            for feature in dataset.columns:
                if '(' in feature and ')' in feature:
                    population_zero = dataset.loc[dataset['Scar'] == 0][feature].values
                    population_target = dataset.loc[dataset[region_name] > 0][feature].values
                    _, p_value = ttest_ind(a=population_zero, b=population_target, equal_var=False)
                    features.append((feature, p_value))
            features = sorted(features, key=lambda x: x[1])
            max_p_value = 0.0001
            selected_features = [f[0] for f in features if f[1] < max_p_value]
            # while len(selected_features) < 50 and max_p_value <= 0.01:
            #     max_p_value *= 10
            #     selected_features = [f[0] for f in features if f[1] < max_p_value]

            dataset = dataset.loc[(dataset['Scar'] == 0) | (dataset[region_name] > 0)]
            dataset['target'] = [0 if scar == 0 else 1 for scar in dataset['Scar'].values]

            n_0 = len(set(dataset.loc[dataset['Scar'] == 0]['Record_ID'].values))
            n_1 = len(set(dataset.loc[dataset[region_name] > 0]['Record_ID']))
            print(f'--- {region_name} – Group {group} ---')
            print(f'NoScar Patients={n_0} | Scar Patients={n_1}')

            # Step 2: Get the SHAP values for the group
            pids = list(set(dataset['Record_ID'].values))
            fold = 0
            kf = KFold(n_splits=5, random_state=None, shuffle=False)
            for train_idx, test_idx in kf.split(pids):
                fold += 1
                train_pids = [pids[i] for i in train_idx]
                test_pids = [pids[i] for i in test_idx]
                train = dataset.loc[dataset['Record_ID'].isin(train_pids)]
                test = dataset.loc[dataset['Record_ID'].isin(test_pids)]

                train_data = train[selected_features]
                train_target = train['target']

                test_data = test[selected_features]
                test_target = test['target']

                sm = SMOTE(random_state=42)
                train_data, train_target = sm.fit_resample(train_data, train_target)

                scaler = StandardScaler()
                train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
                test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
                train_target = list(train_target.values)
                test_target = list(test_target.values)

                clf = LogisticRegression()
                clf.fit(train_data, train_target)
                pred = clf.predict(test_data)
                pred_prob = clf.predict_proba(test_data)
                test['pred'] = pred
                # print(classification_report(y_true=test_target, y_pred=pred))
                test['prob_noscar'] = [p[0] for p in pred_prob]
                test['prob_scar'] = [p[1] for p in pred_prob]
                test['Region'] = region_name
                test['score'] = clf.score(test_data, test_target)
                pred_df = pred_df.append(test[['Record_ID', 'ECG_ID', 'Region', 'target', 'pred', 'prob_noscar', 'prob_scar', 'score']])

    def compute_lv_confidence(row: pd.Series):
        if row['pred'] == 0:
            return row['prob_noscar']
        return row['prob_scar']

    pred_df_lv = pred_df.groupby(['Record_ID', 'ECG_ID'], as_index=False).agg({'target': 'max', 'pred': 'max', 'prob_noscar': 'max', 'prob_scar': 'max', 'score': 'mean'})
    pred_df_lv['Confidence'] = pred_df_lv.apply(compute_lv_confidence, axis=1)
    pred_df_lv = pred_df_lv.groupby(['Record_ID'], as_index=False).agg({'target': 'mean', 'pred': 'mean', 'Confidence': 'mean', 'score': 'mean'})
    pred_df_lv['target'] = pred_df_lv['target'].apply(lambda x: round(x))
    pred_df_lv['pred'] = pred_df_lv['pred'].apply(lambda x: round(x))
    print(classification_report(pred_df_lv['target'], pred_df_lv['pred']))

    def translate_prediction(row: pd.Series):
        if row['pred'] == row['target']:
            return 'Correct'
        return 'Incorrect'

    pred_df_lv['Scar Prediction'] = pred_df_lv.apply(translate_prediction, axis=1)
    pred_df_lv['Region'] = 'Entire LV'
    pred_df_lv = pred_df_lv[['Record_ID', 'Region', 'Scar Prediction', 'target', 'pred', 'Confidence', 'score']]

    pred_df_basal = pred_df.loc[pred_df['Region'] == 'Basal']
    pred_df_basal['Confidence'] = pred_df_basal.apply(compute_lv_confidence, axis=1)
    pred_df_basal = pred_df_basal.groupby(['Record_ID'], as_index=False).agg({'target': 'mean', 'pred': 'mean', 'Confidence': 'mean', 'score': 'mean'})
    pred_df_basal['target'] = pred_df_basal['target'].apply(lambda x: round(x))
    pred_df_basal['pred'] = pred_df_basal['pred'].apply(lambda x: round(x))
    pred_df_basal['Scar Prediction'] = pred_df_basal.apply(translate_prediction, axis=1)
    pred_df_basal['Region'] = 'Basal'
    pred_df_basal = pred_df_basal[['Record_ID', 'Region', 'Scar Prediction', 'target', 'pred', 'Confidence', 'score']]

    pred_df_mid = pred_df.loc[pred_df['Region'] == 'Mid']
    pred_df_mid['Confidence'] = pred_df_mid.apply(compute_lv_confidence, axis=1)
    pred_df_mid = pred_df_mid.groupby(['Record_ID'], as_index=False).agg({'target': 'mean', 'pred': 'mean', 'Confidence': 'mean', 'score': 'mean'})
    pred_df_mid['target'] = pred_df_mid['target'].apply(lambda x: round(x))
    pred_df_mid['pred'] = pred_df_mid['pred'].apply(lambda x: round(x))
    pred_df_mid['Scar Prediction'] = pred_df_mid.apply(translate_prediction, axis=1)
    pred_df_mid['Region'] = 'Mid'
    pred_df_mid = pred_df_mid[['Record_ID', 'Region', 'Scar Prediction', 'target', 'pred', 'Confidence', 'score']]

    pred_df_apical = pred_df.loc[pred_df['Region'] == 'Apical']
    pred_df_apical['Confidence'] = pred_df_apical.apply(compute_lv_confidence, axis=1)
    pred_df_apical = pred_df_apical.groupby(['Record_ID'], as_index=False).agg({'target': 'mean', 'pred': 'mean', 'Confidence': 'mean', 'score': 'mean'})
    pred_df_apical['target'] = pred_df_apical['target'].apply(lambda x: round(x))
    pred_df_apical['pred'] = pred_df_apical['pred'].apply(lambda x: round(x))
    pred_df_apical['Scar Prediction'] = pred_df_apical.apply(translate_prediction, axis=1)
    pred_df_apical['Region'] = 'Apical'
    pred_df_apical = pred_df_apical[['Record_ID', 'Region', 'Scar Prediction', 'target', 'pred', 'Confidence', 'score']]

    mri = pd.read_excel('MRI_meta.xlsx')[['Record_ID', 'Scar tissue %']]
    total = pd.concat([pred_df_lv, pred_df_basal, pred_df_mid, pred_df_apical], axis=0)
    total = pd.merge(left=total, right=mri, on='Record_ID', how='left')

    for region_name in ['Entire LV', 'Basal', 'Mid', 'Apical']:
        df = total.loc[total['Region'] == region_name]
        mean_score = np.mean(df['score'])
        std_error = np.std(df['score']) / np.sqrt(len(df['score']))
        conf_int = (mean_score - 1.96 * std_error, mean_score + 1.96 * std_error)
        print(f'{region_name} Accuracy: {mean_score:.3f} ± {std_error:.3f} (95% CI: {conf_int[0]:.3f} – {conf_int[1]:.3f})')
    v = 9

    return total


def plot_confidence_figures(total: pd.DataFrame):
    # sns.set_theme()
    sns.set_theme(style="white")
    plt.figure(figsize=(20, 20))
    g = sns.catplot(data=total.loc[total['Region'] == 'Entire LV'], x="Scar Prediction", y="Confidence", kind="violin", color='.95',  inner=None)
    sns.swarmplot(data=total.loc[total['Region'] == 'Entire LV'], x="Scar Prediction", y="Confidence", size=4, palette=['#008000', '#E31B23'])

    g.ax.set_xlabel('Method\'s Prediction', fontsize=14)
    g.ax.set_ylabel('Method\'s Confidence', fontsize=14)
    g.ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('Adjusted/confidence_per_detection.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    g = sns.histplot(data=total.loc[total['Region'] == 'Entire LV'], x="Scar tissue %", hue="Scar Prediction", element="poly", common_norm=False, multiple="stack", palette=['#008000', '#E31B23'])
    g.set_xlabel('LV Scar Tissue %', fontsize=14)
    g.set_ylabel('Number of Patients', fontsize=14)
    g.tick_params(axis='both', which='major', labelsize=14)
    g.legend(['Incorrect prediction', 'Correct prediction'], fontsize=14)
    g.spines['right'].set_visible(False)
    g.spines['top'].set_visible(False)
    plt.savefig('Adjusted/scar_stratified.png', dpi=300)
    plt.show()

    df = total.loc[total['Region'] == 'Entire LV']
    x = df['Scar tissue %']
    y = df['Confidence']
    r, p = stats.pearsonr(x, y)
    plt.figure(figsize=(20, 20))
    plt.figure(layout='constrained')
    graph = sns.jointplot(data=df, x='Scar tissue %', y='Confidence', kind='reg',
                          color=[0 / 255, 92 / 255, 171 / 255], height=8, ratio=4,
                          joint_kws={'line_kws': {'color': [227 / 255, 27 / 255, 35 / 255]}})
    phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0.0)
    graph.ax_joint.legend([phantom], [f'r={round(r, 2)}, p<0.001'], loc='upper right', fontsize=18, frameon=True,
                          facecolor='white', edgecolor='white')
    graph.ax_joint.set_xlabel('LV Scar Tissue %', fontsize=15)
    graph.ax_joint.set_ylabel('Method\'s Confidence', fontsize=15)
    graph.ax_joint.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('Adjusted/scar_percent_by_conf.png', dpi=300)
    plt.show()


def get_shap_values(region_name: str, n_groups: int):
    for group in range(1, n_groups + 1):
        shaps = []
        # Step 1: Get the dataset for the group
        gmm_scar = pd.read_excel(f'Adjusted/Final/{region_name}_g{group}_scar.xlsx')
        gmm_noscar = pd.read_excel(f'Adjusted/Final/{region_name}_g{group}_noscar.xlsx')
        pids_scar = set(gmm_scar.loc[gmm_scar['target'] == 1]['Record_ID'])
        pids_noscar = set(gmm_noscar.loc[gmm_noscar['target'] == 0]['Record_ID'])

        dataset = pd.read_excel(f'{dataset_name}.xlsx')
        dataset = dataset.loc[dataset['Record_ID'].isin(pids_scar | pids_noscar)]

        features = []
        for feature in dataset.columns:
            if '(' in feature and ')' in feature:
                population_zero = dataset.loc[dataset['Scar'] == 0][feature].values
                if region_name == 'NonApical':
                    population_target = \
                    dataset.loc[((dataset['Basal'] > 0) | (dataset['Mid'] > 0) & (dataset['Apical'] == 0))][
                        feature].values
                else:
                    population_target = dataset.loc[dataset[region_name] > 0][feature].values
                _, p_value = ttest_ind(a=population_zero, b=population_target, equal_var=False)
                features.append((feature, p_value))

        features = sorted(features, key=lambda x: x[1])
        max_p_value = 0.0001
        selected_features = [f[0] for f in features if f[1] < max_p_value]
        while len(selected_features) < 50 and max_p_value <= 0.01:
            max_p_value *= 10
            selected_features = [f[0] for f in features if f[1] < max_p_value]

        if region_name == 'NonApical':
            dataset = dataset.loc[
                (dataset['Scar'] == 0) | (((dataset['Basal'] > 0) | (dataset['Mid'] > 0)) & (dataset['Apical'] == 0))]
        else:
            dataset = dataset.loc[(dataset['Scar'] == 0) | (dataset[region_name] > 0)]
        dataset['target'] = [0 if scar == 0 else 1 for scar in dataset['Scar'].values]

        n_0 = len(set(dataset.loc[dataset['Scar'] == 0]['Record_ID'].values))
        if region_name == 'NonApical':
            n_1 = len(set(
                dataset.loc[((dataset['Basal'] > 0) | (dataset['Mid'] > 0)) & (dataset['Apical'] == 0)]['Record_ID']))
        else:
            n_1 = len(set(dataset.loc[dataset[region_name] > 0]['Record_ID']))
        print(f'--- {region_name} – Group {group} ---')
        print(f'NoScar Patients={n_0} | Scar Patients={n_1}')

        # Step 2: Get the SHAP values for the group
        pids = list(set(dataset['Record_ID'].values))
        fold = 0
        fold_shap_values = []
        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        for train_idx, test_idx in kf.split(pids):
            fold += 1
            train_pids = [pids[i] for i in train_idx]
            test_pids = [pids[i] for i in test_idx]
            train = dataset.loc[dataset['Record_ID'].isin(train_pids)]
            test = dataset.loc[dataset['Record_ID'].isin(test_pids)]

            train_data = train[selected_features]
            train_target = train['target']

            test_data = test[selected_features]
            test_target = test['target']

            sm = SMOTE(random_state=42)
            train_data, train_target = sm.fit_resample(train_data, train_target)

            scaler = StandardScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
            test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

            # to torch dataset
            train_ds = ScarfDataset(
                train_data.to_numpy(),
                train_target.to_numpy(),
                columns=train_data.columns
            )
            test_ds = ScarfDataset(
                test_data.to_numpy(),
                test_data.to_numpy(),
                columns=test_data.columns
            )

            epochs = 15_000
            batch_size = 64
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            model = SCARF(
                input_dim=train_ds.shape[1],
                emb_dim=16,
                corruption_rate=0.6,
            ).to(device)
            optimizer = Adam(model.parameters(), lr=0.001)
            ntxent_loss = NTXent()

            loss_history = []

            for epoch in range(1, epochs + 1):
                epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
                loss_history.append(epoch_loss)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # get embeddings for training and test set
            train_embeddings = dataset_embeddings(model, train_loader, device)
            test_embeddings = dataset_embeddings(model, test_loader, device)

            train_target = list(train_target.values)
            test_target = list(test_target.values)

            clf = LogisticRegression()

            # Train the classifier on original data
            clf.fit(train_data, train_target)
            pred = clf.predict(test_data)
            test['pred_original'] = pred
            # print(classification_report(test_target, pred))

            clf.fit(train_embeddings, train_target)
            pred = clf.predict(test_embeddings)
            test['pred_scarf'] = pred
            # print(classification_report(test_target, pred))
            test[['Record_ID', 'ECG_ID', region_name, 'target', 'pred_original', 'pred_scarf']].to_excel(f'Adjusted/Pred/{region_name}_g{group}_f{fold}.xlsx', index=False)

            def get_predict(data: pd.DataFrame):
                dummy_target = np.array([-1 for _ in range(len(data))])
                data = ScarfDataset(data, dummy_target)
                loader = DataLoader(data, batch_size=batch_size, shuffle=False)
                embeddings = dataset_embeddings(model, loader, device)
                pred = clf.predict_proba(embeddings)
                return pred

            explainer = shap.KernelExplainer(get_predict, train_data)
            shap_value = explainer.shap_values(test_data)
            fold_shap_values.append(shap_value[0])

            # Save SHAP values for each ECG
            test_df_copy = test.copy()
            test_df_copy.reset_index(drop=True, inplace=True)
            for i, row in test_df_copy.iterrows():
                ecg_id = int(row['ECG_ID'])
                shap_vector = shap_value[0][i]
                shap_df = pd.DataFrame(data={'Feature': selected_features,
                                             'SHAP': shap_vector})
                shap_df.to_excel(f'Adjusted/SHAP/Values/{ecg_id}_{region_name}_shap.xlsx', index=False)

        mean_shap_values_no_scar = []
        for i in range(fold_shap_values[0].shape[1]):
            temp = []
            for shaps in fold_shap_values:
                temp += list(shaps[:, i])
            mean_shap_values_no_scar.append(statistics.mean([abs(t) for t in temp]))

        pd.Series(data=mean_shap_values_no_scar, index=selected_features).to_excel(f'Adjusted/SHAP/{region_name}_g{group}_shap.xlsx')


if __name__ == '__main__':
    dataset_name = 'datasetV3_adjusted'
    prepare_dataset(dataset_name, mri_ecg_days=31)
    dataset = pd.read_excel(f'{dataset_name}.xlsx')
    dataset = dataset.dropna(subset=['Scar tissue %'])

    region_name = input('Enter region name: ')

    # Step 1: Run clustering to identify scar dominant and non-scar dominant patients.
    pids_to_remove = []
    # run_clustering(region_name=region_name, dataset=dataset, recursive=2, remove_pids=pids_to_remove) ...
    # Save the results of clustering in `Unsupervised` folder. Per each recursion i the `run_clustering` has made, this
    # folder should contain 2 files: `{region_name}_{i}_noscar.xlsx` and `{region_name}_{i}_scar.xlsx`
    i = 1   # The number of recursion that has been used to generate the clustering results.
    gmm_scar = pd.read_excel(f'Unsupervised/{region_name}_{i}_scar.xlsx')
    gmm_noscar = pd.read_excel(f'Unsupervised/{region_name}_{i}_noscar.xlsx')
    # Step 2: Predict scar.
    predict_scarf(region_name, gmm_scar, gmm_noscar)
    get_performance_all_groups(region_name=region_name, n_groups=2)
    get_shap_values(region_name=region_name, n_groups=2)





