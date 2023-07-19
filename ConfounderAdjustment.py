import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import GlobalPaths
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def adjust_for_covariant(covariant_df: pd.DataFrame, dataset: pd.DataFrame):
    """
    Adjust ECG dataset for covariant_df
    :param covariant_df: covariant data frame containing age, gender, and LV_Mass_Index per ECG_ID per patient.
    :param dataset: ECG data frame containing ECG features per ECG_ID per patient.
    :return: adjusted dataset
    """
    assert covariant_df.shape[0] == dataset.shape[0]
    assert covariant_df.shape[1] == 4
    assert set(covariant_df['Record_ID']) - set(covariant_df['Record_ID']) == set()

    def compute_y_residuals(y):
        cov_data = covariant_df.drop(['Record_ID'], axis=1).values
        lr = LinearRegression().fit(cov_data, y)
        y_residuals = y - np.matmul(cov_data, lr.coef_) - lr.intercept_
        return y_residuals

    ecg_features = [f for f in dataset.columns if '(' in f and ')' in f]
    for feature in ecg_features:
        dataset[feature] = compute_y_residuals(dataset[feature])
    return dataset


def check():
    df_original = pd.read_excel('datasetV3.xlsx')
    df_adjusted = pd.read_excel('datasetV3_adjusted.xlsx')
    df_original = pd.merge(df_original, df_adjusted[['ECG_ID', 'LV_Mass_Index']], on='ECG_ID', how='inner')

    var_name = '(V3)_QRS_energy'
    x = df_original['LV_Mass_Index']
    y_orig = df_original[var_name]
    y_adjusted = df_adjusted[var_name]
    r, p = scipy.stats.pearsonr(x, y_orig)
    # plt.figure(layout='constrained')
    graph = sns.jointplot(data=df_original, x='LV_Mass_Index', y=var_name, kind='reg',
                          color=[0 / 255, 92 / 255, 171 / 255], height=8, ratio=4,
                          joint_kws={'line_kws': {'color': [227 / 255, 27 / 255, 35 / 255]}})
    phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0.0)
    graph.ax_joint.legend([phantom], [f'r={round(r, 2)}, p={round(p, 3)}'], loc='upper right', fontsize=18, frameon=True,
                          facecolor='white', edgecolor='white')
    graph.ax_joint.set_xlabel('LV Mass Index', fontsize=15)
    graph.ax_joint.set_ylabel(f'Original {var_name}', fontsize=15)
    graph.ax_joint.tick_params(axis='both', which='major', labelsize=15)
    # plt.savefig('Adjusted/mass_scar.png', dpi=300)
    plt.show()

    r, p = scipy.stats.pearsonr(x, y_adjusted)
    # plt.figure(layout='constrained')
    graph = sns.jointplot(data=df_adjusted, x='LV_Mass_Index', y=var_name, kind='reg',
                          color=[0 / 255, 92 / 255, 171 / 255], height=8, ratio=4,
                          joint_kws={'line_kws': {'color': [227 / 255, 27 / 255, 35 / 255]}})
    phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0.0)
    graph.ax_joint.legend([phantom], [f'r={round(r, 2)}, p={round(p, 3)}'], loc='upper right', fontsize=18, frameon=True,
                          facecolor='white', edgecolor='white')
    graph.ax_joint.set_xlabel('LV Mass Index', fontsize=15)
    graph.ax_joint.set_ylabel(f'Adjusted {var_name}', fontsize=15)
    graph.ax_joint.tick_params(axis='both', which='major', labelsize=15)
    # plt.savefig('Adjusted/mass_scar.png', dpi=300)
    plt.show()

    # Check if the adjustment is done correctly by plotting histograms.
    ecg_features = [f for f in df_original.columns if '(' in f and ')' in f]
    for feature in ecg_features:
        df_original[feature].hist(bins=50, grid=False, legend=True)
        df_adjusted[feature].hist(bins=50, grid=False, legend=True)
        plt.show()


if __name__ == '__main__':
    # check()
    mri = pd.read_excel('MRI_meta.xlsx')[['Record_ID', 'LV_Mass_Index']]
    ehr = pd.read_excel(GlobalPaths.ehr)[['Record_ID', 'AGE', 'GENDER']]
    ehr.columns = ['Record_ID', 'Age', 'Gender']  # 1: Male
    mri = pd.merge(left=mri, right=ehr, on='Record_ID', how='inner').dropna()

    ecg = pd.read_excel('datasetV3.xlsx')
    ecg_features = [f for f in ecg.columns if '(' in f and ')' in f]
    dataset = ecg[['Record_ID', 'ECG_ID'] + ecg_features]
    dataset = pd.merge(left=dataset, right=mri, on='Record_ID', how='inner').dropna().reset_index(drop=True)
    covariant_df = dataset[['Record_ID', 'Age', 'Gender', 'LV_Mass_Index']]
    dataset_new = adjust_for_covariant(covariant_df, dataset)
    labels = ecg.loc[ecg['Record_ID'].isin(dataset_new['Record_ID'])][['Record_ID', 'Scar', 'Basal', 'Mid', 'Apical', 'Scar tissue %']]
    dataset_new = pd.merge(left=dataset_new, right=labels, on='Record_ID', how='left').drop_duplicates()
    dataset_new.to_excel('datasetV3_adjusted.xlsx', index=False)


