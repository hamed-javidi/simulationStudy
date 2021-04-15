import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pylab as plt
from matplotlib import ticker
from pandas.plotting import andrews_curves
from datetime import *
import sklearn

from pylab import rcParams
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score


def GetInfo(df):
    pd.set_option('display.max_columns', None)
    print(f"Number of patients: {(df['PatientID'].value_counts()).count()}")
    print(df.info())
    print(df.dtypes)
    print(df.isna().sum())
    print(df.head())


def ShowICDcodeHolder(df):
    total_records = len(df.index)
    labels = "ICD9Code", "ICD10Codse", 'NoCode'
    sizes = [df['ICD9Code'].isna().sum(), df['ICD10Code'].isna().sum(),
             total_records - df['ICD9Code'].isna().sum() - df['ICD10Code'].isna().sum()]
    explode = (0.1, 0.1, 0)  # only "explode" the 2nd slice
    pyplot_show(sizes, labels, explode, "Code holders")


def pyplot_show(sizes, labels, explode, title):
    data = sizes

    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))

    ingredients = [x.split()[-1] for x in labels]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n{:d} ".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    ax.legend(wedges, ingredients,

              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('../plot/T2D_percentage.jpg', dpi=300)


def ShowHistICD9Values(diag):
    f1 = plt.figure()
    (diag['ICD9Code'].value_counts().head(10) / diag['ICD9Code'].count() * 100).plot(kind='bar',
                                                                                     figsize=(7, 6), rot=0);
    plt.ylabel("Percentage", labelpad=14)
    plt.xticks(rotation=90)
    plt.title("Percentage of ICD9 Code", y=1.02);
    plt.legend()
    f1.show()


def ShowHistICD10Values(diag):
    f1 = plt.figure()
    (diag['ICD10Code'].value_counts().head(10) / diag['ICD10Code'].count() * 100).plot(kind='bar',
                                                                                       figsize=(7, 6), rot=0);
    plt.ylabel("Percentage", labelpad=14)
    plt.title("Percentage of ICD10 Code", y=1.02);
    plt.legend()
    plt.xticks(rotation=90)
    f1.show()


def IdentifyT2DPatients(diag, ICD9_code_list, ICD10_code_list, age_limit):
    import re
    # --- filter out diag code after age limit; need to be multiply by 12 to get month
    diag = diag[diag['age'] <= age_limit * 12]

    diag.loc[(diag['ICD9Code'].str.match(ICD9_code_list, na=False)) | (diag['ICD10Code'].str.match(ICD10_code_list, na=False)), 'T2D'] = 1
    diag.loc[diag['T2D'].isna(), 'T2D'] = 0
    diag['ContactDate'] = pd.to_datetime(diag['ContactDate'], format='%Y-%m-%d %H:%M:%S.%f')
    diag['T2Ddate'] = diag.loc[diag['T2D'] == 1, 'ContactDate']
    # diag['T2Ddate'] = diag.apply(lambda x: x['ContactDate'] if x['T2D'] == 1 else "", axis=1)
    diag['T2DFristDetectionDate'] = diag.groupby('PatientID')['T2Ddate'].transform('min')

    diag = diag.join(diag.groupby('PatientID')['T2D'].max(), on='PatientID', rsuffix='_r')
    del diag['T2D']
    diag = diag.rename(columns={'T2D_r': 'T2D'})
    # diag['T2D'] = diag.groupby('PatientID').agg({'T2D': 'max'})
    print(
        f"T2D:{diag[diag['T2D'] == 1]['PatientID'].nunique()}, non-T2: {diag[diag['T2D'] == 0]['PatientID'].nunique()}")

    # T2D_detected = T2D_detected.rename(columns={'ContactDate': 'T2Ddate'})
    # T2D_detected['T2Ddate'] = pd.to_datetime(T2D_detected['T2Ddate'], format='%Y-%m-%d %H:%M:%S.%f')
    # T2D_detected['T2DFristDetectionDate'] = T2D_detected.groupby('PatientID')['T2Ddate'].transform('min')
    del diag['T2Ddate']
    diag.to_csv("../Output/Diag_T2D_labeled.csv", index=False)

    all_patients = diag['PatientID'].nunique()
    T2D_patients = diag.loc[diag['T2D'] == 1, 'PatientID'].nunique()
    # T2D_PatientID = T2D_detected_copy['PatientID'].unique()
    print(f"Number of Type-II Diabetes patients: {T2D_patients} out of {all_patients}")
    # T2D = diag.assign(Keep=diag.PatientID.isin(T2D_PatientID).astype(int))
    # nonT2D = T2D[T2D['Keep'] == 0]
    # T2D = T2D[T2D['Keep'] == 1]
    # del T2D['Keep']
    # nonT2D.to_csv("../Input/nonT2D_patients.csv", index=False)
    # T2D.to_csv("../Input/T2D_patients_all_records.csv", index=False)

    labels = "Non-T2D", "T2D"
    sizes = [all_patients-T2D_patients, T2D_patients]
    explode = (0, 0)  # only "explode" the 2nd slice
    pyplot_show(sizes, labels, explode, "Percentage of T2D patients")


def LastDiagDateFind(diag):
    # for i in range(len(T2D_detected)):
    #     id = T2D_detected.iloc[i]["PatientID"]
    #     records = diag[diag["PatientID"] == id]["ContactDate"]
    #     records= pd.to_datetime(records, format='%Y-%m-%d %H:%M:%S.%f')
    #     T2D_detected.loc[T2D_detected['PatientID'] == id, 'LastDiagDate'] = max(records)
    #     # T2D_detected.at[i, "LastDiagDate"] = max(records['ContactDate'])
    # T2D_detected["LastDiagDate"]= T2D_detected["LastDiagDate"].dt.date

    diag.groupby("PatientID")['ContactDate'].transform('max')
    return T2D_detected


def TakeChildrenFromDiagnosisFile():
    # Take out childern from extended diag
    diag = pd.read_table('../Input/ExtendedCohort/CHObesity_Diagnoses.txt', converters={0: str})
    BMI = pd.read_csv('../Output/BMI_data.csv', delimiter=',',
                      dtype={'PatientID': np.str, 'GenderDescription': np.str, 'age': np.int32, 'BMI': np.float64})
    BMI_pat = BMI.PatientID.unique()
    BMI_pat_len = BMI.PatientID.nunique()
    diag_pat_len = diag.PatientID.nunique()
    diag_pat = diag.PatientID.unique()
    union = pd.Series(np.union1d(BMI_pat, diag_pat))
    intersect = pd.Series(np.intersect1d(BMI_pat, diag_pat))
    print(f"non commons BMI&diagE = {len(union[~union.isin(intersect)])}")
    diagE = diag.assign(Keep=diag.PatientID.isin(BMI_pat).astype(int))
    print(len(diagE[diagE['Keep'] == 1]), len(diagE[diagE['Keep'] == 0]), diagE['PatientID'].nunique())
    diagE = diagE[diagE['Keep'] == 1]
    del diagE['Keep']
    diagE.to_csv("../Input/Diagnosis_Clipped.csv", index=False)


def GetICDCodeList(filename):
    codes_guide = pd.read_table(filename, converters={0: str, 1: str})
    ICD_code_list = codes_guide['code'].values
    icd_codes = ""
    for i in ICD_code_list:
        i = i.replace('.', '\.')
        icd_codes += '\\b' + i + '\\b|'
    return icd_codes[:-1]


def ReadDiagnosesDataAndFilter():
    diag = pd.read_csv("../Input/Diagnosis_Clipped.csv", delimiter=',', dtype={'PatientID': np.str})
    print(f"Number of Orignal Patients: {diag['PatientID'].nunique()}")
    # ----remove no date records
    diag = diag[diag['ContactDate'].isna() == 0]
    # ----remove did not show up records
    diag = diag[diag['ConceptID'] != 'C2051454']
    # ----remove ContactDate > 2020-10-1
    diag['ContactDate'] = pd.to_datetime(diag['ContactDate'], format='%Y-%m-%d %H:%M:%S.%f')
    diag = diag[diag['ContactDate'] < datetime(2020, 10, 1)]
    print(f"Number of Patients after cleaning: {diag['PatientID'].nunique()}")
    diag.to_csv("../Input/Diagnoses_Clipped_Filtered.csv", index=False)


def ReadCategoriesCode():
    file_name = "../Input/comorbidity.xlsx"
    file = pd.read_excel("../Input/comorbidity.xlsx", sheet_name=None)

    # xls = pd.ExcelFile('../Input/comorbidity.xlsx')
    # categories = {}
    #
    # for cat in xls.sheet_names:
    #     df = pd.read_excel(file_name, cat)
    #     categories[cat] = df

    return file


def LookForHavingICD9_10(icd9, icd10, patient):
    count = \
    patient[patient['ICD9Code'].str.startswith(icd9, na=False) | patient['ICD10Code'].str.startswith(icd10, na=False)][
        'PatientID'].nunique()
    if icd10[0] == 'E10':
        data = patient[
            patient['ICD9Code'].str.startswith(icd9, na=False) | patient['ICD10Code'].str.startswith(icd10, na=False)]
        size = pd.concat([data['ICD9Code'].value_counts(), data['ICD10Code'].value_counts()]).sort_values(
            ascending=False).sum()
        pd.concat([data['ICD9Code'].value_counts() / size, data['ICD10Code'].value_counts() / size]).sort_values(
            ascending=False).plot(kind='bar')
        plt.show()
    return count


def ComorbidityAnalysis(patient):
    code_categories = pd.read_excel("../Input/comorbidity.xlsx", sheet_name=None,
                                    converters={'ICD9': str, 'ICD10': str})
    detected = {}
    for key, val in code_categories.items():
        icd10 = tuple(val['ICD10'].dropna().values)
        icd9 = tuple(val['ICD9'].dropna().values)
        detected[key] = LookForHavingICD9_10(icd9, icd10, patient)

    out = pd.DataFrame.from_dict(detected, orient='index')
    out = out[0].sort_values()
    return out


def get_comorbidities(file):
    diag = pd.read_csv(file, delimiter=',', dtype={'PatientID': np.str})
    Comor_data = ComorbidityAnalysis(diag.loc[diag['T2D'] == 1])
    Comor_data_non = ComorbidityAnalysis(diag.loc[diag['T2D'] == 0])
    Comor_data = pd.DataFrame(Comor_data)
    Comor_data = Comor_data.rename(columns={0: 'T2D'})
    Comor_data['Non-T2D'] = Comor_data_non
    T2D_len = diag.loc[diag['T2D'] == 1, 'PatientID'].nunique()
    nonT2D_len = diag.loc[diag['T2D'] == 0, 'PatientID'].nunique()
    Comor_data['T2D'] = Comor_data['T2D'].apply(lambda x: x * 100 / T2D_len)
    Comor_data['Non-T2D'] = Comor_data['Non-T2D'].apply(lambda x: x * 100 / nonT2D_len)
    ax = Comor_data.plot.barh(rot=0)
    plt.xlabel("Percentage")
    plt.title("T2D Comorbidities")
    plt.tight_layout()
    plt.legend()
    plt.savefig('../plot/comorbidity.jpg', dpi=300)


def results_summary_to_dataframe(results, col, pos_patients, code_desc):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]
    bse = results.bse

    results_df = pd.DataFrame({"code": col,
                               "description": code_desc,
                               "Positive": pos_patients,
                               "pvals": pvals,
                               "coeff": coeff,
                               "conf_lower": conf_lower,
                               "conf_higher": conf_higher,
                               "bse": bse
                               })
    return results_df


def GetLogisticRegression(bmi, diag, being_patient_list, age_limit = 0):
    import time

    # bmi = pd.read_csv(bmi_file, delimiter=',', dtype={'PatientID': np.str})
    # diag = pd.read_csv(diag_file, delimiter=',', dtype={'PatientID': np.str})

    patients_id = bmi['PatientID'].unique()
    icd9 = diag['ICD9Code'].dropna().unique()
    icd10 = diag['ICD10Code'].dropna().unique()
    if age_limit > 0:
        diag = diag[diag['age'] <= (age_limit * 12)]
    for col in ('ICD9Code', 'ICD10Code'):
        if col == 'ICD9Code':
            codes = icd9
        else:
            codes = icd10
        results = pd.DataFrame()
        for i, code in enumerate(codes):
            s_bmi = bmi.copy()
            s_diag = diag[diag[col] == code][['PatientID', 'ConceptDescription', 'age']]

            s_diag['FristDetectionDate'] = s_diag.groupby('PatientID')['age'].transform('min')
            s_diag=s_diag.groupby('PatientID').head(1)
            s = s_diag.set_index('PatientID')['FristDetectionDate']
            s_bmi['FristDetectionDate'] = s_bmi['PatientID'].map(s)
            s_bmi.loc[s_bmi['FristDetectionDate'].isna(),'FristDetectionDate'] = sys.maxsize
            s_bmi = s_bmi[s_bmi['age'] <= s_bmi['FristDetectionDate']]

            having_code = s_bmi.loc[s_bmi['FristDetectionDate'] < sys.maxsize, 'PatientID'].unique()
            if len(having_code) > 10:
                s_bmi['Class'] = np.where(s_bmi['PatientID'].isin(having_code), 1, 0)
                s_bmi = s_bmi[((s_bmi['Class']==0) & (s_bmi['PatientID'].isin(being_patient_list['0']))) | (s_bmi['Class']==1)]
                idx = s_bmi.groupby(['PatientID'])['Percentile'].transform(max) == s_bmi['Percentile']
                s_bmi = s_bmi[idx][['PatientID', 'Percentile', 'Class']]
                s_bmi.drop_duplicates(inplace=True)
                result = LogReg_R_function(s_bmi[['Percentile', 'Class']], code)
                results_df = pd.DataFrame({"code": code,
                                           "description": s_diag.head(1)['ConceptDescription'].values,
                                           "Positive": len(having_code),
                                           # "Intercept-Estimate": result[0][0],
                                           # "Intercept-Std. Error": result[0][1],
                                           # "Intercept-z value": result[0][2],
                                           # "Intercept-Pr(>|z|": result[0][3],
                                           "Percentile-Estimate": result[1][0],
                                           "Percentile-Std. Error": result[1][1],
                                           "Percentile-z value": result[1][2],
                                           "Percentile-Pr(>|z|": result[1][3]
                                           })
            else:
                results_df = pd.DataFrame({"code": code,
                                           "description": s_diag.head(1)['ConceptDescription'].values,
                                           "Positive": s_diag['PatientID'].nunique(),
                                           # "Intercept-Estimate": result[0][0],
                                           # "Intercept-Std. Error": result[0][1],
                                           # "Intercept-z value": result[0][2],
                                           # "Intercept-Pr(>|z|": result[0][3],
                                           "Percentile-Estimate": 1000,
                                           "Percentile-Std. Error": 1000,
                                           "Percentile-z value": 1000,
                                           "Percentile-Pr(>|z|": 1000

                                           })
            results = pd.concat([results, results_df], ignore_index=True)
            print(i)

        results.to_csv("../Output/LogisticReg_results" + str(col) +"_diag_age_limit"+age_limit+ ".csv", index=False)


def LogReg_R_function(df, code):
    import rpy2.robjects as robjects
    from rpy2.robjects import r, pandas2ri
    import scipy.stats as stats
    r['source']('Logistic-Regression.R')
    function_r = robjects.globalenv["LogReg"]  # Reading and processing data
    pandas2ri.activate()
    df_result_r = function_r(df, code)
    return df_result_r

def CreateFinalVersionOfBMI_DiagFilesUSingCPTCodes():
    diag = pd.read_csv("../Input/Diagnoses_Clipped_Filtered.csv", delimiter=',', dtype={'PatientID': np.str})

    proc = pd.read_table("../Input/ExtendedCohort/CHObesity_Procedures.txt", converters={0: str})
    BMI = pd.read_csv('../Output/BMI_data.csv', delimiter=',',
                      dtype={'PatientID': np.str, 'GenderDescription': np.str, 'age': np.int32, 'BMI': np.float64})
    info = pd.read_table('../Input/ExtendedCohort/CHObesity_Demographics.txt', converters={0: str})
    info['DateOfBirth'] = pd.to_datetime(info['DateOfBirth'], format='%Y-%m-%d %H:%M:%S.%f').dt.date

    s = info.set_index('PatientID')['DateOfBirth']
    BMI['DateOfBirth'] = BMI['PatientID'].map(s)

    diag['DateOfBirth'] = diag['PatientID'].map(s)
    proc['DateOfBirth'] = diag['PatientID'].map(s)
    # Calculating age in month
    diag['age'] = (
            (pd.to_datetime(diag['ContactDate'], format='%Y-%m-%d %H:%M:%S.%f') - pd.to_datetime(
                diag.DateOfBirth, format='%Y-%m-%d')).astype('timedelta64[D]') / 30.4375).astype(int)
    proc = proc[(proc['OrderTime'].notnull()) & (proc['CancelationReason'].isnull())]
    proc['age'] = (
            (pd.to_datetime(proc['OrderTime'], format='%Y-%m-%d %H:%M:%S.%f') - pd.to_datetime(
                proc.DateOfBirth, format='%Y-%m-%d')).astype('timedelta64[D]') / 30.4375).astype(int)

    # Calculate the first BMI measurement and the last for each patient
    BMI['FirstRecord'] = BMI.groupby('PatientID')["age"].transform('min')
    BMI['LastRecord'] = BMI.groupby('PatientID')["age"].transform('max')
    BMI_number_of_patients_based_on_last_record = []
    for i in range(18):
        BMI_number_of_patients_based_on_last_record.append(len(BMI[BMI['LastRecord'] > i * 12].groupby('PatientID')))
    plt.bar(range(1, 19), BMI_number_of_patients_based_on_last_record)
    plt.xticks(range(1, 19))
    plt.ylabel('Patients')
    plt.xlabel('Age (year)')
    plt.title('Histogram of ages at last W/H record')
    plt.plt.savefig('../plot/Histogram_of_ages_at_last_W-H_record.jpg', dpi=300)

    # Calculate the first diag measurement and the last for each patient
    diag['FirstRecord'] = diag.groupby('PatientID')["age"].transform('min')
    diag['LastRecord'] = diag.groupby('PatientID')["age"].transform('max')
    diag_number_of_patients_based_on_last_record = []
    for i in range(1, 34):
        diag_number_of_patients_based_on_last_record.append(
            len(diag[(diag['LastRecord'] > i * 12)].groupby('PatientID')))
    plt.bar(range(1, 34), diag_number_of_patients_based_on_last_record)
    plt.xticks(range(1, 34))
    plt.ylabel('Patients')
    plt.xlabel('Age (year)')
    plt.title('Histogram of ages at last diagnosis record')
    plt.savefig('../plot/Histogram_of_ages_at_last_diagnosis_record.jpg', dpi=300)

    proc['LastRecord'] = proc.groupby('PatientID')["age"].transform('max')

    proc = proc.assign(Keep=proc.PatientID.isin(BMI['PatientID'].unique()).astype(int))
    proc = proc[proc['Keep'] == 1]
    del proc['Keep']
    being_patient_list = np.union1d(diag[(diag['LastRecord'] > 204)]['PatientID'].unique(), BMI[BMI['LastRecord'] > 204]['PatientID'].unique())
    being_patient_list = pd.Series(np.union1d(being_patient_list, proc[proc['LastRecord'] > 204]['PatientID'].unique()))

    being_patient_list.to_csv("../Output/being_patient_list.csv", index=False)
    BMI.to_csv("../Output/final_BMI_data.csv", index=False)
    diag.to_csv("../Output/final_diag.csv", index=False)
    # ----Note: No need to filterout patient who stop being patient here; Do it after get regression

    # filtered_BMI = BMI.assign(Keep=BMI.PatientID.isin(being_patient_list).astype(int))
    # filtered_BMI = filtered_BMI[filtered_BMI['Keep'] == 1]
    # del filtered_BMI['Keep']
    # final_diag = diag.assign(Keep=diag.PatientID.isin(being_patient_list).astype(int))
    # final_diag = final_diag[final_diag['Keep'] == 1]
    # del final_diag['Keep']
    #
    # filtered_BMI.to_csv("../Output/final_BMI_data.csv", index=False)
    # final_diag.to_csv("../Output/final_diag.csv", index=False)
    return BMI, diag

def main():
    # ----Step 1: Clip and filter diagnoses file----
    # TakeChildrenFromDiagnosisFile()
    # ReadDiagnosesDataAndFilter()
    # bmi, diag = CreateFinalVersionOfBMI_DiagFilesUSingCPTCodes()
    being_patient_list = pd.read_csv("../Output/being_patient_list.csv", dtype=np.str)
    diag = pd.read_csv("../Output/final_diag.csv", delimiter=',', dtype={'PatientID': np.str})
    BMI = pd.read_csv('../Output/final_BMI_data.csv', delimiter=',',
                      dtype={'PatientID': np.str, 'GenderDescription': np.str, 'age': np.int32, 'BMI': np.float64})
    age_limit = 20
    GetLogisticRegression(BMI, diag, being_patient_list, age_limit)
    return
    # diag = pd.read_csv("../Output/final_diag.csv", delimiter=',', dtype={'PatientID': np.str})
    # icd9_codes = GetICDCodeList('../Input/ICD9Code_T2D.txt')
    # icd10_codes = GetICDCodeList('../Input/ICD10Code_T2D.txt')
    # IdentifyT2DPatients(diag, icd9_codes, icd10_codes, age_limit)

    # Taking birthday from demographics file and add it to BMI and diag files
    diag = pd.read_csv("../Output/Diag_T2D_labeled.csv", delimiter=',', dtype={'PatientID': np.str})
    BMI = pd.read_csv('../Output/final_BMI_data.csv', delimiter=',',
                      dtype={'PatientID': np.str, 'GenderDescription': np.str, 'age': np.int32, 'BMI': np.float64})

    diag = diag
    # Insert T2D fisrt detection date from diag file to BMI file
    diag['DateOfBirth'] = pd.to_datetime(diag[diag['DateOfBirth'].notna()]['DateOfBirth'], format='%Y-%m-%d %H:%M:%S.%f').dt.date
    diag['T2DFristDetectionDate'] = pd.to_datetime(diag[diag['T2DFristDetectionDate'].notna()]['T2DFristDetectionDate'], format='%Y-%m-%d %H:%M:%S.%f').dt.date
    s = diag.groupby('PatientID').head(1).set_index('PatientID')['T2DFristDetectionDate']
    BMI['T2DFristDetectionDate'] = BMI['PatientID'].map(s)
    BMI.loc[BMI['T2DFristDetectionDate'].isna() == 0, 'T2D'] = 1

    # ------ first T2D diagnosis date distribution-------
    diag['T2DDetectionAge'] = diag[diag['T2D'] == 1].eval('T2DFristDetectionDate - DateOfBirth').astype('timedelta64[Y]')
    diag[diag['T2D'] == 1].groupby('PatientID').head(1)['T2DDetectionAge'].hist(bins=32)
    plt.ylabel('Count')
    plt.xlabel('Age (year)')
    plt.title('Histogram of ages at T2D diagnosis')
    plt.plt.savefig('../plot/first_T2D_diagnosis_date_distribution.jpg', dpi=300)

    BMI.loc[filtered_BMI['T2D'].isna(), 'T2D'] = 0
    print(f"Number of final BMI Patients: {filtered_BMI['PatientID'].nunique()}, Records: {len(filtered_BMI)}")
    print(
        f"T2D Patients: {filtered_BMI[BMI['T2D'] == 1]['PatientID'].nunique()}, non-T2D: {BMI[BMI['T2D'] == 0]['PatientID'].nunique()}")
    BMI.to_csv("../Output/final_BMI_data_labeled.csv", index=False)
    diag.to_csv("../Output/final_diag_labeled.csv", index=False)

    get_comorbidities("../Output/final_diag_labeled.csv")


if __name__ == "__main__":
    main()
