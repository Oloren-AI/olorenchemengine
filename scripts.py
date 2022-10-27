from tdc.benchmark_group import admet_group
from tqdm import tqdm 
import olorenchemengine as oce
import numpy as np
import pandas as pd
import psycopg2

def get_db_connection():
    print("Connecting to database")
    conn = psycopg2.connect(host="containers-us-west-37.railway.app", password="SKczx0lCpYCEy0l85wfx", user="postgres", database="railway", port="7202")
    print("Opened database successfully")
    return conn

ALL_TDC_BENCHMARKS_DIR = {
    'Caco2_Wang':"MIN",
    'Bioavailability_Ma':"MAX",
    'Lipophilicity_AstraZeneca':"MIN",
    "Solubility_AqSolDB":"MIN",
    'HIA_Hou':"MAX",
    'Pgp_Broccatelli':"MAX",
    'BBB_Martins':"MAX",
    'PPBR_AZ':"MIN",
    "VDss_Lombardo":"MAX",
    'CYP2C9_Veith':"MAX",
    'CYP2D6_Veith':"MAX",
    'CYP3A4_Veith':"MAX",
    'CYP2C9_Substrate_CarbonMangels':"MAX",
    'CYP2D6_Substrate_CarbonMangels':"MAX",
    'CYP3A4_Substrate_CarbonMangels':"MAX",
    'Half_Life_Obach':"MAX",
    'Clearance_Hepatocyte_AZ':"MAX",
    'Clearance_Microsome_AZ':"MAX",
    'LD50_Zhu':"MIN",
    'hERG':"MAX",
    'AMES':"MAX",
    'DILI':"MAX",
}

ALL_MOLNET_BENCHMARKS_DIR = {
    "bace_classification": "MAX",
    "bbbp": "MAX",
    "clintox": "MAX",
    "hiv": "MAX",
    "muv": "MAX",
    "pcba": "MAX",
    "sider": "MAX",
    "tox21": "MAX",
    "toxcast": "MAX",
    "delaney_esol": "MIN",
    "freesolv": "MIN",
    "lipo": "MIN",
    "bace_regression": "MIN"
}

# MOLNET BENCHMARKS

datasets_loaded = {
    "bace_classification": "MoleculeNet/load_bace_classification.csv",
    "bbbp": "MoleculeNet/load_bbbp.csv",
    "clintox": "MoleculeNet/load_clintox.csv",
    "hiv": "MoleculeNet/load_hiv.csv",
    "muv": "MoleculeNet/load_muv.csv",
    "pcba": "MoleculeNet/load_pcba.csv",
    "sider": "MoleculeNet/load_sider.csv",
    "tox21": "MoleculeNet/load_tox21.csv",
    "toxcast": "MoleculeNet/load_toxcast.csv",
    "delaney_esol": "MoleculeNet/load_delaney.csv",
    "freesolv": "MoleculeNet/load_freesolv.csv",
    "lipo": "MoleculeNet/load_lipo.csv",
    "bace_regression": "MoleculeNet/load_bace_regression.csv"
}

datasets_types = {
    "bace_classification": "classification",
    "bbbp": "classification",
    "clintox": "classification",
    "hiv": "classification",
    "muv": "classification",
    "pcba": "classification",
    "sider": "classification",
    "tox21": "classification",
    "toxcast": "classification",
    "delaney_esol": "regression",
    "freesolv": "regression",
    "lipo": "regression",
    "bace_regression": "regression"
}

dataset_property_cols = {
    "bace_classification": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED","CT_TOX"],
    "hiv": ["HIV_active"],
    "muv": ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713","MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"],
    "pcba": ["PCBA-1030","PCBA-1379","PCBA-1452","PCBA-1454","PCBA-1457","PCBA-1458","PCBA-1460","PCBA-1461","PCBA-1468","PCBA-1469","PCBA-1471","PCBA-1479","PCBA-1631","PCBA-1634","PCBA-1688","PCBA-1721","PCBA-2100","PCBA-2101","PCBA-2147","PCBA-2242","PCBA-2326","PCBA-2451","PCBA-2517","PCBA-2528","PCBA-2546","PCBA-2549","PCBA-2551","PCBA-2662","PCBA-2675","PCBA-2676","PCBA-411","PCBA-463254","PCBA-485281","PCBA-485290","PCBA-485294","PCBA-485297","PCBA-485313","PCBA-485314","PCBA-485341","PCBA-485349","PCBA-485353","PCBA-485360","PCBA-485364","PCBA-485367","PCBA-492947","PCBA-493208","PCBA-504327","PCBA-504332","PCBA-504333","PCBA-504339","PCBA-504444","PCBA-504466","PCBA-504467","PCBA-504706","PCBA-504842","PCBA-504845","PCBA-504847","PCBA-504891","PCBA-540276","PCBA-540317","PCBA-588342","PCBA-588453","PCBA-588456","PCBA-588579","PCBA-588590","PCBA-588591","PCBA-588795","PCBA-588855","PCBA-602179","PCBA-602233","PCBA-602310","PCBA-602313","PCBA-602332","PCBA-624170","PCBA-624171","PCBA-624173","PCBA-624202","PCBA-624246","PCBA-624287","PCBA-624288","PCBA-624291","PCBA-624296","PCBA-624297","PCBA-624417","PCBA-651635","PCBA-651644","PCBA-651768","PCBA-651965","PCBA-652025","PCBA-652104","PCBA-652105","PCBA-652106","PCBA-686970","PCBA-686978","PCBA-686979","PCBA-720504","PCBA-720532","PCBA-720542","PCBA-720551","PCBA-720553","PCBA-720579","PCBA-720580","PCBA-720707","PCBA-720708","PCBA-720709","PCBA-720711","PCBA-743255","PCBA-743266","PCBA-875","PCBA-881","PCBA-883","PCBA-884","PCBA-885","PCBA-887","PCBA-891","PCBA-899","PCBA-902","PCBA-903","PCBA-904","PCBA-912","PCBA-914","PCBA-915","PCBA-924","PCBA-925","PCBA-926","PCBA-927","PCBA-938","PCBA-995"],
    "sider": ["Hepatobiliary disorders","Metabolism and nutrition disorders","Product issues","Eye disorders","Investigations","Musculoskeletal and connective tissue disorders","Gastrointestinal disorders","Social circumstances","Immune system disorders","Reproductive system and breast disorders","Neoplasms benign, malignant and unspecified (incl cysts and polyps)","General disorders and administration site conditions","Endocrine disorders","Surgical and medical procedures","Vascular disorders","Blood and lymphatic system disorders","Skin and subcutaneous tissue disorders","Congenital, familial and genetic disorders","Infections and infestations","Respiratory, thoracic and mediastinal disorders","Psychiatric disorders","Renal and urinary disorders","Pregnancy, puerperium and perinatal conditions","Ear and labyrinth disorders","Cardiac disorders","Nervous system disorders","Injury, poisoning and procedural complications"],
    "tox21": ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"],
    "toxcast": ["ACEA_T47D_80hr_Negative","ACEA_T47D_80hr_Positive","APR_HepG2_CellCycleArrest_24h_dn","APR_HepG2_CellCycleArrest_24h_up","APR_HepG2_CellCycleArrest_72h_dn","APR_HepG2_CellLoss_24h_dn","APR_HepG2_CellLoss_72h_dn","APR_HepG2_MicrotubuleCSK_24h_dn","APR_HepG2_MicrotubuleCSK_24h_up","APR_HepG2_MicrotubuleCSK_72h_dn","APR_HepG2_MicrotubuleCSK_72h_up","APR_HepG2_MitoMass_24h_dn","APR_HepG2_MitoMass_24h_up","APR_HepG2_MitoMass_72h_dn","APR_HepG2_MitoMass_72h_up","APR_HepG2_MitoMembPot_1h_dn","APR_HepG2_MitoMembPot_24h_dn","APR_HepG2_MitoMembPot_72h_dn","APR_HepG2_MitoticArrest_24h_up","APR_HepG2_MitoticArrest_72h_up","APR_HepG2_NuclearSize_24h_dn","APR_HepG2_NuclearSize_72h_dn","APR_HepG2_NuclearSize_72h_up","APR_HepG2_OxidativeStress_24h_up","APR_HepG2_OxidativeStress_72h_up","APR_HepG2_StressKinase_1h_up","APR_HepG2_StressKinase_24h_up","APR_HepG2_StressKinase_72h_up","APR_HepG2_p53Act_24h_up","APR_HepG2_p53Act_72h_up","APR_Hepat_Apoptosis_24hr_up","APR_Hepat_Apoptosis_48hr_up","APR_Hepat_CellLoss_24hr_dn","APR_Hepat_CellLoss_48hr_dn","APR_Hepat_DNADamage_24hr_up","APR_Hepat_DNADamage_48hr_up","APR_Hepat_DNATexture_24hr_up","APR_Hepat_DNATexture_48hr_up","APR_Hepat_MitoFxnI_1hr_dn","APR_Hepat_MitoFxnI_24hr_dn","APR_Hepat_MitoFxnI_48hr_dn","APR_Hepat_NuclearSize_24hr_dn","APR_Hepat_NuclearSize_48hr_dn","APR_Hepat_Steatosis_24hr_up","APR_Hepat_Steatosis_48hr_up","ATG_AP_1_CIS_dn","ATG_AP_1_CIS_up","ATG_AP_2_CIS_dn","ATG_AP_2_CIS_up","ATG_AR_TRANS_dn","ATG_AR_TRANS_up","ATG_Ahr_CIS_dn","ATG_Ahr_CIS_up","ATG_BRE_CIS_dn","ATG_BRE_CIS_up","ATG_CAR_TRANS_dn","ATG_CAR_TRANS_up","ATG_CMV_CIS_dn","ATG_CMV_CIS_up","ATG_CRE_CIS_dn","ATG_CRE_CIS_up","ATG_C_EBP_CIS_dn","ATG_C_EBP_CIS_up","ATG_DR4_LXR_CIS_dn","ATG_DR4_LXR_CIS_up","ATG_DR5_CIS_dn","ATG_DR5_CIS_up","ATG_E2F_CIS_dn","ATG_E2F_CIS_up","ATG_EGR_CIS_up","ATG_ERE_CIS_dn","ATG_ERE_CIS_up","ATG_ERRa_TRANS_dn","ATG_ERRg_TRANS_dn","ATG_ERRg_TRANS_up","ATG_ERa_TRANS_up","ATG_E_Box_CIS_dn","ATG_E_Box_CIS_up","ATG_Ets_CIS_dn","ATG_Ets_CIS_up","ATG_FXR_TRANS_up","ATG_FoxA2_CIS_dn","ATG_FoxA2_CIS_up","ATG_FoxO_CIS_dn","ATG_FoxO_CIS_up","ATG_GAL4_TRANS_dn","ATG_GATA_CIS_dn","ATG_GATA_CIS_up","ATG_GLI_CIS_dn","ATG_GLI_CIS_up","ATG_GRE_CIS_dn","ATG_GRE_CIS_up","ATG_GR_TRANS_dn","ATG_GR_TRANS_up","ATG_HIF1a_CIS_dn","ATG_HIF1a_CIS_up","ATG_HNF4a_TRANS_dn","ATG_HNF4a_TRANS_up","ATG_HNF6_CIS_dn","ATG_HNF6_CIS_up","ATG_HSE_CIS_dn","ATG_HSE_CIS_up","ATG_IR1_CIS_dn","ATG_IR1_CIS_up","ATG_ISRE_CIS_dn","ATG_ISRE_CIS_up","ATG_LXRa_TRANS_dn","ATG_LXRa_TRANS_up","ATG_LXRb_TRANS_dn","ATG_LXRb_TRANS_up","ATG_MRE_CIS_up","ATG_M_06_TRANS_up","ATG_M_19_CIS_dn","ATG_M_19_TRANS_dn","ATG_M_19_TRANS_up","ATG_M_32_CIS_dn","ATG_M_32_CIS_up","ATG_M_32_TRANS_dn","ATG_M_32_TRANS_up","ATG_M_61_TRANS_up","ATG_Myb_CIS_dn","ATG_Myb_CIS_up","ATG_Myc_CIS_dn","ATG_Myc_CIS_up","ATG_NFI_CIS_dn","ATG_NFI_CIS_up","ATG_NF_kB_CIS_dn","ATG_NF_kB_CIS_up","ATG_NRF1_CIS_dn","ATG_NRF1_CIS_up","ATG_NRF2_ARE_CIS_dn","ATG_NRF2_ARE_CIS_up","ATG_NURR1_TRANS_dn","ATG_NURR1_TRANS_up","ATG_Oct_MLP_CIS_dn","ATG_Oct_MLP_CIS_up","ATG_PBREM_CIS_dn","ATG_PBREM_CIS_up","ATG_PPARa_TRANS_dn","ATG_PPARa_TRANS_up","ATG_PPARd_TRANS_up","ATG_PPARg_TRANS_up","ATG_PPRE_CIS_dn","ATG_PPRE_CIS_up","ATG_PXRE_CIS_dn","ATG_PXRE_CIS_up","ATG_PXR_TRANS_dn","ATG_PXR_TRANS_up","ATG_Pax6_CIS_up","ATG_RARa_TRANS_dn","ATG_RARa_TRANS_up","ATG_RARb_TRANS_dn","ATG_RARb_TRANS_up","ATG_RARg_TRANS_dn","ATG_RARg_TRANS_up","ATG_RORE_CIS_dn","ATG_RORE_CIS_up","ATG_RORb_TRANS_dn","ATG_RORg_TRANS_dn","ATG_RORg_TRANS_up","ATG_RXRa_TRANS_dn","ATG_RXRa_TRANS_up","ATG_RXRb_TRANS_dn","ATG_RXRb_TRANS_up","ATG_SREBP_CIS_dn","ATG_SREBP_CIS_up","ATG_STAT3_CIS_dn","ATG_STAT3_CIS_up","ATG_Sox_CIS_dn","ATG_Sox_CIS_up","ATG_Sp1_CIS_dn","ATG_Sp1_CIS_up","ATG_TAL_CIS_dn","ATG_TAL_CIS_up","ATG_TA_CIS_dn","ATG_TA_CIS_up","ATG_TCF_b_cat_CIS_dn","ATG_TCF_b_cat_CIS_up","ATG_TGFb_CIS_dn","ATG_TGFb_CIS_up","ATG_THRa1_TRANS_dn","ATG_THRa1_TRANS_up","ATG_VDRE_CIS_dn","ATG_VDRE_CIS_up","ATG_VDR_TRANS_dn","ATG_VDR_TRANS_up","ATG_XTT_Cytotoxicity_up","ATG_Xbp1_CIS_dn","ATG_Xbp1_CIS_up","ATG_p53_CIS_dn","ATG_p53_CIS_up","BSK_3C_Eselectin_down","BSK_3C_HLADR_down","BSK_3C_ICAM1_down","BSK_3C_IL8_down","BSK_3C_MCP1_down","BSK_3C_MIG_down","BSK_3C_Proliferation_down","BSK_3C_SRB_down","BSK_3C_Thrombomodulin_down","BSK_3C_Thrombomodulin_up","BSK_3C_TissueFactor_down","BSK_3C_TissueFactor_up","BSK_3C_VCAM1_down","BSK_3C_Vis_down","BSK_3C_uPAR_down","BSK_4H_Eotaxin3_down","BSK_4H_MCP1_down","BSK_4H_Pselectin_down","BSK_4H_Pselectin_up","BSK_4H_SRB_down","BSK_4H_VCAM1_down","BSK_4H_VEGFRII_down","BSK_4H_uPAR_down","BSK_4H_uPAR_up","BSK_BE3C_HLADR_down","BSK_BE3C_IL1a_down","BSK_BE3C_IP10_down","BSK_BE3C_MIG_down","BSK_BE3C_MMP1_down","BSK_BE3C_MMP1_up","BSK_BE3C_PAI1_down","BSK_BE3C_SRB_down","BSK_BE3C_TGFb1_down","BSK_BE3C_tPA_down","BSK_BE3C_uPAR_down","BSK_BE3C_uPAR_up","BSK_BE3C_uPA_down","BSK_CASM3C_HLADR_down","BSK_CASM3C_IL6_down","BSK_CASM3C_IL6_up","BSK_CASM3C_IL8_down","BSK_CASM3C_LDLR_down","BSK_CASM3C_LDLR_up","BSK_CASM3C_MCP1_down","BSK_CASM3C_MCP1_up","BSK_CASM3C_MCSF_down","BSK_CASM3C_MCSF_up","BSK_CASM3C_MIG_down","BSK_CASM3C_Proliferation_down","BSK_CASM3C_Proliferation_up","BSK_CASM3C_SAA_down","BSK_CASM3C_SAA_up","BSK_CASM3C_SRB_down","BSK_CASM3C_Thrombomodulin_down","BSK_CASM3C_Thrombomodulin_up","BSK_CASM3C_TissueFactor_down","BSK_CASM3C_VCAM1_down","BSK_CASM3C_VCAM1_up","BSK_CASM3C_uPAR_down","BSK_CASM3C_uPAR_up","BSK_KF3CT_ICAM1_down","BSK_KF3CT_IL1a_down","BSK_KF3CT_IP10_down","BSK_KF3CT_IP10_up","BSK_KF3CT_MCP1_down","BSK_KF3CT_MCP1_up","BSK_KF3CT_MMP9_down","BSK_KF3CT_SRB_down","BSK_KF3CT_TGFb1_down","BSK_KF3CT_TIMP2_down","BSK_KF3CT_uPA_down","BSK_LPS_CD40_down","BSK_LPS_Eselectin_down","BSK_LPS_Eselectin_up","BSK_LPS_IL1a_down","BSK_LPS_IL1a_up","BSK_LPS_IL8_down","BSK_LPS_IL8_up","BSK_LPS_MCP1_down","BSK_LPS_MCSF_down","BSK_LPS_PGE2_down","BSK_LPS_PGE2_up","BSK_LPS_SRB_down","BSK_LPS_TNFa_down","BSK_LPS_TNFa_up","BSK_LPS_TissueFactor_down","BSK_LPS_TissueFactor_up","BSK_LPS_VCAM1_down","BSK_SAg_CD38_down","BSK_SAg_CD40_down","BSK_SAg_CD69_down","BSK_SAg_Eselectin_down","BSK_SAg_Eselectin_up","BSK_SAg_IL8_down","BSK_SAg_IL8_up","BSK_SAg_MCP1_down","BSK_SAg_MIG_down","BSK_SAg_PBMCCytotoxicity_down","BSK_SAg_PBMCCytotoxicity_up","BSK_SAg_Proliferation_down","BSK_SAg_SRB_down","BSK_hDFCGF_CollagenIII_down","BSK_hDFCGF_EGFR_down","BSK_hDFCGF_EGFR_up","BSK_hDFCGF_IL8_down","BSK_hDFCGF_IP10_down","BSK_hDFCGF_MCSF_down","BSK_hDFCGF_MIG_down","BSK_hDFCGF_MMP1_down","BSK_hDFCGF_MMP1_up","BSK_hDFCGF_PAI1_down","BSK_hDFCGF_Proliferation_down","BSK_hDFCGF_SRB_down","BSK_hDFCGF_TIMP1_down","BSK_hDFCGF_VCAM1_down","CEETOX_H295R_11DCORT_dn","CEETOX_H295R_ANDR_dn","CEETOX_H295R_CORTISOL_dn","CEETOX_H295R_DOC_dn","CEETOX_H295R_DOC_up","CEETOX_H295R_ESTRADIOL_dn","CEETOX_H295R_ESTRADIOL_up","CEETOX_H295R_ESTRONE_dn","CEETOX_H295R_ESTRONE_up","CEETOX_H295R_OHPREG_up","CEETOX_H295R_OHPROG_dn","CEETOX_H295R_OHPROG_up","CEETOX_H295R_PROG_up","CEETOX_H295R_TESTO_dn","CLD_ABCB1_48hr","CLD_ABCG2_48hr","CLD_CYP1A1_24hr","CLD_CYP1A1_48hr","CLD_CYP1A1_6hr","CLD_CYP1A2_24hr","CLD_CYP1A2_48hr","CLD_CYP1A2_6hr","CLD_CYP2B6_24hr","CLD_CYP2B6_48hr","CLD_CYP2B6_6hr","CLD_CYP3A4_24hr","CLD_CYP3A4_48hr","CLD_CYP3A4_6hr","CLD_GSTA2_48hr","CLD_SULT2A_24hr","CLD_SULT2A_48hr","CLD_UGT1A1_24hr","CLD_UGT1A1_48hr","NCCT_HEK293T_CellTiterGLO","NCCT_QuantiLum_inhib_2_dn","NCCT_QuantiLum_inhib_dn","NCCT_TPO_AUR_dn","NCCT_TPO_GUA_dn","NHEERL_ZF_144hpf_TERATOSCORE_up",
                "NVS_ADME_hCYP19A1","NVS_ADME_hCYP1A1","NVS_ADME_hCYP1A2","NVS_ADME_hCYP2A6","NVS_ADME_hCYP2B6","NVS_ADME_hCYP2C19","NVS_ADME_hCYP2C9","NVS_ADME_hCYP2D6","NVS_ADME_hCYP3A4","NVS_ADME_hCYP4F12","NVS_ADME_rCYP2C12","NVS_ENZ_hAChE","NVS_ENZ_hAMPKa1","NVS_ENZ_hAurA","NVS_ENZ_hBACE","NVS_ENZ_hCASP5","NVS_ENZ_hCK1D","NVS_ENZ_hDUSP3","NVS_ENZ_hES","NVS_ENZ_hElastase","NVS_ENZ_hFGFR1","NVS_ENZ_hGSK3b","NVS_ENZ_hMMP1","NVS_ENZ_hMMP13","NVS_ENZ_hMMP2","NVS_ENZ_hMMP3","NVS_ENZ_hMMP7","NVS_ENZ_hMMP9","NVS_ENZ_hPDE10","NVS_ENZ_hPDE4A1","NVS_ENZ_hPDE5","NVS_ENZ_hPI3Ka","NVS_ENZ_hPTEN","NVS_ENZ_hPTPN11","NVS_ENZ_hPTPN12","NVS_ENZ_hPTPN13","NVS_ENZ_hPTPN9","NVS_ENZ_hPTPRC","NVS_ENZ_hSIRT1","NVS_ENZ_hSIRT2","NVS_ENZ_hTrkA","NVS_ENZ_hVEGFR2","NVS_ENZ_oCOX1","NVS_ENZ_oCOX2","NVS_ENZ_rAChE","NVS_ENZ_rCNOS","NVS_ENZ_rMAOAC","NVS_ENZ_rMAOAP","NVS_ENZ_rMAOBC","NVS_ENZ_rMAOBP","NVS_ENZ_rabI2C","NVS_GPCR_bAdoR_NonSelective","NVS_GPCR_bDR_NonSelective","NVS_GPCR_g5HT4","NVS_GPCR_gH2","NVS_GPCR_gLTB4","NVS_GPCR_gLTD4","NVS_GPCR_gMPeripheral_NonSelective","NVS_GPCR_gOpiateK","NVS_GPCR_h5HT2A","NVS_GPCR_h5HT5A","NVS_GPCR_h5HT6","NVS_GPCR_h5HT7","NVS_GPCR_hAT1","NVS_GPCR_hAdoRA1","NVS_GPCR_hAdoRA2a","NVS_GPCR_hAdra2A","NVS_GPCR_hAdra2C","NVS_GPCR_hAdrb1","NVS_GPCR_hAdrb2","NVS_GPCR_hAdrb3","NVS_GPCR_hDRD1","NVS_GPCR_hDRD2s","NVS_GPCR_hDRD4.4","NVS_GPCR_hH1","NVS_GPCR_hLTB4_BLT1","NVS_GPCR_hM1","NVS_GPCR_hM2","NVS_GPCR_hM3","NVS_GPCR_hM4","NVS_GPCR_hNK2","NVS_GPCR_hOpiate_D1","NVS_GPCR_hOpiate_mu","NVS_GPCR_hTXA2","NVS_GPCR_p5HT2C","NVS_GPCR_r5HT1_NonSelective","NVS_GPCR_r5HT_NonSelective","NVS_GPCR_rAdra1B","NVS_GPCR_rAdra1_NonSelective","NVS_GPCR_rAdra2_NonSelective","NVS_GPCR_rAdrb_NonSelective","NVS_GPCR_rNK1","NVS_GPCR_rNK3","NVS_GPCR_rOpiate_NonSelective","NVS_GPCR_rOpiate_NonSelectiveNa","NVS_GPCR_rSST","NVS_GPCR_rTRH","NVS_GPCR_rV1","NVS_GPCR_rabPAF","NVS_GPCR_rmAdra2B","NVS_IC_hKhERGCh","NVS_IC_rCaBTZCHL","NVS_IC_rCaDHPRCh_L","NVS_IC_rNaCh_site2","NVS_LGIC_bGABARa1","NVS_LGIC_h5HT3","NVS_LGIC_hNNR_NBungSens","NVS_LGIC_rGABAR_NonSelective","NVS_LGIC_rNNR_BungSens","NVS_MP_hPBR","NVS_MP_rPBR","NVS_NR_bER","NVS_NR_bPR","NVS_NR_cAR","NVS_NR_hAR","NVS_NR_hCAR_Antagonist","NVS_NR_hER","NVS_NR_hFXR_Agonist","NVS_NR_hFXR_Antagonist","NVS_NR_hGR","NVS_NR_hPPARa","NVS_NR_hPPARg","NVS_NR_hPR","NVS_NR_hPXR","NVS_NR_hRAR_Antagonist","NVS_NR_hRARa_Agonist","NVS_NR_hTRa_Antagonist","NVS_NR_mERa","NVS_NR_rAR","NVS_NR_rMR","NVS_OR_gSIGMA_NonSelective","NVS_TR_gDAT","NVS_TR_hAdoT","NVS_TR_hDAT","NVS_TR_hNET","NVS_TR_hSERT","NVS_TR_rNET","NVS_TR_rSERT","NVS_TR_rVMAT2","OT_AR_ARELUC_AG_1440","OT_AR_ARSRC1_0480","OT_AR_ARSRC1_0960","OT_ER_ERaERa_0480","OT_ER_ERaERa_1440","OT_ER_ERaERb_0480","OT_ER_ERaERb_1440","OT_ER_ERbERb_0480","OT_ER_ERbERb_1440","OT_ERa_EREGFP_0120","OT_ERa_EREGFP_0480","OT_FXR_FXRSRC1_0480","OT_FXR_FXRSRC1_1440","OT_NURR1_NURR1RXRa_0480","OT_NURR1_NURR1RXRa_1440","TOX21_ARE_BLA_Agonist_ch1","TOX21_ARE_BLA_Agonist_ch2",
                "TOX21_ARE_BLA_agonist_ratio","TOX21_ARE_BLA_agonist_viability","TOX21_AR_BLA_Agonist_ch1","TOX21_AR_BLA_Agonist_ch2","TOX21_AR_BLA_Agonist_ratio","TOX21_AR_BLA_Antagonist_ch1","TOX21_AR_BLA_Antagonist_ch2","TOX21_AR_BLA_Antagonist_ratio","TOX21_AR_BLA_Antagonist_viability","TOX21_AR_LUC_MDAKB2_Agonist","TOX21_AR_LUC_MDAKB2_Antagonist","TOX21_AR_LUC_MDAKB2_Antagonist2","TOX21_AhR_LUC_Agonist","TOX21_Aromatase_Inhibition","TOX21_AutoFluor_HEK293_Cell_blue","TOX21_AutoFluor_HEK293_Media_blue","TOX21_AutoFluor_HEPG2_Cell_blue","TOX21_AutoFluor_HEPG2_Cell_green","TOX21_AutoFluor_HEPG2_Media_blue","TOX21_AutoFluor_HEPG2_Media_green","TOX21_ELG1_LUC_Agonist","TOX21_ERa_BLA_Agonist_ch1","TOX21_ERa_BLA_Agonist_ch2","TOX21_ERa_BLA_Agonist_ratio","TOX21_ERa_BLA_Antagonist_ch1","TOX21_ERa_BLA_Antagonist_ch2","TOX21_ERa_BLA_Antagonist_ratio","TOX21_ERa_BLA_Antagonist_viability","TOX21_ERa_LUC_BG1_Agonist","TOX21_ERa_LUC_BG1_Antagonist","TOX21_ESRE_BLA_ch1","TOX21_ESRE_BLA_ch2","TOX21_ESRE_BLA_ratio","TOX21_ESRE_BLA_viability","TOX21_FXR_BLA_Antagonist_ch1","TOX21_FXR_BLA_Antagonist_ch2","TOX21_FXR_BLA_agonist_ch2","TOX21_FXR_BLA_agonist_ratio","TOX21_FXR_BLA_antagonist_ratio","TOX21_FXR_BLA_antagonist_viability","TOX21_GR_BLA_Agonist_ch1","TOX21_GR_BLA_Agonist_ch2","TOX21_GR_BLA_Agonist_ratio","TOX21_GR_BLA_Antagonist_ch2","TOX21_GR_BLA_Antagonist_ratio","TOX21_GR_BLA_Antagonist_viability","TOX21_HSE_BLA_agonist_ch1","TOX21_HSE_BLA_agonist_ch2","TOX21_HSE_BLA_agonist_ratio","TOX21_HSE_BLA_agonist_viability","TOX21_MMP_ratio_down","TOX21_MMP_ratio_up","TOX21_MMP_viability","TOX21_NFkB_BLA_agonist_ch1","TOX21_NFkB_BLA_agonist_ch2","TOX21_NFkB_BLA_agonist_ratio","TOX21_NFkB_BLA_agonist_viability","TOX21_PPARd_BLA_Agonist_viability","TOX21_PPARd_BLA_Antagonist_ch1","TOX21_PPARd_BLA_agonist_ch1","TOX21_PPARd_BLA_agonist_ch2","TOX21_PPARd_BLA_agonist_ratio","TOX21_PPARd_BLA_antagonist_ratio","TOX21_PPARd_BLA_antagonist_viability","TOX21_PPARg_BLA_Agonist_ch1","TOX21_PPARg_BLA_Agonist_ch2","TOX21_PPARg_BLA_Agonist_ratio","TOX21_PPARg_BLA_Antagonist_ch1","TOX21_PPARg_BLA_antagonist_ratio","TOX21_PPARg_BLA_antagonist_viability","TOX21_TR_LUC_GH3_Agonist","TOX21_TR_LUC_GH3_Antagonist","TOX21_VDR_BLA_Agonist_viability","TOX21_VDR_BLA_Antagonist_ch1","TOX21_VDR_BLA_agonist_ch2","TOX21_VDR_BLA_agonist_ratio","TOX21_VDR_BLA_antagonist_ratio","TOX21_VDR_BLA_antagonist_viability","TOX21_p53_BLA_p1_ch1","TOX21_p53_BLA_p1_ch2","TOX21_p53_BLA_p1_ratio","TOX21_p53_BLA_p1_viability","TOX21_p53_BLA_p2_ch1","TOX21_p53_BLA_p2_ch2","TOX21_p53_BLA_p2_ratio","TOX21_p53_BLA_p2_viability","TOX21_p53_BLA_p3_ch1","TOX21_p53_BLA_p3_ch2","TOX21_p53_BLA_p3_ratio","TOX21_p53_BLA_p3_viability","TOX21_p53_BLA_p4_ch1","TOX21_p53_BLA_p4_ch2","TOX21_p53_BLA_p4_ratio","TOX21_p53_BLA_p4_viability","TOX21_p53_BLA_p5_ch1","TOX21_p53_BLA_p5_ch2","TOX21_p53_BLA_p5_ratio","TOX21_p53_BLA_p5_viability","Tanguay_ZF_120hpf_AXIS_up","Tanguay_ZF_120hpf_ActivityScore","Tanguay_ZF_120hpf_BRAI_up","Tanguay_ZF_120hpf_CFIN_up","Tanguay_ZF_120hpf_CIRC_up","Tanguay_ZF_120hpf_EYE_up","Tanguay_ZF_120hpf_JAW_up","Tanguay_ZF_120hpf_MORT_up","Tanguay_ZF_120hpf_OTIC_up","Tanguay_ZF_120hpf_PE_up","Tanguay_ZF_120hpf_PFIN_up","Tanguay_ZF_120hpf_PIG_up","Tanguay_ZF_120hpf_SNOU_up","Tanguay_ZF_120hpf_SOMI_up","Tanguay_ZF_120hpf_SWIM_up","Tanguay_ZF_120hpf_TRUN_up","Tanguay_ZF_120hpf_TR_up","Tanguay_ZF_120hpf_YSE_up"],
    "delaney_esol": ["measured log solubility in mols per litre"],
    "freesolv": ["y"],
    "lipo": ["exp"],
    "bace_regression": ["pIC50"]
}

ALL_MOLNET_BENCHMARKS = {}
for k, v in datasets_loaded.items():
    ALL_MOLNET_BENCHMARKS[k] = (datasets_loaded[k], datasets_types[k], dataset_property_cols[k])

def benchmark_model_molnet(task, model):
    
    connection = get_db_connection()
    
    mp = str(oce.parameterize(model))
    cur = connection.cursor()
    cur.execute("SELECT 1 FROM benchmark_entries WHERE benchmark_name = %s AND model_parameters = %s", (task, mp))
    matching_entries = cur.fetchall()    
    connection.commit()
        
    connection.close()

    if len(list(matching_entries)) > 0:
        return
    
    if not issubclass(type(model), oce.BaseModel):
        model = oce.create_BC(model)
        
    path = oce.download_public_file(ALL_MOLNET_BENCHMARKS[task][0])
    df = pd.read_csv(path)
    
    if 'split' in df.columns:
        df["split"] = df['split'].str.lower()
        
    if "smiles" in df.columns:
        structure_col = "smiles"
    elif "mol" in df.columns:
        structure_col = "mol"
    
    dataset = oce.BaseDataset(data = df.to_csv(), structure_col = structure_col) + oce.CleanStructures() + oce.ScaffoldSplit(split_proportions = [0.8, 0.1, 0.1])
    
    l = list()
    for property_col in tqdm(ALL_MOLNET_BENCHMARKS[task][2]):
        dataset2 = dataset.copy()
        dataset2.property_col = property_col
        dataset2 = dataset2 + oce.CleanStructures()

        
        if len(dataset2.test_dataset[1].unique()) >= 2:
            model.fit(*dataset2.train_dataset)
            results = model.test(*dataset2.test_dataset)

            if ALL_MOLNET_BENCHMARKS[task][1] == 'classification':
                l.append(results['ROC-AUC'])
            elif ALL_MOLNET_BENCHMARKS[task][1]:
                l.append(results['Root Mean Squared Error'])
        else:
            continue
    
    value = np.mean(l)
    stdev = np.std(l)
    
    print(f"{task}: {value} +/- {stdev}")
    
    connection = get_db_connection()
        
    cur = connection.cursor()
    cur.execute("INSERT INTO benchmark_entries (benchmark_name, model_parameters, benchmark_value, benchmark_stdev) VALUES (%s, %s, %s, %s)",
            (task, str(oce.parameterize(model)), value, stdev)
    )
    
    connection.commit()
    
    connection.close()
    
    if ALL_MOLNET_BENCHMARKS[task] == "MIN":
        return value
    elif ALL_MOLNET_BENCHMARKS[task] == "MAX":
        return -1*value

def benchmark_all_molnet(model):
    for task in ALL_MOLNET_BENCHMARKS.keys():
        try:
            benchmark_model_molnet(task, model)
        except Exception as e:
            print(f"Error on {task}: {oce.parameterize(model)}")
            print(e)
            
# TDC BENCHMARKS

ALL_TDC_BENCHMARKS = [
    'Caco2_Wang',
    'Bioavailability_Ma',
    'Lipophilicity_AstraZeneca',
    "Solubility_AqSolDB",
    'HIA_Hou',
    'Pgp_Broccatelli',
    'BBB_Martins',
    'PPBR_AZ',
    "VDss_Lombardo",
    'CYP2C9_Veith',
    'CYP2D6_Veith',
    'CYP3A4_Veith',
    'CYP2C9_Substrate_CarbonMangels',
    'CYP2D6_Substrate_CarbonMangels',
    'CYP3A4_Substrate_CarbonMangels',
    'Half_Life_Obach',
    'Clearance_Hepatocyte_AZ',
    'Clearance_Microsome_AZ',
    'LD50_Zhu',
    'hERG',
    'AMES',
    'DILI',
]

def benchmark_model_tdc(task, model):
    
    connection = get_db_connection()

    mp = str(oce.parameterize(model))
    cur = connection.cursor()
    cur.execute("SELECT 1 FROM benchmark_entries WHERE benchmark_name = %s AND model_parameters = %s", (task, mp))
    matching_entries =  cur.fetchall()
    connection.commit()
        
    connection.close()

    if len(list(matching_entries)) > 0:
        return
    
    if not issubclass(type(model), oce.BaseModel):
        model = oce.create_BC(model)

    group = admet_group(path = 'data/')
    predictions_list = []
    
    for seed in tqdm([1, 2, 3, 4, 5]):
        model = model.copy()
        benchmark = group.get(task) 
        predictions = {}
        name = benchmark['name']
        train, test = benchmark['train_val'], benchmark['test']

        model.fit(train['Drug'], train['Y'])
        y_pred_test = model.predict(test['Drug'])

        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    value = list(results.items())[0][1][0]
    stdev = list(results.items())[0][1][1]
    print(f"{task}: {value} +/- {stdev}")
    
    connection = get_db_connection()
        
    cur = connection.cursor()
    cur.execute("INSERT INTO benchmark_entries (benchmark_name, model_parameters, benchmark_value, benchmark_stdev) VALUES (%s, %s, %s, %s)",
            (task, str(oce.parameterize(model)), value, stdev)
    )
    
    connection.commit()
    
    connection.close()
    
    if ALL_TDC_BENCHMARKS_DIR[task] == "MIN":
        return value
    elif ALL_TDC_BENCHMARKS_DIR[task] == "MAX":
        return -1*value

def benchmark_all_tdc(model):
    for task in ALL_TDC_BENCHMARKS:
        try:
            benchmark_model_tdc(task, model)
        except Exception as e:
            print(f"Error on {task}: {oce.parameterize(model)}")
            print(e)