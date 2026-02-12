#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib

# =====================================================
# 1. GLOBALER SETUP-MODUS (Hier auswählen!)
# =====================================================

# --- MODELL-LISTEN ---
MODELS_ORIGINAL = {
    "Rang_1": "Rang_1_unet_25d_TripleLoss_a0.33_b0.17_bf64_D5_20260121-090819_loss0.0518_val0.0510.keras",
    "Rang_2": "Rang_2_unet_25d_TripleLoss_a0.17_b0.0_bf64_D5_20260121-012804_loss0.0259_val0.0296.keras",
    "Rang_3": "Rang_3_unet_25d_TripleLoss_a0.33_b0.33_bf64_D5_20260121-100752_loss0.0626_val0.0610.keras",
    "Rang_4": "Rang_4_unet_25d_TripleLoss_RESCUE_a0.17_b0.17_bf64_D5_20260123-223753_loss0.0450_val0.0428.keras",
    "Rang_5": "Rang_5_unet_25d_TripleLoss_RESCUE_a0.33_b0.0_bf64_D5_20260123-233434_loss0.0356_val0.0404.keras",
    "Rang_6": "Rang_6_unet_25d_TripleLoss_a0.17_b0.67_bf64_D5_20260121-051812_loss0.0981_val0.0815.keras",
    "Rang_7": "Rang_7_unet_25d_DeepScan_a0.25_b0.0833_bf64_D5_20260127-093131_loss0.0383_val0.0410.keras",
    "Rang_8": "Rang_8_unet_25d_DeepScan_a0.25_b0.0_bf64_D5_20260126-122819_loss0.0304_val0.0353.keras",
    "Rang_9": "Rang_9_unet_25d_TripleLoss_RESCUE_a0.17_b0.5_bf64_D5_20260123-214154_loss0.0832_val0.0685.keras",
    "Rank_10": "Rang_10_unet_25d_DeepScan_a0.25_b0.1667_bf64_D5_20260126-094704_loss0.0461_val0.0468.keras",
}

MODELS_RERUN = {
    "Rank_01": "Rank_01__DeepScan_a0.1667_b0.0_seed42_20260128-234241_loss0.0257_val0.0294.keras",
    "Rank_02": "Rank_02__DeepScan_a0.25_b0.0_seed42_20260129-121519_loss0.0309_val0.0351.keras",
    "Rank_03": "Rank_03__DeepScan_a0.1667_b0.1667_seed42_20260129-010522_loss0.0407_val0.0425.keras",
    "Rank_04": "Rank_04__DeepScan_a0.25_b0.5_seed42_20260129-201142_loss0.0899_val0.0702.keras",
    "Rank_05": "Rank_05__DeepScan_a0.25_b0.0833_seed42_20260129-133505_loss0.0373_val0.0410.keras",
    "Rank_06": "Rank_06__DeepScan_a0.25_b0.3333_seed42_20260129-175254_loss0.0702_val0.0585.keras",
    "Rank_07": "Rank_07__DeepScan_a0.25_b0.25_seed42_20260129-162232_loss0.0504_val0.0527.keras",
    "Rank_08": "Rank_08__DeepScan_a0.25_b0.1667_seed42_20260129-150606_loss0.0476_val0.0469.keras",
    "Rank_09": "Rank_09__DeepScan_a0.3333_b0.1667_seed42_20260129-102303_loss0.0505_val0.0512.keras",
    "Rank_10": "Rank_10__DeepScan_a0.1667_b0.3333_seed42_20260129-023131_loss0.0558_val0.0555.keras",
    "Rank_11": "Rank_11__DeepScan_a0.3333_b0.3333_seed42_20260129-113724_loss0.0708_val0.0616.keras",
    "Rank_12": "Rank_12__DeepScan_a0.5_b0.0_seed42_20260129-160827_loss0.0461_val0.0520.keras",
    "Rank_13": "Rank_13__DeepScan_a0.3333_b0.0_seed42_20260129-090833_loss0.0361_val0.0409.keras",
    "Rank_14": "Rank_14__DeepScan_a0.1667_b0.5_seed42_20260129-035815_loss0.0709_val0.0685.keras",
    "Rank_15": "Rank_15__DeepScan_a0.25_b0.4167_seed42_20260129-190608_loss0.0762_val0.0645.keras",
    "Rank_16": "Rank_16__DeepScan_a0.25_b0.5833_seed42_20260129-212442_loss0.0832_val0.0761.keras",
    "Rank_17": "Rank_17__DeepScan_a0.3333_b0.5_seed42_20260129-124730_loss0.0838_val0.0720.keras",
    "Rank_18": "Rank_18__DeepScan_a0.1667_b0.6667_seed42_20260129-052209_loss0.1121_val0.0815.keras",
    "Rank_19": "Rank_19__DeepScan_a0.25_b0.6667_seed42_20260129-224041_loss0.1027_val0.0819.keras",
    "Rank_20": "Rank_20__DeepScan_a0.5_b0.1667_seed43_20260129-173735_loss0.0590_val0.0602.keras",
    "Rank_21": "Rank_21__DeepScan_a0.3333_b0.6667_seed42_20260129-140531_loss0.1004_val0.0824.keras",
    "Rank_22": "Rank_22__DeepScan_a0.6667_b0.0_seed43_20260130-005859_loss0.0558_val0.0636.keras",
    "Rank_23": "Rank_23__DeepScan_a0.6667_b0.1667_seed43_20260130-020940_loss0.0630_val0.0688.keras",
    "Rank_24": "Rank_24__DeepScan_a0.5_b0.3333_seed43_20260129-190228_loss0.0743_val0.0680.keras",
    "Rank_25": "Rank_25__DeepScan_a0.3333_b0.8333_seed45_20260129-205112_loss0.1053_val0.0927.keras",
    "Rank_26": "Rank_26__DeepScan_a0.5_b0.5_seed43_20260129-203141_loss0.0805_val0.0756.keras",
    "Rank_27": "Rank_27__DeepScan_a0.25_b0.75_seed42_20260129-234822_loss0.1124_val0.0878.keras",
    "Rank_28": "Rank_28__DeepScan_a0.5_b0.6667_seed43_20260129-215115_loss0.0971_val0.0836.keras",
    "Rank_29": "Rank_29__DeepScan_a0.1667_b0.8333_seed42_20260129-064731_loss0.1090_val0.0945.keras",
    "Rank_30": "Rank_30__DeepScan_a0.25_b0.8333_seed43_20260130-011305_loss0.1068_val0.0937.keras",
    "Rank_31": "Rank_31__DeepScan_a0.5_b0.8333_seed44_20260130-093833_loss0.0939_val0.0910.keras",
    "Rank_32": "Rank_32__DeepScan_a0.6667_b0.3333_seed43_20260130-032200_loss0.0723_val0.0740.keras",
    "Rank_33": "Rank_33__DeepScan_a0.25_b0.9167_seed43_20260130-024019_loss0.1209_val0.0995.keras",
    "Rank_34": "Rank_34__DeepScan_a0.8333_b0.0_seed43_20260129-192305_loss0.0659_val0.0751.keras",
    "Rank_35": "Rank_35__DeepScan_a0.6667_b0.6667_seed43_20260130-052955_loss0.0866_val0.0844.keras",
    "Rank_36": "Rank_36__DeepScan_a0.6667_b0.5_seed43_20260130-042141_loss0.0804_val0.0794.keras",
    "Rank_37": "Rank_37__DeepScan_a0.8333_b0.6667_seed44_20260129-224337_loss0.0808_val0.0849.keras",
    "Rank_38": "Rank_38__DeepScan_a0.8333_b0.3333_seed43_20260129-171101_loss0.0743_val0.0802.keras",
    "Rank_39": "Rank_39__DeepScan_a0.6667_b0.8333_seed43_20260129-210109_loss0.0942_val0.0897.keras",
    "Rank_40": "Rank_40__DeepScan_a0.8333_b0.5_seed43_20260129-160104_loss0.0764_val0.0828.keras",
    "Rank_41": "Rank_41__DeepScan_a0.3333_b1.0_seed44_20260129-214956_loss0.1431_val0.1031.keras",
    "Rank_42": "Rank_42__DeepScan_a0.1667_b1.0_seed42_20260129-075907_loss0.1454_val0.1075.keras",
    "Rank_43": "Rank_43__DeepScan_a0.25_b1.0_seed43_20260130-040610_loss0.1373_val0.1054.keras",
    "Rank_44": "Rank_44__DeepScan_a0.8333_b0.1667_seed43_20260129-182440_loss0.0702_val0.0777.keras",
    "Rank_45": "Rank_45__DeepScan_a0.8333_b0.8333_seed43_20260129-144838_loss0.0823_val0.0880.keras",
    "Rank_46": "Rank_46__DeepScan_a0.6667_b1.0_seed44_20260130-102728_loss0.0957_val0.0946.keras",
    "Rank_47": "Rank_47__DeepScan_a0.5_b1.0_seed43_20260129-233812_loss0.1115_val0.0991.keras",
    "Rank_48": "Rank_48__DeepScan_a0.8333_b1.0_seed43_20260129-133532_loss0.0888_val0.0906.keras",
    "Rank_49": "Rank_49__DeepScan_a1.0_b0.0_seed42_20260129-121932_loss0.0759_val0.0859.keras",
    "Rank_50": "Rank_50__DeepScan_a0.0_b0.0_seed42_20260128-145959_loss0.0199_val0.0225.keras",
    "Rank_51": "Rank_51__DeepScan_a0.0_b0.1667_seed42_20260128-161918_loss0.0421_val0.0374.keras",
    "Rank_52": "Rank_52__DeepScan_a0.0_b0.3333_seed42_20260128-174208_loss0.0645_val0.0523.keras",
    "Rank_53": "Rank_53__DeepScan_a0.0_b0.5_seed42_20260128-190417_loss0.0847_val0.0672.keras",
    "Rank_54": "Rank_54__DeepScan_a0.0_b0.6667_seed42_20260128-201608_loss0.1064_val0.0821.keras",
    "Rank_55": "Rank_55__DeepScan_a0.0_b0.8333_seed42_20260128-212613_loss0.1281_val0.0970.keras",
    "Rank_56": "Rank_56__DeepScan_a0.0_b1.0_seed42_20260128-223731_loss0.1660_val0.1118.keras",
}

MODELS_NEW_RERUN = {
    "Rang_01": "Rang_01_DeepScan_a0.1667_b0.0000_seed42_20260131-011255_loss0.0257_val0.0294.keras",
    "Rang_02": "Rang_02_DeepScan_a0.3333_b0.5000_seed42_20260131-022758_loss0.0839_val0.0717.keras",
    "Rang_03": "Rang_03_DeepScan_a0.3333_b0.0000_seed42_20260131-021835_loss0.0361_val0.0407.keras",
    "Rang_04": "Rang_04_DeepScan_a0.3333_b0.3333_seed42_20260131-010926_loss0.0657_val0.0614.keras",
    "Rang_05": "Rang_05_DeepScan_a0.1667_b0.3333_seed43_20260131-002251_loss0.0582_val0.0554.keras",
    "Rang_06": "Rang_06_DeepScan_a0.3333_b0.1667_seed42_20260131-000951_loss0.0533_val0.0511.keras",
    "Rang_07": "Rang_07_DeepScan_a0.1667_b0.5000_seed42_20260131-014158_loss0.0744_val0.0684.keras",
    "Rang_08": "Rang_08_DeepScan_a0.5000_b0.1667_seed42_20260131-011728_loss0.0574_val0.0601.keras",
    "Rang_09": "Rang_09_DeepScan_a0.1667_b0.1667_seed42_20260131-030855_loss0.0479_val0.0425.keras",
    "Rang_10": "Rang_10_DeepScan_a0.5000_b0.0000_seed42_20260131-001704_loss0.0463_val0.0521.keras",
    "Rang_11": "Rang_11_DeepScan_a0.5000_b0.3333_seed42_20260131-031358_loss0.0683_val0.0678.keras",
    "Rang_12": "Rang_12_DeepScan_a0.0000_b0.1667_seed42_20260131-021040_loss0.0401_val0.0337.keras",
    "Rang_13": "Rang_13_DeepScan_a0.1667_b0.6667_seed42_20260131-023652_loss0.1009_val0.0814.keras",
    "Rang_14": "Rang_14_DeepScan_a0.3333_b0.6667_seed42_20260131-001041_loss0.0879_val0.0822.keras",
    "Rang_15": "Rang_15_DeepScan_a0.1667_b0.8333_seed42_20260131-000640_loss0.1355_val0.0945.keras",
    "Rang_16": "Rang_16_DeepScan_a0.5000_b0.5000_seed42_20260131-002205_loss0.0756_val0.0753.keras",
    "Rang_17": "Rang_17_DeepScan_a0.6667_b0.1667_seed42_20260131-053158_loss0.0631_val0.0685.keras",
    "Rang_18": "Rang_18_DeepScan_a0.3333_b0.8333_seed42_20260131-010607_loss0.1229_val0.0927.keras",
    "Rang_19": "Rang_19_DeepScan_a0.5000_b0.6667_seed42_20260131-011741_loss0.0923_val0.0830.keras",
    "Rang_20": "Rang_20_DeepScan_a0.6667_b0.3333_seed43_20260131-032304_loss0.0723_val0.0736.keras",
    "Rang_21": "Rang_21_DeepScan_a0.6667_b0.5000_seed42_20260131-042251_loss0.0825_val0.0788.keras",
    "Rang_22": "Rang_22_DeepScan_a0.6667_b0.0000_seed42_20260131-044258_loss0.0577_val0.0638.keras",
    "Rang_23": "Rang_23_DeepScan_a0.5000_b1.0000_seed43_20260131-031943_loss0.0993_val0.0984.keras",
    "Rang_24": "Rang_24_DeepScan_a0.5000_b0.8333_seed43_20260131-024341_loss0.1022_val0.0910.keras",
    "Rang_25": "Rang_25_DeepScan_a0.8333_b0.0000_seed42_20260131-061901_loss0.0662_val0.0746.keras",
    "Rang_26": "Rang_26_DeepScan_a0.6667_b0.6667_seed44_20260131-055426_loss0.0887_val0.0842.keras",
    "Rang_27": "Rang_27_DeepScan_a0.3333_b1.0000_seed42_20260131-020559_loss0.1302_val0.1031.keras",
    "Rang_28": "Rang_28_DeepScan_a0.8333_b0.1667_seed43_20260131-041442_loss0.0707_val0.0773.keras",
    "Rang_29": "Rang_29_DeepScan_a0.8333_b0.6667_seed43_20260131-043624_loss0.0821_val0.0851.keras",
    "Rang_30": "Rang_30_DeepScan_a0.8333_b0.3333_seed43_20260131-052558_loss0.0741_val0.0801.keras",
    "Rang_31": "Rang_31_DeepScan_a0.6667_b0.8333_seed42_20260131-033152_loss0.0990_val0.0892.keras",
    "Rang_32": "Rang_32_DeepScan_a0.1667_b1.0000_seed42_20260131-012331_loss0.1229_val0.1074.keras",
    "Rang_33": "Rang_33_DeepScan_a0.0000_b0.0000_seed43_20260131-003040_loss0.0152_val0.0184.keras",
    "Rang_34": "Rang_34_DeepScan_a0.8333_b0.5000_seed42_20260131-061643_loss0.0791_val0.0824.keras",
    "Rang_35": "Rang_35_DeepScan_a0.8333_b0.8333_seed42_20260131-052931_loss0.0845_val0.0878.keras",
    "Rang_36": "Rang_36_DeepScan_a0.6667_b1.0000_seed42_20260131-052908_loss0.1067_val0.0946.keras",
    "Rang_37": "Rang_37_DeepScan_a0.8333_b1.0000_seed43_20260131-064157_loss0.0893_val0.0904.keras",
    "Rang_38": "Rang_38_DeepScan_a1.0000_b0.0000_seed42_20260131-073210_loss0.0766_val0.0862.keras",
    "Rang_39": "Rang_39_DeepScan_a0.0000_b0.5000_seed42_20260131-000641_loss0.0809_val0.0671.keras",
    "Rang_40": "Rang_40_DeepScan_a0.0000_b0.3333_seed42_20260131-040643_loss0.0541_val0.0522.keras",
    "Rang_41": "Rang_41_DeepScan_a0.0000_b0.6667_seed42_20260131-015925_loss0.1080_val0.0820.keras",
    "Rang_42": "Rang_42_DeepScan_a0.0000_b0.8333_seed42_20260131-034436_loss0.1071_val0.0969.keras",
    "Rang_43": "Rang_43_DeepScan_a0.0000_b1.0000_seed42_20260131-000640_loss0.1372_val0.1118.keras",
}

# MODELL-LISTEN
MODELS_INF_SEED = {
    # POINT 0 (P0)
    "P0_Seed43": "InfSeed_P0_a0.0000_b0.0000_seed43_20260210-170149_loss0.0195_val0.0224.keras",
    "P0_Seed44": "InfSeed_P0_a0.0000_b0.0000_seed44_20260210-180919_loss0.0195_val0.0224.keras",
    "P0_Seed47": "InfSeed_P0_a0.0000_b0.0000_seed47_20260210-193132_loss0.0158_val0.0181.keras",
    "P0_Seed50": "InfSeed_P0_a0.0000_b0.0000_seed50_20260210-204618_loss0.0191_val0.0219.keras",
    "P0_Seed62": "InfSeed_P0_a0.0000_b0.0000_seed62_20260211-001540_loss0.0152_val0.0180.keras",
    "P0_Seed63": "InfSeed_P0_a0.0000_b0.0000_seed63_20260211-013023_loss0.0155_val0.0180.keras",
    "P0_Seed65": "InfSeed_P0_a0.0000_b0.0000_seed65_20260211-024800_loss0.0154_val0.0182.keras",
    "P0_Seed69": "InfSeed_P0_a0.0000_b0.0000_seed69_20260212-092553_loss0.0161_val0.0182.keras",
    "P0_Seed75": "InfSeed_P0_a0.0000_b0.0000_seed75_20260212-110809_loss0.0203_val0.0226.keras",

    # POINT 1 (P1)
    "P1_Seed43": "InfSeed_P1_a0.8333_b0.0000_seed43_20260210-170150_loss0.0662_val0.0747.keras",
    "P1_Seed44": "InfSeed_P1_a0.8333_b0.0000_seed44_20260210-180249_loss0.0655_val0.0741.keras",
    "P1_Seed45": "InfSeed_P1_a0.8333_b0.0000_seed45_20260210-190307_loss0.0655_val0.0746.keras",
    "P1_Seed46": "InfSeed_P1_a0.8333_b0.0000_seed46_20260210-200941_loss0.0659_val0.0745.keras",
    "P1_Seed47": "InfSeed_P1_a0.8333_b0.0000_seed47_20260210-211638_loss0.0662_val0.0744.keras",
    "P1_Seed48": "InfSeed_P1_a0.8333_b0.0000_seed48_20260210-222216_loss0.0658_val0.0751.keras",
    "P1_Seed49": "InfSeed_P1_a0.8333_b0.0000_seed49_20260210-233303_loss0.0661_val0.0744.keras",
    "P1_Seed50": "InfSeed_P1_a0.8333_b0.0000_seed50_20260211-003034_loss0.0663_val0.0752.keras",
    "P1_Seed53": "InfSeed_P1_a0.8333_b0.0000_seed53_20260211-020101_loss0.0658_val0.0742.keras",
}

MODE = "INF_SEED"

# --- AUTOMATISCHE PFAD-STEUERUNG ---
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")

if MODE == "INF_SEED":
    print(">>> Modus: INF_SEED (18 Modelle - Seed Study)")
    MODELS = MODELS_INF_SEED
    # Korrektur: Den Ordner "Prediction.npz" im Pfad einfügen!
    IN_DIR  = ROOT_DIR / "Unet/Analysis_ROI/Prediction.npz/Predictions_Raw_21_33" 
    OUT_DIR = ROOT_DIR / "Unet/Analysis_ROI/H_K_L_Plots/Analysis_K_Direction_INF_SEED"
elif MODE == "NEW_RERUN":
    print(">>> Modus: NEW_RERUN (43 Modelle)")
    MODELS = MODELS_NEW_RERUN
    IN_DIR  = ROOT_DIR / "Unet/Analysis_ROI/Predictions_Raw_new_RERUN"
    OUT_DIR = ROOT_DIR / "Unet/Analysis_ROI/Analysis_K_Direction_NEW_RERUN"
elif MODE == "RERUN":
    print(">>> Modus: RERUN (56 Modelle)")
    MODELS = MODELS_RERUN
    IN_DIR  = ROOT_DIR / "Unet/Analysis_ROI/Predictions_Raw_RERUN"
    OUT_DIR = ROOT_DIR / "Unet/Analysis_ROI/Analysis_K_Direction_RERUN"
else:
    print(">>> Modus: ORIGINAL (10 Modelle)")
    MODELS = MODELS_ORIGINAL
    IN_DIR  = ROOT_DIR / "Unet/Analysis_ROI/Predictions_Raw"
    OUT_DIR = ROOT_DIR / "Unet/Analysis_ROI/Analysis_K_Direction"

OUT_DIR.mkdir(parents=True, exist_ok=True)





# HINZUGEFÜGT: ylim_raw (Zeile 2) und ylim_sbr (Zeile 3) für jede Serie
ROI_Y = (0, 192)
FIT_WINDOW = (90, 130)

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "bg_gap": -211, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "bg_gap": 52,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "bg_gap": 64,   "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
    15: {"slice_idx": 19, "roi_x": (141, 162), "bg_gap": -162, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    16: {"slice_idx": 17, "roi_x": (121, 142), "bg_gap": 78,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "bg_gap": -213, "vis_p": (0.5, 99.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    22: {"slice_idx": 17, "roi_x": (173, 194), "bg_gap": -183, "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "bg_gap": -80,  "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    35: {"slice_idx": 24, "roi_x": (106, 127), "bg_gap": 45,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
    50: {"slice_idx": 13, "roi_x": (97, 118),  "bg_gap": 38,   "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]
IMAGE_WIDTH = 240

# =====================================================
# 2. Hilfsfunktionen
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def get_background_std(image, roi_y, bg_coords):
    box_right = image[roi_y[0]:roi_y[1], bg_coords[0]:bg_coords[1]]
    return np.std(box_right)

def calculate_sbr_k_profiles(image, cfg, bg_coords, force_noise_std=None):
    roi_x = cfg["roi_x"]
    signal_slice = image[ROI_Y[0]:ROI_Y[1], roi_x[0]:roi_x[1]]
    n_signal_cols = signal_slice.shape[1]
    profile_signal = np.sum(signal_slice, axis=1)
    
    background_slice = image[ROI_Y[0]:ROI_Y[1], bg_coords[0]:bg_coords[1]]
    n_background_cols = background_slice.shape[1]
    
    profile_background_raw = np.sum(background_slice, axis=1)
    scale = n_signal_cols / n_background_cols if n_background_cols > 0 else 0
    profile_background = profile_background_raw * scale

    profile_net = profile_signal - profile_background
    denom = np.where(profile_background == 0, 1e-9, profile_background)
    profile_sbr = profile_net / denom

    pixel_noise_std = force_noise_std if force_noise_std is not None else np.std(background_slice)
    
    err_signal_sum = pixel_noise_std * np.sqrt(n_signal_cols)
    err_bg_sum     = pixel_noise_std * np.sqrt(n_background_cols) * scale
    scalar_err_net = np.sqrt(err_signal_sum**2 + err_bg_sum**2)
    
    rel_err_net = scalar_err_net / np.abs(np.where(profile_net == 0, 1.0, profile_net))
    rel_err_bg  = err_bg_sum / np.abs(denom)
    profile_sbr_error = np.abs(profile_sbr) * np.sqrt(rel_err_net**2 + rel_err_bg**2)
    
    return np.arange(ROI_Y[0], ROI_Y[1]), profile_signal, profile_background, profile_sbr, profile_sbr_error

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 3: 
        return None, None, None
    p0 = [np.max(y_f) - np.median(y_f), x_f[np.argmax(y_f)], 5.0]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], 15.0])
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=y_err[mask], 
                               absolute_sigma=True, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except:
        return None, None, None

# =====================================================
# 3. Prozess-Funktion
# =====================================================
def process_combination(model_id, s_id, cfg):
    # Probiere erst den vollen Namen, dann den bereinigten
    path = IN_DIR / f"Pred_{model_id}_D5_S{s_id}_FullSeries.npz"
    
    if not path.exists():
        # Falls nicht gefunden, versuche die Logik aus dem L-Skript
        if model_id.startswith("Rang_"):
            pure_id = model_id[8:]
        elif model_id.startswith("Rank_"): 
            pure_id = model_id.split("__")[1] if "__" in model_id else model_id[8:]
        else:
            pure_id = model_id
        path = IN_DIR / f"Pred_{pure_id}_D5_S{s_id}_FullSeries.npz"

    if not path.exists():
        # Debug-Meldung, damit du siehst, welche Datei fehlt:
        # print(f" Datei nicht gefunden: Pred_{model_id}...") 
        return

    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    bg_l = min(IMAGE_WIDTH, cfg["roi_x"][1] + cfg["bg_gap"])
    bg_r = min(IMAGE_WIDTH, bg_l + 20)
    bg_coords = (bg_l, bg_r)

    gt_noise = get_background_std(imgs[2], ROI_Y, bg_coords) 
    results = []
    
    y_ax = np.arange(ROI_Y[0], ROI_Y[1])
    for i, img in enumerate(imgs):
        # 1. Signale und Hintergründe extrahieren (K-Richtung)
        sig_s = img[ROI_Y[0]:ROI_Y[1], cfg["roi_x"][0]:cfg["roi_x"][1]]
        bg_s  = img[ROI_Y[0]:ROI_Y[1], bg_l:bg_r]
        
        # 2. Profile berechnen (axis=1 für vertikale Profile)
        prof_sig = np.sum(sig_s, axis=1)
        scale = sig_s.shape[1] / bg_s.shape[1]
        prof_bg = np.sum(bg_s, axis=1) * scale
        
        # 3. SBR berechnen
        denom = np.where(prof_bg == 0, 1e-9, prof_bg)
        sbr = (prof_sig - prof_bg) / denom
        
        # 4. Fehlerrechnung
        p_std = gt_noise if i == 1 else np.std(bg_s)
        err_net = np.sqrt((p_std * np.sqrt(sig_s.shape[1]))**2 + (p_std * np.sqrt(bg_s.shape[1]) * scale)**2)
        sbr_err = np.abs(sbr) * np.sqrt((err_net/np.abs(np.where(prof_sig-prof_bg==0,1,prof_sig-prof_bg)))**2 + (p_std*np.sqrt(bg_s.shape[1])*scale/np.abs(denom))**2)

        # 5. FIT-LOGIK: Low Count (0) überspringen
        fit_y, par, perr = (None, None, None)
        if i > 0:
            fit_y, par, perr = perform_gaussian_fit(y_ax, sbr, sbr_err, FIT_WINDOW)
        
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    # Visualisierung (3x3 Layout)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        # --- ZEILE 0: BILDER ---
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_low, p_high), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        roi_x = cfg["roi_x"]
        roi_w = roi_x[1] - roi_x[0]
        roi_h = ROI_Y[1] - ROI_Y[0]
        ax.add_patch(patches.Rectangle((roi_x[0], ROI_Y[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((roi_x[0], FIT_WINDOW[0]), roi_w, FIT_WINDOW[1]-FIT_WINDOW[0], lw=0, fc='green', alpha=0.2))
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y[0]), bg_r-bg_l, roi_h, lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        # --- ZEILE 1: RAW INTENSITÄTEN (Individuelles ylim) ---
        ax2 = axes[1, i]
        ax2.plot(y_ax, results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(y_ax, results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.15)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(cfg.get("ylim_raw", (2.5, 7.0))) 
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)

        # --- ZEILE 2: SRBR & GAUSS FIT (Individuelles ylim) ---
        ax3 = axes[2, i]
        ax3.errorbar(y_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if results[i]['fit'] is not None:
                p, e = results[i]['par'], results[i]['perr']
                # Amplitude (p[0]) hinzugefügt
                l = (f"Gauss (Amp={p[0]:.2f}±{e[0]:.2f}, "
                    f"Peak={p[1]:.1f}±{e[1]:.1f}, "
                    f"σ={p[2]:.2f}±{e[2]:.2f})")
                ax3.plot(y_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        
        ax3.set_xlabel("Pixel Y")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlim(FIT_WINDOW)
        ax3.set_ylim(cfg.get("ylim_sbr", (-0.2, 0.5)))

    plt.tight_layout()
    save_dir = OUT_DIR / f"Serie_{s_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_{model_id}_S{s_id}_Slice{idx}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. Main
# =====================================================
def main():
    matplotlib.use("Agg")
    for rank_key, full_name in MODELS.items():
        # WICHTIG: Nutze für INF_SEED direkt den rank_key (z.B. "P0_Seed43")
        # für die Suche der .npz Datei!
        if MODE == "INF_SEED":
            model_id = rank_key
        else:
            model_id = full_name.replace(".keras", "").split("_2026")[0] if "_2026" in full_name else rank_key
            
        print(f"Processing {model_id}...")
        for s_id, cfg in sorted(SERIES_CONFIG.items()):
            process_combination(model_id, s_id, cfg)

if __name__ == "__main__":
    main()