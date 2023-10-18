import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

"""
Default dataset file for training on COD10K dataset and test on COD10K, NC4K dataset

Additional train/test data:
CODR1: LDR with 1 class (foreground)
CODR4: LDR with 4 classes (background, easy, medium, hard)
CODR4_extended: combine both LDR train and test dataset (maximum the rank label)
CODR4_semi: Semi-supervised Labeling COD10K with 4 classes (background, easy, medium, hard)
COD10K_Ziss: COD10K with Ziss score
CAMO: CAMO dataset
"""

COD10K_DATASET_ROOT = "/home/dtpthao/data_unzip/camo/COD10K-v3"
CAMO_DATASET_ROOT = "/home/dtpthao/data_unzip/camo/camo++"

# COD10K train 
COD10K_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
COD10K_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "Train/CAM_Instance_Train.json")
SUBSET_COD10K_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/Subset/CAM_Instance_Train_subset.json")

# COD10K test
COD10K_TEST_PATH = os.path.join(COD10K_DATASET_ROOT, "Test/Image")
COD10K_TEST_JSON = os.path.join(COD10K_DATASET_ROOT, "Test/CAM_Instance_Test.json")
SUBSET_COD10K_TEST_JSON = os.path.join(COD10K_DATASET_ROOT, "json/Subset/CAM_Instance_Test_subset.json")

# NC4K test
NC4K_ROOT = "/home/dtpthao/data_unzip/camo/NC4K"
NC4K_PATH = os.path.join(NC4K_ROOT, "test/image")
NC4K_JSON = os.path.join(NC4K_ROOT, "nc4k_test.json")
SUBSET_NC4K_JSON = os.path.join(COD10K_DATASET_ROOT, "json/Subset/NC4K_subset.json")

# Addition train and test dataset

# CODR1
CODR1_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
CODR1_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/LDR/fix_rank_dataset2000_1cls.json")
CODR1_TEST_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
CODR1_TEST_JSON = os.path.join(COD10K_DATASET_ROOT, "json/LDR/fix_rank_test_dataset_1cls.json")

# CODR4
CODR4_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
CODR4_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/COD10K/CAM_Instance_Train_WithRankLabel_Train.json")

# CODR4_extended
CODR4_EXTENDED_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
CODR4_EXTENDED_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/COD10K/CAM_Instance_Train_WithRankLabel_Full.json")

# CODR4_semi
CODR4_SEMI_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
CODR4_SEMI_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/COD10K/CAM_Instance_Train_WithSemiRankLabel.json")

# COD10K_ziss
COD10K_ZISS_TRAIN_PATH = os.path.join(COD10K_DATASET_ROOT, "Train/Image")
COD10K_ZISS_TRAIN_JSON = os.path.join(COD10K_DATASET_ROOT, "json/COD10K_Ziss/CAM_Instance_Train_Ziss.json")

# CAMO
CAMO_TRAIN_PATH = os.path.join(CAMO_DATASET_ROOT, "Train/Image")
CAMO_TEST_PATH = os.path.join(CAMO_DATASET_ROOT, "Test/Image")
CAMO_TRAIN_JSON = os.path.join(CAMO_DATASET_ROOT, "Train/camo_train_1.0.json")
CAMO_TEST_JSON = os.path.join(CAMO_DATASET_ROOT, "Test/camo_test_1.0.json")

# CLASS_NAMES = ["foreground"] # Commented out since not used


PREDEFINED_SPLITS_DATASET = {
    "COD10K_TRAIN": (COD10K_TRAIN_PATH, COD10K_TRAIN_JSON),
    "COD10K_TEST": (COD10K_TEST_PATH, COD10K_TEST_JSON),
    
    "SUBSET_COD10K_TRAIN": (COD10K_TRAIN_PATH, SUBSET_COD10K_TRAIN_JSON),
    "SUBSET_COD10K_TEST": (COD10K_TEST_PATH, SUBSET_COD10K_TEST_JSON),
    
    "CODR1_TRAIN": (CODR1_TRAIN_PATH, CODR1_TRAIN_JSON),
    "CODR1_TEST": (CODR1_TEST_PATH, CODR1_TEST_JSON),
    
    "CODR4_TRAIN": (CODR4_TRAIN_PATH, CODR4_TRAIN_JSON),
    "CODR4_EXTENDED_TRAIN": (CODR4_EXTENDED_TRAIN_PATH, CODR4_EXTENDED_TRAIN_JSON),
    "CODR4_SEMI_TRAIN": (CODR4_SEMI_TRAIN_PATH, CODR4_SEMI_TRAIN_JSON),
    
    "COD10K_ZISS": (COD10K_ZISS_TRAIN_PATH, COD10K_ZISS_TRAIN_JSON),
    
    "CAMO_TRAIN": (CAMO_TRAIN_PATH, CAMO_TRAIN_JSON),
    "CAMO_TEST": (CAMO_TEST_PATH, CAMO_TEST_JSON),
    
    "NC4K_TEST": (NC4K_PATH, NC4K_JSON),
    "SUBSET_NC4K_TEST": (NC4K_PATH, SUBSET_NC4K_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, json_file=json_file, image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name, extra_annotation_keys=["s_alpha"]))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco"
    )
