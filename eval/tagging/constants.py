CLASSIFICATION_SYSTEM = """
***
### **Classification System: Medical Tasks**

*   **organ_understanding:** The sample's primary goal is to identify, locate, or assess the characteristics (e.g., size, shape) of specific anatomical organs or structures.
*   **lesion_understanding:** The sample focuses on detecting, localizing, or characterizing pathological findings or abnormalities, such as tumors, nodules, fractures, or inflammation.
*   **modality_recognition:** The sample's main task is to identify the type of medical imaging used (e.g., "Is this a CT or an MRI?").
*   **change_comparison:** The sample requires comparing two or more images taken at different times to assess the progression, regression, or stability of a condition.
*   **examination_planning:** The sample asks "What should be done next?" It involves recommending the most appropriate subsequent diagnostic test or imaging exam.
*   **etiological_diagnosis:** The sample requires identifying the most probable cause or specific disease that explains the presented clinical and/or imaging findings.
*   **disease_staging:** The sample focuses on determining the extent or severity of a confirmed disease, such as staging a cancer based on tumor size and metastasis.
*   **drug_usage:** The sample requires recommending a specific medication, class of drugs, or its dosage for a given condition.
*   **protocol_design:** The sample involves recommending a non-pharmacological treatment, such as a surgical procedure, radiation therapy plan, or other therapeutic protocol.
*   **treatment_response:** The sample asks to predict how a patient's condition is likely to respond to a particular treatment.
*   **outcome_prediction:** The sample requires forecasting the long-term outcome, such as survival rate, recurrence probability, or likelihood of functional recovery.
*   **chronic_disease_management:** The sample focuses on recommendations for the ongoing care of a chronic condition, such as monitoring schedules or lifestyle adjustments.
*   **dietary_recommendation:** The sample requires providing specific nutritional advice or dietary plans tailored to a patient's health condition.
*   **basic_science_knowledge:** The sample tests knowledge of preclinical subjects like biochemistry, physiology, anatomy (in the abstract), or pharmacology mechanisms.
*   **clinical_knowledge:** The sample tests factual recall of clinical definitions, diagnostic criteria, or standard-of-care guidelines without requiring diagnostic reasoning for a specific case.
"""

VALID_ORGANS = [
    # ----------------------------------------------------------------------
    # Major Anatomical Regions (NEW)
    # ----------------------------------------------------------------------
    "abdomen",           # 腹部 (Abdominal cavity)
    "head",              # 头部
    "mediastinum",       # 纵隔
    "neck",              # 颈部
    "pelvis",            # 盆腔
    "retroperitoneum",   # 腹膜后
    "thorax",            # 胸廓 (Chest)

    # ----------------------------------------------------------------------
    # Head, Neck & Nervous System
    # ----------------------------------------------------------------------
    "brain",             # 脑
    "ear",               # 耳
    "eye",               # 眼
    "larynx",            # 喉
    "nerve",             # 神经
    "parathyroid_gland", # 甲状旁腺
    "pharynx",           # 咽
    "pituitary_gland",   # 垂体
    "salivary_gland",    # 唾液腺
    "sinus",             # 鼻窦
    "spinal_cord",       # 脊髓
    "thyroid_gland",     # 甲状腺
    "tongue",            # 舌
    "trachea",           # 气管

    # ----------------------------------------------------------------------
    # Thorax & Vascular
    # ----------------------------------------------------------------------
    "aorta",             # 主动脉
    "bronchus",          # 支气管
    "diaphragm",         # 膈肌
    "esophagus",         # 食管
    "heart",             # 心脏
    "lung",              # 肺
    "lymph_node",        # 淋巴结
    "myocardium",        # 心肌
    "pericardium",       # 心包
    "pleura",            # 胸膜
    "portal_vein",       # 门静脉
    "pulmonary_artery",  # 肺动脉
    "thymus",            # 胸腺

    # ----------------------------------------------------------------------
    # Abdomen & GI Tract
    # ----------------------------------------------------------------------
    "adrenal_gland",     # 肾上腺
    "anal_canal",        # 肛管
    "appendix",          # 阑尾
    "bile_duct",         # 胆管
    "colon",             # 结肠
    "duodenum",          # 十二指肠
    "gallbladder",       # 胆囊
    "jejunum_ileum",     # 空回肠
    "kidney",            # 肾
    "liver",             # 肝
    "mesentery",         # 肠系膜
    "omentum",           # 网膜
    "pancreas",          # 胰腺
    "peritoneum",        # 腹膜
    "rectum",            # 直肠
    "spleen",            # 脾
    "stomach",           # 胃

    # ----------------------------------------------------------------------
    # Pelvis & Reproductive System
    # ----------------------------------------------------------------------
    "adnexa",            # 附件
    "bladder",           # 膀胱
    "cervix",            # 子宫颈
    "ejaculatory_duct",  # 射精管
    "epididymis",        # 附睾
    "fetus",             # 胎儿 (NEW)
    "ovary",             # 卵巢
    "penis",             # 阴茎
    "placenta",          # 胎盘
    "prostate",          # 前列腺
    "sacral_canal",      # 骶管
    "scrotum",           # 阴囊
    "seminal_vesicle",   # 精囊腺
    "testis",            # 睾丸
    "ureter",            # 输尿管
    "urethra",           # 尿道
    "urachus",           # 脐尿管
    "uterus",            # 子宫
    "vagina",            # 阴道
    "vas_deferens",      # 输精管
    "vulva",             # 外阴

    # ----------------------------------------------------------------------
    # General & Musculoskeletal
    # ----------------------------------------------------------------------
    "blood_vessel",      # 血管
    "bone",              # 骨
    "breast",            # 乳腺
    "cartilage",         # 软骨
    "joint",             # 关节
    "ligament",          # 韧带
    "muscle",            # 肌肉
    "skin",              # 皮肤
    "soft_tissue",       # 软组织
    "tendon",            # 肌腱
    "tooth",             # 牙齿
    
    # ----------------------------------------------------------------------
    # Not Applicable
    # ----------------------------------------------------------------------
    "N/A"
]




VALID_TASKS = [
    # Image Understanding
    "organ_understanding", 
    "lesion_understanding", 
    "modality_recognition", 
    "change_comparison",
    # Differential Diagnosis
    "examination_planning", 
    "etiological_diagnosis", 
    "disease_staging",
    # Treatment Recommendation
    "drug_usage", 
    "protocol_design",
    # Prognosis Prediction
    "treatment_response", 
    "outcome_prediction",
    # Health Management
    "chronic_disease_management", 
    "dietary_recommendation",
    # Foundational Knowledge
    "basic_science_knowledge", 
    "clinical_knowledge"
]

# --- List of Valid Quality and Hardness Levels ---
VALID_QUALITIES = ["good", "mild", "bad"]
VALID_HARDNESS_LEVELS = ["easy", "mild", "hard"]

# --- Create sets for faster lookups in the validation script ---
# This is crucial for performance.
VALID_ORGANS_SET = set(VALID_ORGANS)
VALID_TASKS_SET = set(VALID_TASKS) 
VALID_QUALITIES_SET = set(VALID_QUALITIES)
VALID_HARDNESS_LEVELS_SET = set(VALID_HARDNESS_LEVELS)


ORGAN_SYNONYM_MAP = {
    # --- GI Tract / Digestive System ---
    "gastrointestinal": ["stomach", "jejunum_ileum", "colon"],
    "gastrointestinal_system": ["stomach", "jejunum_ileum", "colon"],
    "gastrointestinal_tract": ["stomach", "jejunum_ileum", "colon"],
    "GI_tract": ["stomach", "jejunum_ileum", "colon"],
    "GIT": ["stomach", "jejunum_ileum", "colon"],
    "digestive_system": ["stomach", "jejunum_ileum", "colon"],
    "gut": ["jejunum_ileum", "colon"],
    "intestine": ["jejunum_ileum", "colon"],
    "intestines": ["jejunum_ileum", "colon"],
    "bowel": ["jejunum_ileum", "colon"],
    "small_intestine": ["duodenum", "jejunum_ileum"],
    "small_bowel": ["duodenum", "jejunum_ileum"],
    "large_intestine": ["colon"],
    "large_bowel": ["colon"],
    "colon_rectum": ["colon", "rectum"],
    "ileum": "jejunum_ileum", "jejunum": "jejunum_ileum", "terminal_ileum": "jejunum_ileum",
    "cecum": "colon", "caecum": "colon", "sigmoid_colon": "colon", "sigmoid": "colon", "transverse_colon": "colon",
    "gastric_stomach": "stomach", "gastric": "stomach", "gastric_body": "stomach",
    "pylorus": "stomach", "gastric_antrum": "stomach", "gastric_fundus": "stomach",
    "rectal_wall": "rectum",
    "anus": "anal_canal",
    # --- NEW ADDITIONS (GI) ---
    "gastric_mucosa": "stomach",
    "gastric_bubble": "stomach",
    "gastric_gland": "stomach",
    "gastric_pylorus": "stomach",
    "pyloric_antrum": "stomach",
    "gastric_cardia": "stomach",
    "foregut": "stomach", # Primarily refers to embryological origin of stomach, liver, pancreas, etc. but often used loosely for upper GI. 'stomach' is a safe bet.
    "ileocecal_valve": ["colon", "jejunum_ileum"],
    "rectal": "rectum",
    "ampulla_of_vater": ["bile_duct", "pancreas"], # Junction point
    "gastrocolic_ligament": "ligament", # It connects stomach and colon

    # --- Hepatobiliary, Pancreatic, and Spleen ---
    "common_bile_duct": "bile_duct",
    "脾": "spleen", "脾脏": "spleen",
    # --- NEW ADDITIONS (Hepatobiliary) ---
    "胆囊": "gallbladder",
    "胆道系统": "bile_duct",
    "cystic_duct": "bile_duct",
    "common_hepatic_duct": "bile_duct",
    "pancreatic_duct": "pancreas", # A key part of the pancreas

    # --- Head, Neck, and Mouth ---
    "mouth": ["tongue", "tooth"],
    "oral_cavity": ["tongue", "tooth"],
    "oral_tongue": "tongue",
    "teeth": "tooth", "gingiva": "tooth", "gum": "tooth", "gums": "tooth",
    "throat": ["pharynx", "larynx"],
    "nasopharynx": "pharynx", "oropharynx": "pharynx", "hypopharynx": "pharynx",
    "tonsil": "pharynx", "adenoid": "pharynx", "adenoids": "pharynx",
    "lip": "soft_tissue", "lips": "soft_tissue",
    "nose": ["cartilage", "skin"], "鼻": ["cartilage", "skin"],
    "nasal_septum": ["cartilage", "bone"],
    "soft_palate": "soft_tissue", "hard_palate": "bone",
    "scalp": "skin",
    "唾液腺": "salivary_gland",
    # --- NEW ADDITIONS (Head/Neck) ---
    "face": ["skin", "muscle", "bone", "soft_tissue"],
    "palate": ["bone", "soft_tissue"],
    "parathyroids": "parathyroid_gland",
    "submandibular_duct": "salivary_gland", # Duct of a salivary gland
    "maxillary_third_molar": "tooth",
    "maxillary_teeth": "tooth",
    "upper_incisors": "tooth",
    "root_canal": "tooth",
    "carotid_body": "nerve", # Chemoreceptor body, but anatomically associated with carotid. Mapping to nerve tissue is plausible.

    # --- Sinuses ---
    "nasal_cavity": "sinus", "鼻腔": "sinus", "鼻窦": "sinus",
    "maxillary_sinuses": "sinus", "maxillarysinus": "sinus",
    "frontal_sinuses": "sinus", "sphenoid_sinuses": "sinus", "ethmoid_sinuses": "sinus",
    "paranasal_sinuses": "sinus", "sphenoid_sinus": "sinus", "sphenoidsinus": "sinus",
    # --- NEW ADDITIONS (Sinus) ---
    "ethmoids": "sinus",
    "ethmoidal_air_cells": "sinus",
    "sphenoidal_sinuses": "sinus",
    "sigmoid_sinuses": "blood_vessel", # This is a dural venous sinus, might be 'blood_vessel'. But given the context of other sinuses, 'sinus' might be intended. Review needed. Let's stick with sinus for now.
    "turbinates": "sinus", # Bony structures within the nasal cavity
    "nasal_turbinates": "sinus",
    "nasal_concha": "sinus",

    # --- Brain and Nervous System ---
    "temporal_lobe": "brain", "hypothalamus": "brain", "brainstem": "brain",
    "optic_chiasm": "brain", "ventricle": "brain",
    "peripheral_nerve": "nerve", "sciatic_nerve": "nerve", "optic_nerve": "nerve",
    "facial_nerve": "nerve", "brachial_plexus": "nerve", "median_nerve": "nerve",
    "nerve_root": "nerve", "peroneal_nerve": "nerve", "pudendal_nerve": "nerve",
    "vagus_nerve": "nerve", "trigeminal_nerve": "nerve",
    # --- NEW ADDITIONS (Brain/Nerve) ---
    "midbrain": "brain",
    "occipital_lobe": "brain",
    "brain_stem": "brain",
    "cerebellar_hemisphere": "brain",
    "optic_tract": "brain",
    "ulnar_nerve": "nerve",
    "femoral_nerve": "nerve",
    "genitofemoral_nerve": "nerve",
    "auditory_nerve": "nerve",
    "vestibulocochlear_nerve": "nerve",
    "accessory_nerve": "nerve",
    "trigeminal_ganglion": "nerve",

    # --- Skeletal System / Bones ---
    "jaw": "bone", "jawbone": "bone", "skull": "bone", "mandible": "bone", "maxilla": "bone",
    "rib": "bone", "sternum": "bone", "clavicle": "bone", "clavicles": "bone", "rib_cage": "thorax",
    "vertebra": "bone", "vertebral_body": "bone", "spine": "bone", "vertebral_column": "bone",
    "cervical_spine": "bone", "thoracic_spine": "bone", "disc": "bone",
    "humerus": "bone", "scapula": "bone", "femur": "bone", "tibia": "bone", "patella": "bone",
    "temporal_bone": "bone", "skull_base": "bone", "sella": "bone", "ilium": "bone", "orbit": "bone",
    "hyoid_bone": "bone", "zygomatic_arch": "bone",
    "leg": "bone", "arm": "bone", "forearm": "bone", "thigh": "bone",
    "foot": "bone", "hand": "bone",
    "bone_marrow": "bone",
    # --- NEW ADDITIONS (Bone) ---
    "thoracic_vertebrae": "bone",
    "femoral_head": "bone",
    "alveolar_bone": "bone",
    "clivus": "bone",
    "pubic_ramus": "bone",
    "styloid_process": "bone",
    "mandibular_cortex": "bone",
    "lumbar_vertebral_body": "bone",
    "phalanges": "bone",
    "ribcage": "bone",

    # --- Joints ---
    "hip": "joint", "shoulder": "joint", "ankle": "joint", "wrist": "joint",
    "elbow": "joint", "knee": "joint", "knee_joint": "joint", "hip_joint": "joint",
    "temporomandibular_joint": "joint", "sacroiliac_joint": "joint", "shoulder_joint": "joint",
    # --- NEW ADDITIONS (Joint) ---
    "sacroiliac_joints": "joint",
    "atlantoaxial_joint": "joint",
    "ao_joint": "joint", # Atlanto-occipital joint

    # --- Ligaments ---
    # (Original map has ligament, but no synonyms yet)
    "plantar_fascia": "ligament", "periodontal_ligament": "ligament",
    # --- NEW ADDITIONS (Ligament) ---
    "periodontal_fibres": "ligament",
    "PDL": "ligament", # Abbreviation for Periodontal Ligament
    "uterosacral_ligament": "ligament",
    "cardinal_ligament": "ligament",
    "transverse_carpal_ligament": "ligament",

    # --- Cardiovascular System (Heart & Blood Vessels) ---
    "mitral_valve": "heart", "atrium": "heart", "right_atrium": "heart", "left_atrium": "heart", "valve": "heart",
    "artery": "blood_vessel", "vein": "blood_vessel", "vascular_system": "blood_vessel",
    "inferior_vena_cava": "blood_vessel", "superior_vena_cava": "blood_vessel", "vena_cava": "blood_vessel",
    "IVC": "blood_vessel", "ivc": "blood_vessel",
    "vertebral_artery": "blood_vessel", "vertebral_arteries": "blood_vessel",
    "coronary_artery": "blood_vessel", "renal_artery": "blood_vessel", "hepatic_artery": "blood_vessel",
    "femoral_artery": "blood_vessel", "splenic_artery": "blood_vessel", "subclavian_artery": "blood_vessel",
    "celiac_artery": "blood_vessel", "superior_mesenteric_artery": "blood_vessel",
    "carotid_artery": "blood_vessel", "carotid_arteries": "blood_vessel",
    "common_carotid_artery": "blood_vessel", "internal_carotid_artery": "blood_vessel", "ICA": "blood_vessel",
    "renal_vein": "blood_vessel", "hepatic_vein": "blood_vessel", "splenic_vein": "blood_vessel",
    "subclavian_vein": "blood_vessel", "brachiocephalic_vein": "blood_vessel", "jugular_vein": "blood_vessel",
    "umbilical_cord": "blood_vessel", "umbilical_vein": "blood_vessel",
    "portal_venous_system": "portal_vein",
    "portal_and_splenic_veins": ["portal_vein", "spleen", "blood_vessel"],
    "脾动脉": "blood_vessel", "脾静脉": "blood_vessel",
    # --- NEW ADDITIONS (Cardiovascular) ---
    "right_ventricle": "heart",
    "fetal_heart": "heart",
    "atrial_septum": "heart",
    "tricuspid_valve": "heart",
    "gastroduodenal_artery": "blood_vessel",
    "brachiocephalic_artery": "blood_vessel",
    "brachiocephalic_trunk": "blood_vessel",
    "brachycephalic_vein": "blood_vessel", # Common misspelling of brachiocephalic
    "internal_jugular_vein": "blood_vessel",
    "mesenteric_artery": "blood_vessel",
    "superior_mesenteric_vein": "blood_vessel",
    "SMV": "blood_vessel",
    "celiac_axis": "blood_vessel",
    "celiac_trunk": "blood_vessel",
    "brachial_artery": "blood_vessel",
    "aortic_arch": "aorta",
    "thoracic_aorta": "aorta",
    "azygos_vein": "blood_vessel",
    "azygous_vein": "blood_vessel",
    "pulmonary_veins": "blood_vessel",
    "pulmonary_vein": "blood_vessel",
    "iliac_artery": "blood_vessel",
    "external_iliac_artery": "blood_vessel",
    "common_iliac_artery": "blood_vessel",
    "saphenous_vein": "blood_vessel",
    "gonadal_vessel": "blood_vessel",
    "gonadal_vein": "blood_vessel",
    "gonadal_artery": "blood_vessel",
    "umbilical_artery": "blood_vessel",
    "umbilical_vessel": "blood_vessel",

    # --- Respiratory System ---
    "airway": ["larynx", "trachea", "bronchus"],
    "lungs": "lung",
    "thyroid_cartilage": "cartilage",
    # --- NEW ADDITIONS (Respiratory) ---
    "upper_airway": ["pharynx", "larynx"],
    "subglottis": "larynx", # Region below vocal cords

    # --- Urinary System ---
    "urinary_tract": ["kidney", "ureter", "bladder", "urethra"],
    "urinary_bladder": "bladder",
    "renal_pelvis": "kidney", "renal_collecting_systems": "kidney",
    "肾": "kidney",
    # --- NEW ADDITIONS (Urinary) ---
    "renal_collecting_system": "kidney",
    "renal_cortex": "kidney",

    # --- Reproductive & Pelvic System ---
    "gonad": ["ovary", "testis"],
    "fallopian_tube": "adnexa",
    "myometrium": "uterus", "endometrium": "uterus",
    "cervical_canal": "cervix",
    "labia": "vulva", "labia_majora": "vulva", "labia_minora": "vulva", "clitoris": "vulva",
    "spermatic_cord": ["vas_deferens", "blood_vessel"],
    "vasdeferens": "vas_deferens",
    # --- NEW ADDITIONS (Reproductive/Pelvic) ---
    "corpus_spongiosum": "penis",
    "seminal_vessel": "seminal_vesicle",
    "parametria": ["uterus", "soft_tissue"], # Connective tissue around uterus
    "pelvic_floor": ["muscle", "soft_tissue"],
    "cervical_isthmus": "cervix",

    # --- Eyes and Ears ---
    "retina": "eye", "lens": "eye",
    "inner_ear": "ear", "middle_ear": "ear",
    # --- NEW ADDITIONS (Eyes/Ears) ---
    "ossicular_chain": "ear", # Bones of the middle ear
    "earlobe": "soft_tissue", # or skin
    "choroid": "eye",

    # --- General Tissues, Muscles, and Composite Regions ---
    "subcutaneous_tissue": "soft_tissue", "adipose_tissue": "soft_tissue", "fascia": "soft_tissue",
    "perineum": "soft_tissue", "perianal_skin": "skin",
    "rectus_abdominis": "muscle", "psoas_muscle": "muscle", "iliopsoas": "muscle", "psoas": "muscle",
    "chest_wall": ["soft_tissue", "muscle", "bone"],
    "abdominal_wall": ["muscle", "soft_tissue"],
    "omental": "omentum",
    # --- NEW ADDITIONS (General/Musculoskeletal) ---
    "levator_ani": "muscle",
    "levator_muscle": "muscle",
    "puborectalis_muscle": "muscle",
    "intercostal_muscle": "muscle",
    "quadratus_lumborum_muscle": "muscle",
    "triceps_muscle": "muscle",
    "iliopsoas_region": "muscle",
    "nail": "skin",
    "nail_bed": "skin",
    "hair": "skin",
    "sweat_gland": "skin",
    "oral_mucosa": "soft_tissue",
    "subcutaneous_fat": "soft_tissue",
    "greater_omentum": "omentum",
    "abdominal_cavity": "abdomen",
    "retroperitoneal_space": "retroperitoneum",
    "pelvic_cavity": "pelvis",
    "pelvic_structures": "pelvis",
    "pelvic_organs": "pelvis",
    "chest": "thorax",
    "thoracic_cavity": "thorax",
    "thoracic_cage": "thorax",
    "ribcage": "thorax",
    "mediastinal_structures": "mediastinum",
    "embryo": "fetus",
    "neonate": "fetus",
    "fetal_head": "fetus",
    "head_and_neck": ["head", "neck"],
    "neck_structures": "neck",
    "head_neck": "neck",

    # --- General Regions & Composite Structures ---
    "head_and_neck_structures": ["head", "neck"],
    "inguinal_region": ["skin", "muscle", "ligament", "soft_tissue"],
    "groin": ["skin", "muscle", "ligament", "soft_tissue"],
    "inguinal_canal": ["muscle", "ligament", "soft_tissue"],
    "buccal_region": "soft_tissue", # Cheek
    "submandibular_region": ["salivary_gland", "lymph_node", "soft_tissue"],
    "umbilicus": "skin",
    "perineal_skin": "skin",
    "pelvic_floor_muscles": "muscle",
    "urogenital_diaphragm": ["muscle", "soft_tissue"],
    "pre-peritoneal_space": "peritoneum", # The space just outside the peritoneum

    # --- Musculoskeletal ---
    "osseous_structures": "bone",
    "vertebral_bone": "bone",
    "skull_bone": "bone",
    "pelvic_ring": "bone",
    "pelvic_bone_marrow": ["pelvis", "bone"],
    "great_toe": "bone",
    "symphysis": "joint",
    "smooth_muscle": "muscle",
    "upper_trapezius_muscle": "muscle",
    "perirectal_fascia": ["rectum", "soft_tissue"],
    "prevertebral_fascia": "soft_tissue",
    "iliopsoas_bursae": "soft_tissue", # Bursa is a soft tissue structure

    # --- Nervous System ---
    "nerve_fiber": "nerve",
    "autonomic_nervous_system": "nerve",
    "Alcox_canal": ["nerve", "blood_vessel"], # pudendal canal, contains pudendal nerve/artery.

    # --- Cardiovascular & Lymphatic ---
    "peripheral_blood_vessel": "blood_vessel",
    "mesenteric_vessels": ["mesentery", "blood_vessel"],
    "periuterine_vessels": ["uterus", "blood_vessel"],
    "lung_vessel": ["lung", "blood_vessel"],
    "brachycephalic_veins": "blood_vessel",
    "mesenteric_vein": "blood_vessel",
    "common_hepatic_artery": "blood_vessel",
    "superior_mesentery_artery": "blood_vessel", # Handles misspelling
    "pudendal_artery": "blood_vessel",
    "posterior_vein": "blood_vessel",
    "basilar_artery": "blood_vessel",
    "ileocolic_artery": "blood_vessel",
    "jugular_bulb": "blood_vessel",
    "thoracic_duct": "lymph_node", # Main lymphatic vessel
    "inguinal_lymph_node": "lymph_node",
    "clavicular_lymph_node": "lymph_node",

    # --- Head, Neck, & Dental ---
    "periodontal_tissue": ["tooth", "ligament", "soft_tissue"],
    "maxillary_first_molar": "tooth",
    "maxillary_molar": "tooth",
    "maxillary_LI_region": "tooth", # Lateral Incisor region
    "maxillary_posterior_region": "tooth",
    "root": "tooth", # In dental context
    "incisive_canal": "bone", # Canal in maxilla bone
    "mandibular_canal": "bone", # Canal in mandible bone
    "nasal_passage": "sinus",
    "nasal_mucosa": "sinus", # Mucosa lining the sinus
    "ethmoidal_labyrinth": "sinus",
    "middle_turbinate": "sinus",

    # --- GI, GU & Reproductive ---
    "genitalia": ["penis", "scrotum", "vulva", "vagina"],
    "genital_organ": ["penis", "scrotum", "vulva", "vagina"],
    "genital_region": ["penis", "scrotum", "vulva", "vagina"],
    "genital_tract": ["penis", "scrotum", "vulva", "vagina", "uterus"],
    "anogenital_region": ["anal_canal", "vulva", "skin"],
    "genitourinary_tract": ["kidney", "ureter", "bladder", "urethra", "prostate", "uterus"],
    "lower_vagina": "vagina",
    "rectouterine_pouch": ["rectum", "uterus", "peritoneum"], # Pouch of Douglas
    "gastrointestinal_mucosa": "soft_tissue", # General mapping
    "mucous_membrane": "soft_tissue", # General mapping
    "midgut": "jejunum_ileum",
    "hindgut": ["colon", "rectum"],
    "splenic_gland": "spleen",
    "collecting_duct": "kidney",
    "proximal_convoluted_tubule": "kidney",
    "renal_interstitium": "kidney",
    # More
    "sinuses": "sinus"
}



