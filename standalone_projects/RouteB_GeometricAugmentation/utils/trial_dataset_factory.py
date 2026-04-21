from __future__ import annotations

from typing import Dict, List, Optional

from .aeon_fixedsplit_trials import AeonFixedSplitTrialDataset
from .finger_movements_trials import FingerMovementsTrialDataset
from .har_uci_trials import UCIHARTrialDataset
from .mitbih_trials import MITBIHBeatTrialDataset
from .natops_trials import NATOPSTrialDataset
from .regression import IEEEPPGTrialDataset
from .seed_processed_trials import SeedProcessedTrialDataset
from .seediv_trials import SEEDIVRawTrialDataset
from .seedv_trials import SEEDVRawTrialDataset
from .self_regulation_scp1_trials import SelfRegulationSCP1TrialDataset


DEFAULT_BANDS_EEG = "delta:1-4,theta:4-8,alpha:8-14,beta:14-31,gamma:31-50"
DEFAULT_BANDS_HAR = "band1:0.3-2,band2:2-5,band3:5-10,band4:10-15,band5:15-20"
DEFAULT_BANDS_MITBIH = "band1:0.5-4,band2:4-8,band3:8-16,band4:16-32,band5:32-45"
DEFAULT_BANDS_NATOPS = "band1:0.5-1.5,band2:1.5-3,band3:3-5,band4:5-7,band5:7-9"
DEFAULT_BANDS_FINGERMOVEMENTS = "band1:0.5-1.5,band2:1.5-3,band3:3-5,band4:5-7,band5:7-9"
DEFAULT_BANDS_SELFREGULATIONSCP1 = "band1:0.5-2,band2:2-4,band3:4-8,band4:8-16,band5:16-30"
DEFAULT_BANDS_BASICMOTIONS = "band1:0.2-0.8,band2:0.8-1.5,band3:1.5-2.5,band4:2.5-4.0,band5:4.0-4.8"
DEFAULT_BANDS_HANDMOVEMENTDIRECTION = "band1:0.5-1.5,band2:1.5-3,band3:3-5,band4:5-7,band5:7-9"
DEFAULT_BANDS_UWAVEGESTURELIBRARY = "band1:0.5-1.5,band2:1.5-3,band3:3-5,band4:5-7,band5:7-9"
DEFAULT_BANDS_EPILEPSY = "band1:0.2-0.8,band2:0.8-1.5,band3:1.5-3,band4:3-5,band5:5-7.5"
DEFAULT_BANDS_ATRIALFIBRILLATION = DEFAULT_BANDS_MITBIH
DEFAULT_BANDS_PENDIGITS = "band1:0.2-0.6,band2:0.6-1.2,band3:1.2-2.0,band4:2.0-3.0,band5:3.0-3.8"
DEFAULT_BANDS_AEON_GENERIC = "band1:0.02-0.06,band2:0.06-0.12,band3:0.12-0.20,band4:0.20-0.30,band5:0.30-0.45"
DEFAULT_HAR_ROOT = "data/HAR"
DEFAULT_MITBIH_NPZ = "data/MITBIH/processed/mitbih_aami44_nsvf_beats.npz"
DEFAULT_SEEDIV_ROOT = "data/SEED_IV"
DEFAULT_SEEDV_ROOT = "data/SEED_V"
DEFAULT_NATOPS_ROOT = "data/NATOPS"
DEFAULT_FINGERMOVEMENTS_ROOT = "data/FingerMovements"
DEFAULT_SELFREGULATIONSCP1_ROOT = "data/SelfRegulationSCP1"
DEFAULT_BASICMOTIONS_ROOT = "data/BasicMotions"
DEFAULT_HANDMOVEMENTDIRECTION_ROOT = "data/HandMovementDirection"
DEFAULT_UWAVEGESTURELIBRARY_ROOT = "data/UWaveGestureLibrary"
DEFAULT_EPILEPSY_ROOT = "data/Epilepsy"
DEFAULT_ATRIALFIBRILLATION_ROOT = "data/AtrialFibrillation"
DEFAULT_PENDIGITS_ROOT = "data/PenDigits"
DEFAULT_RACKETSPORTS_ROOT = "data/RacketSports"
DEFAULT_ARTICULARYWORDRECOGNITION_ROOT = "data/ArticularyWordRecognition"
DEFAULT_HEARTBEAT_ROOT = "data/Heartbeat"
DEFAULT_SELFREGULATIONSCP2_ROOT = "data/SelfRegulationSCP2"
DEFAULT_LIBRAS_ROOT = "data/Libras"
DEFAULT_JAPANESEVOWELS_ROOT = "data/JapaneseVowels"
DEFAULT_CRICKET_ROOT = "data/Cricket"
DEFAULT_HANDWRITING_ROOT = "data/Handwriting"
DEFAULT_ERING_ROOT = "data/ERing"
DEFAULT_MOTORIMAGERY_ROOT = "data/MotorImagery"
DEFAULT_ETHANOLCONCENTRATION_ROOT = "data/EthanolConcentration"
DEFAULT_IEEEPPG_ROOT = "data/regression/aeon"


def normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in {"seed", "seed1"}:
        return "seed1"
    if key in {"har", "uci_har", "ucihar"}:
        return "har"
    if key in {"mitbih", "mit-bih", "mit_bih", "mitbih_arrhythmia"}:
        return "mitbih"
    if key in {"seediv", "seed-iv", "seed_iv", "seediv_raw"}:
        return "seediv"
    if key in {"seedv", "seed-v", "seed_v"}:
        return "seedv"
    if key in {"natops", "natops_uea"}:
        return "natops"
    if key in {"fingermovements", "finger_movements", "finger-movements", "finger"}:
        return "fingermovements"
    if key in {"selfregulationscp1", "self_regulation_scp1", "selfregulation", "scp1"}:
        return "selfregulationscp1"
    if key in {"basicmotions", "basic_motions", "basic-motions"}:
        return "basicmotions"
    if key in {"handmovementdirection", "hand_movement_direction", "hand-movement-direction"}:
        return "handmovementdirection"
    if key in {"uwavegesturelibrary", "u_wave_gesture_library", "uwave", "uwavegesture"}:
        return "uwavegesturelibrary"
    if key in {"epilepsy"}:
        return "epilepsy"
    if key in {"atrialfibrillation", "atrial_fibrillation", "atrial-fibrillation"}:
        return "atrialfibrillation"
    if key in {"pendigits", "pen_digits", "pen-digits"}:
        return "pendigits"
    if key in {"racketsports", "racket_sports", "racket-sports"}:
        return "racketsports"
    if key in {"articularywordrecognition", "articulary_word_recognition", "articulary-word-recognition", "awr"}:
        return "articularywordrecognition"
    if key in {"heartbeat", "heart_beat", "heart-beat"}:
        return "heartbeat"
    if key in {"selfregulationscp2", "self_regulation_scp2", "scp2"}:
        return "selfregulationscp2"
    if key in {"libras"}:
        return "libras"
    if key in {"japanesevowels", "japanese_vowels", "japanese-vowels"}:
        return "japanesevowels"
    if key in {"cricket"}:
        return "cricket"
    if key in {"handwriting", "hand_writing", "hand-writing"}:
        return "handwriting"
    if key in {"ering", "e_ring", "e-ring"}:
        return "ering"
    if key in {"motorimagery", "motor_imagery", "motor-imagery"}:
        return "motorimagery"
    if key in {"ethanolconcentration", "ethanol_concentration", "ethanol-concentration"}:
        return "ethanolconcentration"
    if key in {"ieeeppg", "ieee_ppg"}:
        return "ieeeppg"
    raise ValueError(f"Unsupported dataset: {name}")


def resolve_band_spec(dataset: str, bands: str) -> str:
    ds = normalize_dataset_name(dataset)
    raw = str(bands or "").strip()
    if not raw:
        if ds == "har":
            return DEFAULT_BANDS_HAR
        if ds == "mitbih":
            return DEFAULT_BANDS_MITBIH
        if ds == "natops":
            return DEFAULT_BANDS_NATOPS
        if ds == "fingermovements":
            return DEFAULT_BANDS_FINGERMOVEMENTS
        if ds == "selfregulationscp1":
            return DEFAULT_BANDS_SELFREGULATIONSCP1
        if ds == "basicmotions":
            return DEFAULT_BANDS_BASICMOTIONS
        if ds == "handmovementdirection":
            return DEFAULT_BANDS_HANDMOVEMENTDIRECTION
        if ds == "uwavegesturelibrary":
            return DEFAULT_BANDS_UWAVEGESTURELIBRARY
        if ds == "epilepsy":
            return DEFAULT_BANDS_EPILEPSY
        if ds == "atrialfibrillation":
            return DEFAULT_BANDS_ATRIALFIBRILLATION
        if ds == "pendigits":
            return DEFAULT_BANDS_PENDIGITS
        if ds in {
            "racketsports",
            "articularywordrecognition",
            "heartbeat",
            "selfregulationscp2",
            "libras",
            "japanesevowels",
            "cricket",
            "handwriting",
            "ering",
            "motorimagery",
            "ethanolconcentration",
        }:
            return DEFAULT_BANDS_AEON_GENERIC
        return DEFAULT_BANDS_EEG
    if ds == "har" and raw == DEFAULT_BANDS_EEG:
        # If user keeps EEG default unchanged, auto-switch to HAR-safe bands.
        return DEFAULT_BANDS_HAR
    if ds == "mitbih" and raw == DEFAULT_BANDS_EEG:
        # Avoid blindly using EEG ranges on ECG/heart-beat signals.
        return DEFAULT_BANDS_MITBIH
    if ds == "natops" and raw == DEFAULT_BANDS_EEG:
        # NATOPS sfreq is low; EEG gamma ranges exceed Nyquist.
        return DEFAULT_BANDS_NATOPS
    if ds == "fingermovements" and raw == DEFAULT_BANDS_EEG:
        # FingerMovements time-series should use low bands under low effective sfreq.
        return DEFAULT_BANDS_FINGERMOVEMENTS
    if ds == "selfregulationscp1" and raw == DEFAULT_BANDS_EEG:
        # SCP1 is low-frequency EEG; avoid blindly using SEED gamma defaults.
        return DEFAULT_BANDS_SELFREGULATIONSCP1
    if ds == "basicmotions" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_BASICMOTIONS
    if ds == "handmovementdirection" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_HANDMOVEMENTDIRECTION
    if ds == "uwavegesturelibrary" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_UWAVEGESTURELIBRARY
    if ds == "epilepsy" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_EPILEPSY
    if ds == "atrialfibrillation" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_ATRIALFIBRILLATION
    if ds == "pendigits" and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_PENDIGITS
    if ds in {
        "racketsports",
        "articularywordrecognition",
        "heartbeat",
        "selfregulationscp2",
        "libras",
        "japanesevowels",
        "cricket",
        "handwriting",
        "ering",
        "motorimagery",
        "ethanolconcentration",
    } and raw == DEFAULT_BANDS_EEG:
        return DEFAULT_BANDS_AEON_GENERIC
    return raw


def load_trials_for_dataset(
    *,
    dataset: str,
    processed_root: Optional[str] = None,
    stim_xlsx: Optional[str] = None,
    har_root: Optional[str] = None,
    mitbih_npz: Optional[str] = None,
    seediv_root: Optional[str] = None,
    seedv_root: Optional[str] = None,
    natops_root: Optional[str] = None,
    fingermovements_root: Optional[str] = None,
    selfregulationscp1_root: Optional[str] = None,
    basicmotions_root: Optional[str] = None,
    handmovementdirection_root: Optional[str] = None,
    uwavegesturelibrary_root: Optional[str] = None,
    epilepsy_root: Optional[str] = None,
    atrialfibrillation_root: Optional[str] = None,
    pendigits_root: Optional[str] = None,
    racketsports_root: Optional[str] = None,
    articularywordrecognition_root: Optional[str] = None,
    heartbeat_root: Optional[str] = None,
    selfregulationscp2_root: Optional[str] = None,
    libras_root: Optional[str] = None,
    japanesevowels_root: Optional[str] = None,
    cricket_root: Optional[str] = None,
    handwriting_root: Optional[str] = None,
    ering_root: Optional[str] = None,
    motorimagery_root: Optional[str] = None,
    ethanolconcentration_root: Optional[str] = None,
    ieeeppg_root: Optional[str] = None,
) -> List[Dict]:
    ds = normalize_dataset_name(dataset)
    if ds == "seed1":
        if not processed_root:
            raise ValueError("processed_root is required for dataset=seed1")
        if not stim_xlsx:
            raise ValueError("stim_xlsx is required for dataset=seed1")
        it = SeedProcessedTrialDataset(processed_root, stim_xlsx)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "har":
        root = har_root or DEFAULT_HAR_ROOT
        it = UCIHARTrialDataset(root=root, include_splits=("train", "test"), sfreq=50.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "seediv":
        root = seediv_root or DEFAULT_SEEDIV_ROOT
        it = SEEDIVRawTrialDataset(root=root, include_sessions=(1, 2, 3), sfreq=200.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "seedv":
        root = seedv_root or DEFAULT_SEEDV_ROOT
        it = SEEDVRawTrialDataset(root=root, include_sessions=(1, 2, 3), sfreq=200.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "natops":
        root = natops_root or DEFAULT_NATOPS_ROOT
        it = NATOPSTrialDataset(root=root, include_splits=("train", "test"), sfreq=20.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "fingermovements":
        root = fingermovements_root or DEFAULT_FINGERMOVEMENTS_ROOT
        it = FingerMovementsTrialDataset(root=root, include_splits=("train", "test"), sfreq=20.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "selfregulationscp1":
        root = selfregulationscp1_root or DEFAULT_SELFREGULATIONSCP1_ROOT
        it = SelfRegulationSCP1TrialDataset(root=root, include_splits=("train", "test"), sfreq=256.0)
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "basicmotions":
        root = basicmotions_root or DEFAULT_BASICMOTIONS_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "handmovementdirection":
        root = handmovementdirection_root or DEFAULT_HANDMOVEMENTDIRECTION_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "uwavegesturelibrary":
        root = uwavegesturelibrary_root or DEFAULT_UWAVEGESTURELIBRARY_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "epilepsy":
        root = epilepsy_root or DEFAULT_EPILEPSY_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "atrialfibrillation":
        root = atrialfibrillation_root or DEFAULT_ATRIALFIBRILLATION_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "pendigits":
        root = pendigits_root or DEFAULT_PENDIGITS_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "racketsports":
        root = racketsports_root or DEFAULT_RACKETSPORTS_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "articularywordrecognition":
        root = articularywordrecognition_root or DEFAULT_ARTICULARYWORDRECOGNITION_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "heartbeat":
        root = heartbeat_root or DEFAULT_HEARTBEAT_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "selfregulationscp2":
        root = selfregulationscp2_root or DEFAULT_SELFREGULATIONSCP2_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "libras":
        root = libras_root or DEFAULT_LIBRAS_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "japanesevowels":
        root = japanesevowels_root or DEFAULT_JAPANESEVOWELS_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "cricket":
        root = cricket_root or DEFAULT_CRICKET_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "handwriting":
        root = handwriting_root or DEFAULT_HANDWRITING_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "ering":
        root = ering_root or DEFAULT_ERING_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "motorimagery":
        root = motorimagery_root or DEFAULT_MOTORIMAGERY_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "ethanolconcentration":
        root = ethanolconcentration_root or DEFAULT_ETHANOLCONCENTRATION_ROOT
        it = AeonFixedSplitTrialDataset(root=root, dataset_key=ds, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    if ds == "ieeeppg":
        root = ieeeppg_root or DEFAULT_IEEEPPG_ROOT
        it = IEEEPPGTrialDataset(root=root, include_splits=("train", "test"))
        return sorted(list(it), key=lambda x: str(x["trial_id_str"]))

    npz_path = mitbih_npz or DEFAULT_MITBIH_NPZ
    it = MITBIHBeatTrialDataset(npz_path=npz_path, include_splits=("train", "test"))
    return sorted(list(it), key=lambda x: str(x["trial_id_str"]))
