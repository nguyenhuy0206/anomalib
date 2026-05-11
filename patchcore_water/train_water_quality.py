import os
import numpy as np
import pandas as pd
import time
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch.loggers import CSVLogger

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

OUTDIR = "./results_paper"
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_CSV = os.path.join(OUTDIR, "train_normal_scores.csv")
TEST_CSV = os.path.join(OUTDIR, "test_scores.csv")

def measure_predict_time(engine, model, dataloader, warmup_batches=3):
    # warmup (GPU/MPS often needs warmup)
    _ = engine.predict(model=model, dataloaders=dataloader, ckpt_path=None)

    # đo lại lần 2 để ổn định
    t0 = time.perf_counter()
    _ = engine.predict(model=model, dataloaders=dataloader, ckpt_path=None)
    t1 = time.perf_counter()

    n_images = len(dataloader.dataset)
    total_time = t1 - t0
    return {
        "total_sec": total_time,
        "sec_per_image": total_time / n_images,
        "fps": n_images / total_time,
        "n_images": n_images,
    }
    
def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


def extract_rows(pred_batches):
    rows = []

    def handle_one(b):
        nonlocal rows

        # Case A: dict style
        if isinstance(b, dict):
            paths = b.get("image_path") or b.get("path")
            labels = b.get("gt_label") or b.get("label")
            scores = b.get("pred_score") or b.get("anomaly_score")

            if paths is None or labels is None or scores is None:
                return

            labels = to_numpy(labels).astype(int).tolist()
            scores = to_numpy(scores).astype(float).tolist()
            for p, y, s in zip(paths, labels, scores):
                rows.append({"image_path": str(p), "label": int(y), "score": float(s)})
            return

        # Case B: ImageBatch dataclass
        if hasattr(b, "image_path") and hasattr(b, "gt_label") and hasattr(b, "pred_score"):
            paths = list(b.image_path)
            labels = to_numpy(b.gt_label).astype(int).tolist()
            scores = to_numpy(b.pred_score).astype(float).tolist()
            for p, y, s in zip(paths, labels, scores):
                rows.append({"image_path": str(p), "label": int(y), "score": float(s)})
            return

        # Case list-of-batches
        if isinstance(b, list):
            for bb in b:
                handle_one(bb)

    for b in pred_batches:
        handle_one(b)

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["label"].astype(int).values
    y_score = df["score"].astype(float).values
    y_pred = (y_score >= threshold).astype(int)

    # guard nếu test bị thiếu 1 class
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # confusion matrix luôn 2x2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "image_AUROC": float(auroc),
        "image_AP": float(ap),
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "n_samples": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def infer_cam(path: str) -> str:
    p = str(path)
    if "cam1_" in p:
        return "cam1"
    if "cam2_" in p:
        return "cam2"
    return "unk"


def main():
    # 1) Datamodule (full image setup)
    datamodule = Folder(
        name="water_quality",
        root="./data/water_quality",
        normal_dir="train/normal",
        normal_test_dir="test/normal",
        abnormal_dir="test/abnormal",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
    )

    # 2) Model
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    model.pre_processor = Patchcore.configure_pre_processor(
        image_size=(256, 256),
        center_crop_size=(128, 128)
    )

    # 3) Engine
    logger = CSVLogger(save_dir=OUTDIR, name="metrics")
    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        default_root_dir=OUTDIR,
        logger=logger,
    )

    print("=== FIT (build memory bank from TRAIN NORMAL) ===")
    engine.fit(model=model, datamodule=datamodule)

    # --- TRAIN scores for threshold ---
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()

    print("=== PREDICT TRAIN (train/normal) for threshold ===")
    preds_train = engine.predict(model=model, dataloaders=train_loader)
    df_train = extract_rows(preds_train)

    if df_train.empty:
        raise RuntimeError("df_train is empty. Check extract_rows / anomalib version output keys.")

    df_train.to_csv(TRAIN_CSV, index=False)

    threshold = float(np.percentile(df_train["score"].values, 95))
    print(f"[THRESHOLD] 95th percentile of train-normal = {threshold:.6f}")

    # --- TEST scores ---
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()


    timing = measure_predict_time(engine, model, test_loader)
    print("=== INFERENCE TIME (Engine.predict end-to-end) ===")
    print(timing)
    
    print("=== PREDICT TEST for per-image scores ===")
    preds_test = engine.predict(model=model, dataloaders=test_loader)
    df_test = extract_rows(preds_test)

    if df_test.empty:
        raise RuntimeError("df_test is empty. Check extract_rows / anomalib version output keys.")

    df_test["cam"] = df_test["image_path"].apply(infer_cam)
    df_test.to_csv(TEST_CSV, index=False)

    # --- Metrics overall ---
    overall = compute_metrics(df_test, threshold)
    df_overall = pd.DataFrame([overall])
    overall_path = os.path.join(OUTDIR, "exp1_metrics_overall.csv")
    df_overall.to_csv(overall_path, index=False)

    print("\n=== EXP1 OVERALL METRICS ===")
    for k in ["image_AUROC", "image_AP", "precision", "recall", "f1", "TP", "FP", "TN", "FN", "n_samples"]:
        print(f"{k:>12}: {overall[k]}")

    # --- Metrics by cam (nếu cam nào thiếu 1 class => AUROC = nan, vẫn report F1/PR/Rec) ---
    by_cam_rows = []
    for cam, g in df_test.groupby("cam"):
        m = compute_metrics(g, threshold)
        m["cam"] = cam
        by_cam_rows.append(m)

    df_by_cam = pd.DataFrame(by_cam_rows).sort_values("cam")
    by_cam_path = os.path.join(OUTDIR, "exp1_metrics_by_cam.csv")
    df_by_cam.to_csv(by_cam_path, index=False)

    print("\n=== EXP1 METRICS BY CAM ===")
    print(df_by_cam[["cam", "n_samples", "n_pos", "n_neg", "image_AUROC", "f1", "precision", "recall", "TP", "FP", "TN", "FN"]])

    # --- Quick sanity prints (giống log bạn hay xem) ---
    print("\n=== TEST DISTRIBUTION (cam, label) ===")
    print(df_test.groupby(["cam", "label"]).size())

    print("\n=== SCORE STATS BY CAM ===")
    print(df_test.groupby("cam")["score"].describe())


if __name__ == "__main__":
    main()
