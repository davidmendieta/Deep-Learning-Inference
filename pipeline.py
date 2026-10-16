import copy
import json
import time
import traceback
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorrt as trt
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------config
DATA = "train.csv"
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# Chronological split by trading day (whole days stay together, no leakage)
TRAIN_END = 320   # train: date_id < 320
VAL_END = 400     # val (early stopping / tuning): 320 <= date_id < 400
                  # test (final numbers): date_id >= 400
SEED = 42
EPS = 1e-9
torch.manual_seed(SEED)
np.random.seed(SEED)

ONNX_PATH = ART / "mlp.onnx"
HALF_ONNX = ART / "mlp_fp16.onnx"
MAXB = 131072
BATCH_SIZES = [1, 64, 1024, 16384, 131072]
EVAL_N = 100_000          # evaluate accuracy on rows [0, EVAL_N)
CALIB_START = 100_000     # calibrate INT8 on rows [CALIB_START, end): disjoint from eval
DEV = "cuda"
LOGGER = trt.Logger(trt.Logger.WARNING)


# -----------features
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).reset_index(drop=True)
    f = pd.DataFrame(index=df.index)

    mid = (df.bid_price + df.ask_price) / 2
    book_size = df.bid_size + df.ask_size

    f["seconds_in_bucket"] = df.seconds_in_bucket
    f["spread_bps"] = (df.ask_price - df.bid_price) * 1e4
    f["book_imbalance"] = (df.bid_size - df.ask_size) / (book_size + EPS)
    f["log_book_size"] = np.log1p(book_size)
    f["wap_dev_bps"] = (df.wap - mid) * 1e4

    signed_imb = df.imbalance_size * df.imbalance_buy_sell_flag
    f["signed_imbalance_log"] = np.sign(signed_imb) * np.log1p(np.abs(signed_imb))
    f["imbalance_to_matched"] = df.imbalance_size / (df.matched_size + EPS)
    f["signed_imbalance_to_matched"] = signed_imb / (df.matched_size + EPS)
    f["matched_to_book"] = df.matched_size / (book_size + EPS)
    f["imbalance_flag"] = df.imbalance_buy_sell_flag

    # Auction-vs-continuous price gaps 
    f["ref_wap_gap_bps"] = (df.reference_price - df.wap) * 1e4
    f["near_wap_gap_bps"] = (df.near_price.fillna(df.wap) - df.wap) * 1e4
    f["far_wap_gap_bps"] = (df.far_price.fillna(df.wap) - df.wap) * 1e4
    f["near_missing"] = df.near_price.isna().astype(np.float32)
    f["far_missing"] = df.far_price.isna().astype(np.float32)

    # Short-term momentum within each stock-day
    g = df.groupby(["stock_id", "date_id"])
    for k in (1, 3, 6):
        f[f"wap_ret_{k}_bps"] = g["wap"].pct_change(k) * 1e4
    f["imbalance_change"] = g["imbalance_size"].diff() / (df.matched_size + EPS)

    f = f.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return f, df


# ----------------- helpers
def metrics(y_true, p):
    return {
        "MAE": float(mean_absolute_error(y_true, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, p))),
    }


def report(name, y_true, p, store):
    m = metrics(y_true, p)
    store[name] = m
    print(f"{name:<22} MAE {m['MAE']:.4f}   RMSE {m['RMSE']:.4f}")


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        layers, prev = [], d
        for h in (256, 128, 64):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------train
def train_and_save():
    t0 = time.time()
    raw = pd.read_csv(DATA)
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns")
    print(f"Duplicates: {raw.duplicated(['stock_id','date_id','seconds_in_bucket']).sum()}  "
          f"Null target: {raw.target.isna().sum()}")
    raw = raw.dropna(subset=["target"])

    X_feat, df = make_features(raw)
    y_all = df["target"].to_numpy(np.float32)
    days = df["date_id"].to_numpy()
    feature_names = list(X_feat.columns)
    print(f"Engineered features: {len(feature_names)}")

    tr, va, te = days < TRAIN_END, (days >= TRAIN_END) & (days < VAL_END), days >= VAL_END
    print(f"Split rows  train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")

    # Impute with train medians, standardize with train stats, clip outliers
    Xv = X_feat.to_numpy(np.float32)
    med = np.nanmedian(Xv[tr], axis=0)
    Xv = np.where(np.isnan(Xv), med, Xv)
    mean, std = Xv[tr].mean(0), Xv[tr].std(0) + 1e-6
    Xv = np.clip((Xv - mean) / std, -10, 10).astype(np.float32)
    np.savez(ART / "scaler.npz", median=med, mean=mean, std=std)
    json.dump(feature_names, open(ART / "features.json", "w"))

    Xtr, ytr, Xva, yva, Xte, yte = Xv[tr], y_all[tr], Xv[va], y_all[va], Xv[te], y_all[te]
    results = {}
    report("Predict-zero", yte, np.zeros_like(yte), results)

    # --- Ridge
    ridge = Ridge(alpha=10.0).fit(Xtr, ytr)
    report("Ridge", yte, ridge.predict(Xte), results)

    # --- Random Forest (subsampled: full 5M rows is too slow)
    idx = np.random.choice(len(Xtr), size=min(400_000, len(Xtr)), replace=False)
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=50,
                               n_jobs=-1, random_state=SEED).fit(Xtr[idx], ytr[idx])
    report("RandomForest", yte, rf.predict(Xte), results)

    # --- LightGBM (early stopping on val, final numbers on test)
    gbm = lgb.LGBMRegressor(objective="l1", n_estimators=1000, learning_rate=0.05,
                            num_leaves=63, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, random_state=SEED, verbose=-1)
    gbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    report("LightGBM", yte, gbm.predict(Xte), results)

    # --- PyTorch MLP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MLP training on {dev}")
    Xtr_t, ytr_t = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xva_t, yva_t = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)
    net = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.L1Loss()
    BATCH, EPOCHS, PATIENCE = 4096, 20, 4
    best, best_state, bad = float("inf"), None, 0

    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(net(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_mae = float(loss_fn(net(Xva_t), yva_t))
        sched.step(val_mae)
        print(f"  epoch {ep+1:02d}  val MAE {val_mae:.4f}")
        if val_mae < best - 1e-5:
            best, bad = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        mlp_pred = net(torch.tensor(Xte, device=dev)).cpu().numpy()
    report("PyTorch MLP", yte, mlp_pred, results)

    # --- Save artifacts for deployment
    torch.save(net.state_dict(), ART / "mlp.pt")
    np.save(ART / "test_X.npy", Xte[:200_000])
    np.save(ART / "test_y.npy", yte[:200_000])
    np.save(ART / "mlp_pred_test.npy", mlp_pred[:200_000])
    joblib.dump(rf, ART / "rf.joblib")
    json.dump({"n_features": len(feature_names), "results": results},
              open(ART / "day1_results.json", "w"), indent=2)
    print(f"\nStage 1 done in {(time.time()-t0)/60:.1f} min. Saved to {ART}/")


# Deployment state, loaded from the artifacts
X = y = model = X_gpu = None
D = None


def load_deploy_state():
    global X, y, D, model, X_gpu
    X = np.load(ART / "test_X.npy")
    y = np.load(ART / "test_y.npy")
    D = X.shape[1]
    model = MLP(D)
    model.load_state_dict(torch.load(ART / "mlp.pt", map_location="cpu"))
    model.eval().to(DEV)
    X_gpu = torch.tensor(X, device=DEV)


# ---------------------ONNX
def export_onnx():
    dummy = torch.randn(1024, D, device=DEV)
    kwargs = dict(input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(ONNX_PATH), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(ONNX_PATH), **kwargs)
    print(f"Exported {ONNX_PATH}")


def verify_onnx():
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    xs = X[:10_000]
    onnx_out = sess.run(None, {"input": xs})[0].reshape(-1)
    with torch.no_grad():
        torch_out = model(torch.tensor(xs, device=DEV)).cpu().numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    print(f"ONNX vs PyTorch max abs diff: {diff:.2e}")
    return diff


def export_half_onnx():
    mh = copy.deepcopy(model).half().eval()
    dummy = torch.randn(1024, D, device=DEV).half()
    kw = dict(input_names=["input"], output_names=["output"],
              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(mh, dummy, str(HALF_ONNX), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(mh, dummy, str(HALF_ONNX), **kw)


# ------------------TensorRT
class TRTRunner:
    def __init__(self, plan: bytes):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.out = torch.empty(MAXB, device=DEV, dtype=torch.float32)  # preallocated

    def __call__(self, x):
        b = x.shape[0]
        self.ctx.set_input_shape("input", (b, x.shape[1]))
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.set_tensor_address("output", self.out.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out[:b]


class TRTRunnerHalf(TRTRunner):
    def __init__(self, plan):
        super().__init__(plan)
        self.out = torch.empty(MAXB, device=DEV, dtype=torch.float16)


def build(onnx_path, fp16=False, calib=None, detailed=False, cls=TRTRunner):
    builder = trt.Builder(LOGGER)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, D), (1024, D), (MAXB, D))
    config.add_optimization_profile(profile)
    if calib is not None:  # INT8 only: FP16 flag deliberately NOT set
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        config.set_calibration_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Engine build failed")
    return cls(bytes(plan))


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, bs=1024):
        super().__init__()
        self.data, self.bs, self.i = data, bs, 0
        self.buf = torch.empty(bs, D, device=DEV)

    def get_batch_size(self):
        return self.bs

    def get_batch(self, names):
        if self.i + self.bs > len(self.data):
            return None
        self.buf.copy_(self.data[self.i:self.i + self.bs])
        self.i += self.bs
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def layer_report(engine, title):
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])
    n16 = 0
    print(f"\n{title}: {len(layers)} fused layers")
    for L in layers:
        if not isinstance(L, dict):
            print("  ", L)
            continue
        outs = L.get("Outputs", [])
        fmt_ = outs[0].get("Format/Datatype", "?") if outs else "?"
        n16 += ("FP16" in fmt_) or ("Half" in fmt_)
        print(f"  {str(L.get('LayerType', '?')):<14} {fmt_:<36} {str(L.get('Name', ''))[:70]}")
    print(f"  -> {n16}/{len(layers)} layers produce FP16 output")
    return {"layers": len(layers), "fp16_layers": n16}


# ---------------------benchmarking
def time_gpu(fn, x, warmup=30, iters=200):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms per call


def time_cpu(fn, x, warmup=1, iters=5):
    for _ in range(warmup):
        fn(x)
    t = time.perf_counter()
    for _ in range(iters):
        fn(x)
    return (time.perf_counter() - t) / iters * 1000


def predict(fn, xs, chunk=16384):
    outs = [fn(xs[i:i + chunk]).clone() for i in range(0, len(xs), chunk)]
    torch.cuda.synchronize()
    return torch.cat(outs).cpu().numpy()


def mae(pred, yy):
    return float(np.mean(np.abs(pred - yy)))


def torch_fn(x):
    with torch.inference_mode():
        return model(x)


def make_graph_fn(b):
    static = X_gpu[:b].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            torch_fn(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch_fn(static)
    return lambda x: g.replay()


def make_compiled(mode):
    torch._dynamo.reset()
    m = torch.compile(model, mode=mode, dynamic=False)

    def fn(x):
        with torch.inference_mode():
            return m(x)
    return fn


def fmt(v):
    return f"{v:>10.3f}" if v is not None else f"{'n/a':>10}"


def deploy_and_benchmark():
    assert torch.cuda.is_available(), "Needs an NVIDIA GPU"
    load_deploy_state()
    out = {}

    export_onnx()
    out["onnx_max_abs_diff"] = verify_onnx()

    ev_x, ev_y = X_gpu[:EVAL_N], y[:EVAL_N]
    ev_x_half = ev_x.half()

    print("Building engines...")
    fp32 = build(ONNX_PATH)
    export_half_onnx()
    fp16 = build(HALF_ONNX, fp16=True, detailed=True, cls=TRTRunnerHalf)
    out["fp16_layer_check"] = layer_report(fp16.engine, "TensorRT FP16 engine")
    int8 = build(ONNX_PATH, calib=Calibrator(X_gpu[CALIB_START:]))
    print("INT8-only engine built (FP16 flag off)")
    rf = joblib.load(ART / "rf.joblib")

    acc = {"PyTorch FP32": mae(predict(torch_fn, ev_x), ev_y),
           "TensorRT FP32": mae(predict(fp32, ev_x), ev_y),
           "TensorRT FP16": mae(predict(fp16, ev_x_half).astype(np.float32), ev_y),
           "TensorRT INT8-only": mae(predict(int8, ev_x), ev_y)}
    base = acc["PyTorch FP32"]
    print(f"\nAccuracy on {EVAL_N:,} held-out rows")
    for k, v in acc.items():
        print(f"  {k:<20} MAE {v:.5f}   ({(v - base) / base * 100:+.4f}% vs PyTorch)")
    out["mae"] = acc

    rows = []
    print(f"\n{'batch':>8} {'eager':>10} {'torch+graph':>11} {'TRT FP32':>10} "
          f"{'TRT FP16':>10} {'TRT INT8':>10} {'RF (CPU)':>10}")
    for b in BATCH_SIZES:
        xb = X_gpu[:b].contiguous()
        iters = 200 if b <= 16384 else 50
        r = {"batch": b,
             "eager": time_gpu(torch_fn, xb, iters=iters),
             "torch_cuda_graph": time_gpu(make_graph_fn(b), xb, iters=iters),
             "trt_fp32": time_gpu(fp32, xb, iters=iters),
             "trt_fp16": time_gpu(fp16, xb.half(), iters=iters),
             "trt_int8": time_gpu(int8, xb, iters=iters),
             "rf_cpu_ms": time_cpu(rf.predict, X[:b], iters=10 if b <= 1024 else 3)}
        rows.append(r)
        print(f"{b:>8} {fmt(r['eager'])} {fmt(r['torch_cuda_graph']):>11} {fmt(r['trt_fp32'])} "
              f"{fmt(r['trt_fp16'])} {fmt(r['trt_int8'])} {fmt(r['rf_cpu_ms'])}")
    out["latency_ms"] = rows

    print("\ntorch.compile attempt (full traceback if it fails):")
    try:
        f = make_compiled("default")
        print(f"  compile OK, batch 1024: {time_gpu(f, X_gpu[:1024].contiguous()):.3f} ms")
    except Exception:
        traceback.print_exc()

    json.dump(out, open(ART / "deploy_results.json", "w"), indent=2)
    print(f"\nSaved {ART / 'deploy_results.json'}")




def main():
    train_and_save()
    with torch.cuda.stream(torch.cuda.Stream()):
        deploy_and_benchmark()


if __name__ == "__main__":
    main()
