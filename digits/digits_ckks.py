import math
import numpy as np
import openfhe as fhe
import gc
import ctypes
import csv
libc = ctypes.CDLL("libc.so.6")
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import ndtr

# ─────────────────────────────────────────────────────────────
# Parametri problema — DIGITS (multiclasse, 10 classi)
# ─────────────────────────────────────────────────────────────

SEP = "=" * 60

print(SEP)
print("MODALITÀ")
print(SEP)
risposta_modalita = input(
    "Versione SICURA (128-bit, ring dim scelto automaticamente) "
    "o NON SICURA (HEStd_NotSet, RingDim=2^14 fisso)? [s/n]: "
).strip().lower()
SECURE = risposta_modalita in ("s", "si", "sì", "y", "yes")

if SECURE:
    print("  → modalità SICURA: HEStd_128_classic, ring dim automatico")
else:
    print("  → modalità NON SICURA: HEStd_NotSet, RingDim=2^14, SLOT=8192")
print(SEP)

digits = load_digits()
X_all = StandardScaler().fit_transform(digits.data)
X_raw, _, _, _ = train_test_split(X_all, digits.target, test_size=0.2, random_state=42)

M, L = 64, 16
C = 10 
N = None
N_MINI_BATCHES = None
RING_DIM = 1 << 14
SLOT = RING_DIM // 2

# ─────────────────────────────────────────────────────────────
# Helpers numpy
# ─────────────────────────────────────────────────────────────

def plain_flat_e_repeat_righe(X_flat, n_rep, N, M, slot):
    out = np.zeros(slot)
    for i in range(N):
        for k in range(n_rep):
            src  = i * M
            dest = i * n_rep * M + k * M
            out[dest:dest+M] = X_flat[src:src+M]
    return out

def gelu(x):
    return x * ndtr(x)

def gelu_prime(x):
    return ndtr(x) + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# ─────────────────────────────────────────────────────────────
# Setup contesto CKKS
# ─────────────────────────────────────────────────────────────

def build_context(secure):
    params = fhe.CCParamsCKKSRNS()
    secret_key_dist = fhe.SecretKeyDist.SPARSE_TERNARY
    params.SetSecretKeyDist(secret_key_dist)

    if secure:
        params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    else:
        params.SetSecurityLevel(fhe.SecurityLevel.HEStd_NotSet)
        params.SetRingDim(RING_DIM)
    dcrt_bits    = 55
    first_mod    = 56
    params.SetScalingModSize(dcrt_bits)
    params.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
    params.SetFirstModSize(first_mod)
    level_budget    = [3, 3]
    bootstrap_depth = fhe.FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)
    levels_before   = 6
    depth           = levels_before + bootstrap_depth
    params.SetMultiplicativeDepth(depth)
    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(fhe.PKESchemeFeature.ADVANCEDSHE)
    cc.Enable(fhe.PKESchemeFeature.FHE)
    return cc, level_budget, bootstrap_depth, depth


def calibra_post_boot(cc, kp, depth):
    print(f"  [calibrazione] depth={depth}, limite={depth-2}")
    vec = [0.1] * SLOT
    pt  = cc.MakeCKKSPackedPlaintext(vec, 1, 0, None, SLOT)
    ct  = cc.Encrypt(kp.publicKey, pt)
    for _ in range(depth - 2):
        pt2 = cc.MakeCKKSPackedPlaintext([1.0]*SLOT, 1, ct.GetLevel(), None, SLOT)
        ct  = cc.EvalMult(ct, pt2)
    print(f"  [calibrazione] livello prima del bootstrap: {ct.GetLevel()}")
    ct  = cc.EvalBootstrap(ct)
    post = ct.GetLevel()
    print(f"  [calibrazione] livello dopo bootstrap:      {post}")
    livelli_utili = (depth - 2) - post
    print(f"  [calibrazione] livelli utili per ciclo:     {livelli_utili}")
    return post


def compute_needed_rotations(N, M, L, C, slot):
    needed = set()

    def add_bsgs_positions(count, step):
        baby  = max(1, int(math.isqrt(count)))
        giant = math.ceil(count / baby)
        for b in range(1, baby):
            needed.add(b * step); needed.add(-(b * step))
        for g in range(1, giant):
            needed.add(g * baby * step); needed.add(-(g * baby * step))

    def add_transpose(N_mat, M_mat):
        for i in range(1, N_mat): needed.add(i * (M_mat - 1))
        for j in range(1, M_mat): needed.add(-(j * (N_mat - 1)))

    def add_flat_repeat(n_rep, N_p, M_p):
        for i in range(1, N_p):
            needed.add(-(i * M_p * (n_rep - 1)))
        add_bsgs_positions(n_rep, M_p)

    add_flat_repeat(C, N, L)
    add_flat_repeat(C, L, N)
    add_flat_repeat(L, N, C)

    for N_mat, M_mat in [(M, L), (L, C), (N, C), (C, L), (N, L)]:
        add_transpose(N_mat, M_mat)
    for n_rep, L2 in [(N, M*L), (N, L*C), (L, N*C), (N, C*L), (M, N*L)]:
        add_bsgs_positions(n_rep, L2)

    for M_p in {M, L, N, C}:
        add_bsgs_positions(M_p, 1)
    for M_p, N_p, L_p in [(M, N, L), (L, N, C), (N, L, C), (C, N, L), (N, M, L)]:
        if M_p == 1: continue
        add_bsgs_positions(N_p * L_p, M_p - 1)

    for N_r, M_r in [(N, C), (N, L)]:
        add_bsgs_positions(N_r, M_r)

    for M_bias in [L, C]:
        add_bsgs_positions(N, M_bias)

    needed.discard(0)
    return sorted(needed)

def gen_keys(cc, level_budget):
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    rotations = compute_needed_rotations(N, M, L, C, SLOT)
    print(f"  Chiavi di rotazione: {len(rotations)}")
    cc.EvalRotateKeyGen(kp.secretKey, rotations)
    cc.EvalBootstrapSetup(level_budget, [0, 0], SLOT)
    cc.EvalBootstrapKeyGen(kp.secretKey, SLOT)
    return kp

def force_slots(cc, ct):
    pt = cc.MakeCKKSPackedPlaintext([0.0]*SLOT, 1, ct.GetLevel(), None, SLOT)
    return cc.EvalAdd(ct, pt)

# ─────────────────────────────────────────────────────────────
# Helpers FHE
# ─────────────────────────────────────────────────────────────

def lv(ct, nome=""):
    l = ct.GetLevel()
    if nome:
        print(f"    lv({nome}) = {l}")
    return l

def encrypt(cc, kp, vec):
    pt = cc.MakeCKKSPackedPlaintext(vec.tolist(), 1, 0, None, SLOT)
    return cc.Encrypt(kp.publicKey, pt)

def encrypt_at(cc, kp, vec, level):
    pt = cc.MakeCKKSPackedPlaintext(vec.tolist(), 1, level, None, SLOT)
    return cc.Encrypt(kp.publicKey, pt)

def decrypt(cc, kp, ct, length=None):
    pt = cc.Decrypt(kp.secretKey, ct)
    if length: pt.SetLength(length)
    return np.array(pt.GetRealPackedValue())

def ct_rot(cc, ct, k):
    if k == 0: return ct
    return cc.EvalRotate(ct, int(k))

def ct_add(cc, a, b):
    return cc.EvalAdd(a, b)

def ct_mul_plain(cc, ct, mask_vec, level=None):
    if level is None: level = ct.GetLevel()
    pt = cc.MakeCKKSPackedPlaintext(mask_vec.tolist(), 1, level, None, SLOT)
    return cc.EvalMult(ct, pt)

def ct_rot_pow2(cc, ct, k):
    if k == 0: return ct
    sign = -1 if k < 0 else 1
    abs_k = abs(k); result = ct; bit = 1
    while bit <= abs_k:
        if abs_k & bit: result = cc.EvalRotate(result, sign * bit)
        bit <<= 1
    return result

def bootstrap(cc, ct, nome=""):
    prima = ct.GetLevel()
    ct    = cc.EvalBootstrap(ct)
    dopo  = ct.GetLevel()
    print(f"    [bootstrap{' '+nome if nome else ''}] {prima} → {dopo}")
    return ct


# ─────────────────────────────────────────────────────────────
# Operazioni FHE
# ─────────────────────────────────────────────────────────────

def fhe_flat_e_repeat_righe_bsgs(cc, X_ct, n_rep, N, M, slot):
    baby  = max(1, int(math.isqrt(n_rep)))
    giant = math.ceil(n_rep / baby)
    result = None
    for i in range(N):
        maschera_src = np.zeros(slot); maschera_src[i*M : i*M+M] = 1.0
        riga_i = ct_mul_plain(cc, X_ct, maschera_src)
        shift0  = -(i * M * (n_rep - 1))
        base_ct = ct_rot(cc, riga_i, shift0) if shift0 != 0 else riga_i
        precomp = [ct_rot(cc, base_ct, -(b * M)) if b > 0 else base_ct for b in range(baby)]
        for g in range(giant):
            chunk = None
            for b in range(baby):
                k = g * baby + b
                if k >= n_rep: continue
                chunk = precomp[b] if chunk is None else ct_add(cc, chunk, precomp[b])
            if chunk is None: continue
            shifted = ct_rot(cc, chunk, -(g * baby * M)) if g > 0 else chunk
            result  = shifted if result is None else ct_add(cc, result, shifted)
    return result

def fhe_repeat_flat_colmajor_bsgs(cc, W_ct, n_rep, N_mat, M_mat, slot):
    L2 = N_mat * M_mat
    col_major = fhe_trasposta(cc, W_ct, N_mat, M_mat, slot)

    baby  = max(1, int(math.isqrt(n_rep)))
    giant = math.ceil(n_rep / baby)
    precomp = [ct_rot(cc, col_major, -(b * L2)) if b > 0 else col_major for b in range(baby)]
    result = None
    for g in range(giant):
        chunk = None
        for b in range(baby):
            k = g * baby + b
            if k >= n_rep: continue
            chunk = precomp[b] if chunk is None else ct_add(cc, chunk, precomp[b])
        if chunk is None: continue
        shifted = ct_rot(cc, chunk, -(g * baby * L2)) if g > 0 else chunk
        result  = shifted if result is None else ct_add(cc, result, shifted)
    return result

def fhe_sum_consecutive_bsgs(cc, ct, M, slot):
    if M == 1:
        return ct
    baby  = max(1, int(math.isqrt(M)))
    giant = math.ceil(M / baby)
    precomp = [ct_rot(cc, ct, b) if b > 0 else ct for b in range(baby)]
    result = None
    for g in range(giant):
        chunk = None
        for b in range(baby):
            k = g * baby + b
            if k >= M: continue
            chunk = precomp[b] if chunk is None else ct_add(cc, chunk, precomp[b])
        if chunk is None: continue
        shifted = ct_rot(cc, chunk, g * baby) if g > 0 else chunk
        result  = shifted if result is None else ct_add(cc, result, shifted)
    return result

def fhe_inner_product_batch_bsgs(cc, v1_ct, v2_ct, M, N, L, slot):
    NL      = N * L
    prod_ct = cc.EvalMult(v1_ct, v2_ct)
    acc_ct  = fhe_sum_consecutive_bsgs(cc, prod_ct, M, slot)
    if M == 1: return acc_ct

    step_size = M - 1
    baby  = max(1, int(math.isqrt(NL))); giant = math.ceil(NL / baby)
    precomp = [cc.EvalRotate(acc_ct, b * step_size) if b > 0 else acc_ct for b in range(baby)]
    result_ct = None
    for g in range(giant):
        chunk_ct = None
        for b in range(baby):
            k = g * baby + b
            if k >= NL: continue
            pos = g * baby * M + b
            if pos >= NL * M: continue
            mask = np.zeros(slot); mask[pos % slot] = 1.0
            term     = ct_mul_plain(cc, precomp[b], mask)
            chunk_ct = term if chunk_ct is None else cc.EvalAdd(chunk_ct, term)
        if chunk_ct is None: continue
        rot       = g * baby * step_size
        shifted   = cc.EvalRotate(chunk_ct, rot) if rot > 0 else chunk_ct
        result_ct = shifted if result_ct is None else cc.EvalAdd(result_ct, shifted)
    return result_ct


def fhe_replica_bias_bsgs(cc, b_ct, N, M_bias, slot):
    mask_b = np.zeros(slot); mask_b[:M_bias] = 1.0
    b_solo = ct_mul_plain(cc, b_ct, mask_b)

    baby  = max(1, int(math.isqrt(N)))
    giant = math.ceil(N / baby)

    precomp = [b_solo]
    for b in range(1, baby):
        precomp.append(ct_rot(cc, b_solo, -(b * M_bias)))

    result = None
    for g in range(giant):
        chunk = None
        for b in range(baby):
            k = g * baby + b
            if k >= N:
                break
            chunk = precomp[b] if chunk is None else cc.EvalAdd(chunk, precomp[b])
        if g > 0:
            chunk = ct_rot(cc, chunk, -(g * baby * M_bias))
        result = chunk if result is None else cc.EvalAdd(result, chunk)

    return result

def fhe_aggiungi_bias_ct(cc, z_ct, b_rep_ct):
    return cc.EvalAdd(z_ct, b_rep_ct)

def fhe_trasposta(cc, W_ct, N, M, slot):
    rotations = [ct_rot(cc, W_ct, +i * (M - 1)) for i in range(N)]
    righe_trasposte = None
    for j in range(M):
        acc = None
        for i in range(N):
            maschera = np.zeros(slot); maschera[i + j] = 1.0
            term = ct_mul_plain(cc, rotations[i], maschera)
            acc  = term if acc is None else ct_add(cc, acc, term)
        shifted = ct_rot(cc, acc, -j * (N - 1))
        righe_trasposte = shifted if righe_trasposte is None else ct_add(cc, righe_trasposte, shifted)
    return righe_trasposte

def fhe_somma_batch_bsgs(cc, M_ct, N_r, M_r, slot):
    baby  = max(1, int(math.isqrt(N_r)))
    giant = math.ceil(N_r / baby)
    precomp = [ct_rot(cc, M_ct, b * M_r) if b > 0 else M_ct for b in range(baby)]
    acc = None
    for g in range(giant):
        chunk = None
        for b in range(baby):
            i = g * baby + b
            if i >= N_r: continue
            chunk = precomp[b] if chunk is None else ct_add(cc, chunk, precomp[b])
        if chunk is None: continue
        shifted = ct_rot(cc, chunk, g * baby * M_r) if g > 0 else chunk
        acc     = shifted if acc is None else ct_add(cc, acc, shifted)
    mask = np.zeros(slot); mask[:M_r] = 1.0
    pt = cc.MakeCKKSPackedPlaintext(mask.tolist(), 1, acc.GetLevel(), None, SLOT)
    return cc.EvalMult(acc, pt)


# ─────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────

def forward(cc, kp, X_ripetuto_ct, w1_colmajor_ct, b1_rep_ct, w2_colmajor_ct, b2_rep_ct):
    z1_ct    = fhe_inner_product_batch_bsgs(cc, X_ripetuto_ct, w1_colmajor_ct, M, N, L, SLOT)

    z1b_ct = fhe_aggiungi_bias_ct(cc, z1_ct, b1_rep_ct)

    z1b_ct = force_slots(cc, z1b_ct)
    z1b_ct = cc.EvalBootstrap(z1b_ct)

    a1_ct  = cc.EvalChebyshevFunction(gelu, z1b_ct, -6.0, 6.0, 11)
    a1_ct  = force_slots(cc, a1_ct)
    a1_ct  = cc.EvalBootstrap(a1_ct)
    z1b_ct = force_slots(cc, z1b_ct)
    z1b_ct = cc.EvalBootstrap(z1b_ct)

    new_v1_ct = fhe_flat_e_repeat_righe_bsgs(cc, a1_ct, C, N, L, SLOT)

    z2_ct     = fhe_inner_product_batch_bsgs(cc, new_v1_ct, w2_colmajor_ct, L, N, C, SLOT)

    z2b_ct = fhe_aggiungi_bias_ct(cc, z2_ct, b2_rep_ct)

    z2b_ct = force_slots(cc, z2b_ct)
    z2b_ct = cc.EvalBootstrap(z2b_ct)
    return z2b_ct, a1_ct, z1b_ct

# ─────────────────────────────────────────────────────────────
# Backward pass
# ─────────────────────────────────────────────────────────────

def backward(cc, kp, z2b_ct, a1_ct, z1b_ct, v2_d1_ct, X_flat, y_ct, POST_BOOT, cache):
    dz2_ct   = cc.EvalSub(z2b_ct, y_ct)

    db2_ct   = fhe_somma_batch_bsgs(cc, dz2_ct, N, C, SLOT)

    XT2_ct   = fhe_trasposta(cc, a1_ct, N, L, SLOT)

    v1_dW2   = fhe_flat_e_repeat_righe_bsgs(cc, XT2_ct, C, L, N, SLOT)

    v2_dW2   = fhe_repeat_flat_colmajor_bsgs(cc, dz2_ct, L, N, C, SLOT)

    dW2_ct   = fhe_inner_product_batch_bsgs(cc, v1_dW2, v2_dW2, N, L, C, SLOT)

    v1_d1    = fhe_flat_e_repeat_righe_bsgs(cc, dz2_ct, L, N, C, SLOT)

    der_out1_ct = fhe_inner_product_batch_bsgs(cc, v1_d1, v2_d1_ct, C, N, L, SLOT)

    dgelu_ct = cc.EvalChebyshevFunction(gelu_prime, z1b_ct, -6.0, 6.0, 11)

    der_out1_ct = force_slots(cc, der_out1_ct)
    der_out1_ct = cc.EvalBootstrap(der_out1_ct)

    dgelu_ct    = force_slots(cc, dgelu_ct)
    dgelu_ct    = cc.EvalBootstrap(dgelu_ct)

    dz1_ct   = cc.EvalMult(der_out1_ct, dgelu_ct)

    db1_ct   = fhe_somma_batch_bsgs(cc, dz1_ct, N, L, SLOT)

    v2_dW1 = fhe_repeat_flat_colmajor_bsgs(cc, dz1_ct, M, N, L, SLOT)

    livello_corrente = v2_dW1.GetLevel()
    if livello_corrente not in cache:
        XT1_flat = np.zeros(SLOT)
        XT1_flat[:M*N] = X_flat[:N*M].reshape(N, M).T.flatten()
        v1_dW1_plain = plain_flat_e_repeat_righe(XT1_flat, L, M, N, SLOT)
        cache[livello_corrente] = encrypt_at(cc, kp, v1_dW1_plain, livello_corrente)

    v1_dW1_ct = cache[livello_corrente]
    dW1_ct = fhe_inner_product_batch_bsgs(cc, v1_dW1_ct, v2_dW1, N, M, L, SLOT)
    return dW1_ct, db1_ct, dW2_ct, db2_ct

# ─────────────────────────────────────────────────────────────
# Accuracy in chiaro — MULTICLASSE (argmax su 10 uscite)
# ─────────────────────────────────────────────────────────────

def decripta_pesi(cc, kp, W1_ct, b1_ct, W2_ct, b2_ct):
    W1_dec = decrypt(cc, kp, W1_ct, M*L)[:M*L].reshape(M, L)
    b1_dec = decrypt(cc, kp, b1_ct, L)[:L]
    W2_dec = decrypt(cc, kp, W2_ct, L*C)[:L*C].reshape(L, C)
    b2_dec = decrypt(cc, kp, b2_ct, C)[:C]
    return W1_dec, b1_dec, W2_dec, b2_dec

def eval_accuracy(W1_dec, b1_dec, W2_dec, b2_dec, X_tr, y_tr, X_te, y_te):
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

    def predict(X):
        Z1 = X @ W1_dec + b1_dec
        A1 = np.vectorize(gelu)(Z1)
        Z2 = A1 @ W2_dec + b2_dec
        return np.argmax(Z2, axis=1)

    y_pred_tr = predict(X_tr)
    y_pred_te = predict(X_te)

    acc_tr  = accuracy_score(y_tr, y_pred_tr)
    acc_te  = accuracy_score(y_te, y_pred_te)
    f1_tr   = f1_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    f1_te   = f1_score(y_te, y_pred_te, average='macro', zero_division=0)
    prec_tr = precision_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    prec_te = precision_score(y_te, y_pred_te, average='macro', zero_division=0)
    rec_tr  = recall_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    rec_te  = recall_score(y_te, y_pred_te, average='macro', zero_division=0)
    cm_te   = confusion_matrix(y_te, y_pred_te)

    return acc_tr, acc_te, f1_tr, f1_te, prec_tr, prec_te, rec_tr, rec_te, cm_te

def calcola_mse(W1_dec, b1_dec, W2_dec, b2_dec, X, y_onehot):
    Z1 = X @ W1_dec + b1_dec
    A1 = np.vectorize(gelu)(Z1)
    Z2 = A1 @ W2_dec + b2_dec
    return np.mean((Z2 - y_onehot)**2)

def calcola_loss_training(W1_dec, b1_dec, W2_dec, b2_dec, X, y_onehot):
    """Loss esattamente come ottimizzata in backward: somma su C, media su N."""
    Z1 = X @ W1_dec + b1_dec
    A1 = np.vectorize(gelu)(Z1)
    Z2 = A1 @ W2_dec + b2_dec
    return np.mean(np.sum((Z2 - y_onehot)**2, axis=1))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time, psutil, os

    digits_full = load_digits()
    X_all_full = StandardScaler().fit_transform(digits_full.data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all_full, digits_full.target, test_size=0.2, random_state=42)

    y_onehot_all = np.eye(C)[digits_full.target].astype(float)
    y_onehot_train, _, _, _ = train_test_split(
        y_onehot_all, digits_full.target, test_size=0.2, random_state=42)
    X_raw_train, _, _, _ = train_test_split(
        X_all_full, digits_full.target, test_size=0.2, random_state=42)


    print(SEP)
    print("SETUP")
    print(SEP)
    np.random.seed(42)
    W1 = np.random.randn(M, L) * np.sqrt(1.0 / M)
    W2 = np.random.randn(L, C) * np.sqrt(1.0 / L)
    b1 = np.zeros(L)
    b2 = np.zeros(C)

    cc, level_budget, bootstrap_depth, depth = build_context(SECURE)
    SLOT = cc.GetRingDimension() // 2

    N_MAX = min(SLOT // (M * L), X_raw_train.shape[0])
    N_MINI_BATCHES = math.ceil(X_raw_train.shape[0] / N_MAX)
    N = X_raw_train.shape[0] // N_MINI_BATCHES

    print(f"  Bootstrap depth:  {bootstrap_depth}")
    print(f"  Depth totale:     {depth}  (limite={depth-2})")
    print(f"  Ring dim:         {cc.GetRingDimension()}")
    print(f"  SLOT:             {SLOT}")
    print(f"  N (batch size):   {N}  →  N_MINI_BATCHES={N_MINI_BATCHES}  "
          f"(train={X_raw_train.shape[0]}, scartati per epoca="
          f"{X_raw_train.shape[0] - N_MINI_BATCHES*N})")
    kp = gen_keys(cc, level_budget)

    print(f"\n{SEP}")
    print("CALIBRAZIONE POST_BOOT")
    print(SEP)
    POST_BOOT = calibra_post_boot(cc, kp, depth)
    print(f"  POST_BOOT={POST_BOOT}  livelli utili per ciclo={(depth-2)-POST_BOOT}")

    X_batches = []
    y_batches = []
    for i in range(N_MINI_BATCHES):
        xb = np.zeros(SLOT)
        batch = X_raw_train[i*N:(i+1)*N]
        n_real = batch.shape[0]
        xb[:n_real*M] = np.clip(batch, -5, 5).flatten()
        X_batches.append(xb)
        y_batches.append(y_onehot_train[i*N:(i+1)*N])

    X_ripetuto_ct_batches = []
    y_ct_batches = []
    for i in range(N_MINI_BATCHES):
        n_real = y_batches[i].shape[0]
        y_flat = np.zeros(SLOT)
        y_flat[:n_real*C] = y_batches[i].flatten()
        y_ct_batches.append(encrypt(cc, kp, y_flat))
        x_ripetuto_plain = plain_flat_e_repeat_righe(X_batches[i], L, N, M, SLOT)
        X_ripetuto_ct_batches.append(encrypt(cc, kp, x_ripetuto_plain))
    cache_x_trasposta = [{} for _ in range(N_MINI_BATCHES)]

    W1_flat = np.zeros(SLOT); W1_flat[:M*L] = W1.flatten()
    W2_flat = np.zeros(SLOT); W2_flat[:L*C] = W2.flatten()
    b1_flat = np.zeros(SLOT); b1_flat[:L]   = b1
    b2_flat = np.zeros(SLOT); b2_flat[:C]   = b2

    W1_ct = encrypt_at(cc, kp, W1_flat, POST_BOOT)
    W2_ct = encrypt_at(cc, kp, W2_flat, POST_BOOT)
    b1_ct = encrypt_at(cc, kp, b1_flat, POST_BOOT)
    b2_ct = encrypt_at(cc, kp, b2_flat, POST_BOOT)

    print(f"\n  Pesi cifrati: W1={lv(W1_ct)}  W2={lv(W2_ct)}  ← devono essere {POST_BOOT}")

    LOG_PATH = "digits_ckks_results_sicuro.csv" if SECURE else "digits_ckks_results_non_sicuro.csv"
    with open(LOG_PATH, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow([
            "epoca", "acc_train", "acc_test",
            "f1_train", "f1_test",
            "prec_train", "prec_test",
            "rec_train", "rec_test",
            "mse_train", "mse_test",
            "loss_train", "loss_test",
            "norm_W1", "norm_W2", "norm_b1", "norm_b2",
            "max_W1", "max_W2",
            "livello_pre_boot_W1",
            "cm_test",
            "tempo_s", "tempo_cumulativo_s"
        ])

    lr     = 0.1
    EPOCHE = 120

    print(f"\n{SEP}")
    print(f"TRAINING — {EPOCHE} epoche")
    print(SEP)

    t_totale = time.time()

    for epoca in range(EPOCHE):
            t_ep = time.time()

            w1_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, W1_ct, N, M, L, SLOT)
            w2_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, W2_ct, N, L, C, SLOT)
            b1_rep_ct = fhe_replica_bias_bsgs(cc, b1_ct, N, L, SLOT)
            b2_rep_ct = fhe_replica_bias_bsgs(cc, b2_ct, N, C, SLOT)
            w2T_ct = fhe_trasposta(cc, W2_ct, L, C, SLOT)
            v2_d1_ct = fhe_repeat_flat_colmajor_bsgs(cc, w2T_ct, N, C, L, SLOT)
            del w2T_ct; gc.collect(); libc.malloc_trim(0)

            dW1_acc = None; dW2_acc = None
            db1_acc = None; db2_acc = None

            for i in range(N_MINI_BATCHES):
                z2b_i, a1_i, z1b_i = forward(cc, kp, X_ripetuto_ct_batches[i], w1_colmajor_ct, b1_rep_ct, w2_colmajor_ct, b2_rep_ct)
                dW1_i, db1_i, dW2_i, db2_i = backward(
                    cc, kp, z2b_i, a1_i, z1b_i, v2_d1_ct, X_batches[i], y_ct_batches[i], POST_BOOT, cache_x_trasposta[i])

                print(f"    batch {i+1}/{N_MINI_BATCHES} completato")

                if dW1_acc is None:
                    dW1_acc = dW1_i; dW2_acc = dW2_i
                    db1_acc = db1_i; db2_acc = db2_i
                else:
                    dW1_acc = cc.EvalAdd(dW1_acc, dW1_i)
                    dW2_acc = cc.EvalAdd(dW2_acc, dW2_i)
                    db1_acc = cc.EvalAdd(db1_acc, db1_i)
                    db2_acc = cc.EvalAdd(db2_acc, db2_i)

            scale = cc.MakeCKKSPackedPlaintext([1.0/N_MINI_BATCHES]*SLOT, 1, dW1_acc.GetLevel(), None, SLOT)
            dW1_ct = cc.EvalMult(dW1_acc, scale)
            dW2_ct = cc.EvalMult(dW2_acc, scale)
            db1_ct = cc.EvalMult(db1_acc, scale)
            db2_ct = cc.EvalMult(db2_acc, scale)

            dW1_ct = force_slots(cc, dW1_ct); dW1_ct = cc.EvalBootstrap(dW1_ct)
            dW2_ct = force_slots(cc, dW2_ct); dW2_ct = cc.EvalBootstrap(dW2_ct)
            db1_ct = force_slots(cc, db1_ct); db1_ct = cc.EvalBootstrap(db1_ct)
            db2_ct = force_slots(cc, db2_ct); db2_ct = cc.EvalBootstrap(db2_ct)

            W1_ct = cc.EvalSub(W1_ct, cc.EvalMult(dW1_ct,
                        cc.MakeCKKSPackedPlaintext([lr]*SLOT, 1, dW1_ct.GetLevel(), None, SLOT)))
            livello_pre_boot_W1 = W1_ct.GetLevel()

            W2_ct = cc.EvalSub(W2_ct, cc.EvalMult(dW2_ct,
                        cc.MakeCKKSPackedPlaintext([lr]*SLOT, 1, dW2_ct.GetLevel(), None, SLOT)))
            b1_ct = cc.EvalSub(b1_ct, cc.EvalMult(db1_ct,
                        cc.MakeCKKSPackedPlaintext([lr]*SLOT, 1, db1_ct.GetLevel(), None, SLOT)))
            b2_ct = cc.EvalSub(b2_ct, cc.EvalMult(db2_ct,
                        cc.MakeCKKSPackedPlaintext([lr]*SLOT, 1, db2_ct.GetLevel(), None, SLOT)))

            W1_ct = force_slots(cc, W1_ct); W1_ct = cc.EvalBootstrap(W1_ct)
            W2_ct = force_slots(cc, W2_ct); W2_ct = cc.EvalBootstrap(W2_ct)
            b1_ct = force_slots(cc, b1_ct); b1_ct = cc.EvalBootstrap(b1_ct)
            b2_ct = force_slots(cc, b2_ct); b2_ct = cc.EvalBootstrap(b2_ct)

            W1_dec, b1_dec, W2_dec, b2_dec = decripta_pesi(cc, kp, W1_ct, b1_ct, W2_ct, b2_ct)
            print(f"    [norme] W1={np.linalg.norm(W1_dec):.3f}  W2={np.linalg.norm(W2_dec):.3f}  "
                  f"b1={np.linalg.norm(b1_dec):.3f}  b2={np.linalg.norm(b2_dec):.3f}  "
                  f"W1_max={np.max(np.abs(W1_dec)):.3f}  W2_max={np.max(np.abs(W2_dec)):.3f}")
            acc_tr, acc_te, f1_tr, f1_te, prec_tr, prec_te, rec_tr, rec_te, cm_te = eval_accuracy(
                W1_dec, b1_dec, W2_dec, b2_dec, X_tr, y_tr, X_te, y_te)
            t_ep = time.time() - t_ep
            proc = psutil.Process(os.getpid())

            mse_tr = calcola_mse(W1_dec, b1_dec, W2_dec, b2_dec, X_tr, np.eye(C)[y_tr])
            mse_te = calcola_mse(W1_dec, b1_dec, W2_dec, b2_dec, X_te, np.eye(C)[y_te])
            loss_tr = calcola_loss_training(W1_dec, b1_dec, W2_dec, b2_dec, X_tr, np.eye(C)[y_tr])
            loss_te = calcola_loss_training(W1_dec, b1_dec, W2_dec, b2_dec, X_te, np.eye(C)[y_te])
            cm_flat = cm_te.flatten().tolist()
            t_cumulativo = time.time() - t_totale

            with open(LOG_PATH, "a", newline="") as f_log:
                writer = csv.writer(f_log)
                writer.writerow([
                    epoca + 1, acc_tr, acc_te,
                    f1_tr, f1_te,
                    prec_tr, prec_te,
                    rec_tr, rec_te,
                    mse_tr, mse_te,
                    loss_tr, loss_te,                                  # ← aggiunta
                    np.linalg.norm(W1_dec), np.linalg.norm(W2_dec),
                    np.linalg.norm(b1_dec), np.linalg.norm(b2_dec),
                    np.max(np.abs(W1_dec)), np.max(np.abs(W2_dec)),
                    livello_pre_boot_W1,
                    ";".join(map(str, cm_flat)),
                    round(t_ep, 1), round(t_cumulativo, 1)
                ])

            print(f"  Epoca {epoca+1:3d} | "
                f"acc={acc_tr:.4f}/{acc_te:.4f} | "
                f"f1(macro)={f1_tr:.4f}/{f1_te:.4f} | "
                f"prec(macro)={prec_tr:.4f}/{prec_te:.4f} | "
                f"rec(macro)={rec_tr:.4f}/{rec_te:.4f} | "
                f"t={t_ep:.0f}s | RAM={proc.memory_info().rss/1e9:.1f}GB")
            if (epoca + 1) % 10 == 0:
                print(f"    CM test (10x10):\n{cm_te}")

    print(f"\n{SEP}")
    print(f"FINE TRAINING")
    print(f"  Tempo totale: {(time.time()-t_totale)/60:.1f} min")
    W1_dec, b1_dec, W2_dec, b2_dec = decripta_pesi(cc, kp, W1_ct, b1_ct, W2_ct, b2_ct)
    acc_tr_fin, acc_te_fin, f1_tr_fin, f1_te_fin, prec_tr_fin, prec_te_fin, rec_tr_fin, rec_te_fin, cm_te_fin = eval_accuracy(
        W1_dec, b1_dec, W2_dec, b2_dec, X_tr, y_tr, X_te, y_te)
    print(f"  Accuracy  — train={acc_tr_fin:.4f}  test={acc_te_fin:.4f}")
    print(f"  F1 macro  — train={f1_tr_fin:.4f}  test={f1_te_fin:.4f}")
    print(f"  Precision — train={prec_tr_fin:.4f}  test={prec_te_fin:.4f}")
    print(f"  Recall    — train={rec_tr_fin:.4f}  test={rec_te_fin:.4f}")
    print(f"  Confusion matrix (test, 10x10):")
    print(cm_te_fin)