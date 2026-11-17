import math
import csv
import os
import numpy as np
import openfhe as fhe
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from scipy.special import ndtr

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

M, L, C = 13, 8, 3

RING_DIM = 1 << 14
SLOT = RING_DIM // 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "wine_ckks_results_sicuro_prova_2.csv" if SECURE else "wine_ckks_results_non_sicuro_run3.csv"
)
CSV_HEADER = [
    "epoca", "acc_train", "acc_test", "f1_train", "f1_test",
    "prec_train", "prec_test", "rec_train", "rec_test",
    "mse_train", "mse_test",
    "norm_W1", "norm_W2", "norm_b1", "norm_b2",
    "max_W1", "max_W2",
    "tp_test", "fp_test", "fn_test", "tn_test",
    "tempo_s", "tempo_cumulativo_s",
]


def csv_init():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        f.flush()
        os.fsync(f.fileno())


def csv_append_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def plain_flat_e_repeat_righe(X_flat, n_rep, N, M, slot):
    risultato = np.zeros(slot)
    for indice_riga in range(N):
        for indice_ripetizione in range(n_rep):
            indice_sorgente = indice_riga * M
            indice_destinazione = indice_riga * n_rep * M + indice_ripetizione * M
            risultato[indice_destinazione:indice_destinazione + M] = X_flat[indice_sorgente:indice_sorgente + M]
    return risultato


def gelu(x):
    return x * ndtr(x)


def gelu_prime(x):
    return ndtr(x) + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def build_context(secure):
    params = fhe.CCParamsCKKSRNS()
    secret_key_dist = fhe.SecretKeyDist.SPARSE_TERNARY
    params.SetSecretKeyDist(secret_key_dist)

    if secure:
        params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    else:
        params.SetSecurityLevel(fhe.SecurityLevel.HEStd_NotSet)
        params.SetRingDim(RING_DIM)

    dcrt_bits = 55
    first_mod = 56
    params.SetScalingModSize(dcrt_bits)
    params.SetScalingTechnique(fhe.ScalingTechnique.FLEXIBLEAUTO)
    params.SetFirstModSize(first_mod)
    level_budget = [3, 3]
    bootstrap_depth = fhe.FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)
    levels_before = 6
    depth = levels_before + bootstrap_depth
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
    pt = cc.MakeCKKSPackedPlaintext(vec, 1, 0, None, SLOT)
    ct = cc.Encrypt(kp.publicKey, pt)
    for _ in range(depth - 2):
        pt2 = cc.MakeCKKSPackedPlaintext([1.0] * SLOT, 1, ct.GetLevel(), None, SLOT)
        ct = cc.EvalMult(ct, pt2)
    print(f"  [calibrazione] livello prima del bootstrap: {ct.GetLevel()}")
    ct = cc.EvalBootstrap(ct)
    post = ct.GetLevel()
    print(f"  [calibrazione] livello dopo bootstrap:      {post}")
    livelli_utili = (depth - 2) - post
    print(f"  [calibrazione] livelli utili per ciclo:     {livelli_utili}")
    return post


def compute_needed_rotations(N, M, L, C, slot):
    rotazioni_necessarie = set()

    def aggiungi_rotazioni_bsgs(numero_rotazioni, passo):
        num_baby = max(1, int(math.isqrt(numero_rotazioni)))
        num_giant = math.ceil(numero_rotazioni / num_baby)
        for indice_baby in range(1, num_baby):
            rotazioni_necessarie.add(indice_baby * passo)
            rotazioni_necessarie.add(-(indice_baby * passo))
        for indice_giant in range(1, num_giant):
            rotazioni_necessarie.add(indice_giant * num_baby * passo)
            rotazioni_necessarie.add(-(indice_giant * num_baby * passo))

    def aggiungi_rotazioni_trasposizione(righe, colonne):
        for indice_riga in range(1, righe):
            rotazioni_necessarie.add(indice_riga * (colonne - 1))
        for indice_colonna in range(1, colonne):
            rotazioni_necessarie.add(-(indice_colonna * (righe - 1)))

    def aggiungi_rotazioni_flat_repeat(n_rep, righe, colonne):
        for indice_riga in range(1, righe):
            rotazioni_necessarie.add(-(indice_riga * colonne * (n_rep - 1)))
        aggiungi_rotazioni_bsgs(n_rep, colonne)

    aggiungi_rotazioni_flat_repeat(C, N, L)
    aggiungi_rotazioni_flat_repeat(C, L, N)
    aggiungi_rotazioni_flat_repeat(L, N, C)

    for righe, colonne in [(M, L), (L, C), (N, C), (C, L), (N, L)]:
        aggiungi_rotazioni_trasposizione(righe, colonne)
    for n_rep, dim_blocco in [(N, M * L), (N, L * C), (L, N * C), (N, C * L), (M, N * L)]:
        aggiungi_rotazioni_bsgs(n_rep, dim_blocco)

    for dim_contrazione in {M, L, N, C}:
        aggiungi_rotazioni_bsgs(dim_contrazione, 1)
    for dim_contrazione, dim_a, dim_b in [(M, N, L), (L, N, C), (N, L, C), (C, N, L), (N, M, L)]:
        if dim_contrazione == 1:
            continue
        aggiungi_rotazioni_bsgs(dim_a * dim_b, dim_contrazione - 1)

    for num_campioni, dim_blocco in [(N, C), (N, L)]:
        aggiungi_rotazioni_bsgs(num_campioni, dim_blocco)

    for dim_bias in [L, C]:
        aggiungi_rotazioni_bsgs(N, dim_bias)

    rotazioni_necessarie.discard(0)
    return sorted(rotazioni_necessarie)


def genera_chiavi(cc, level_budget):
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    rotazioni = compute_needed_rotations(N, M, L, C, SLOT)
    print(f"  Chiavi di rotazione: {len(rotazioni)}")
    cc.EvalRotateKeyGen(kp.secretKey, rotazioni)
    cc.EvalBootstrapSetup(level_budget, [0, 0], SLOT)
    cc.EvalBootstrapKeyGen(kp.secretKey, SLOT)
    return kp


def force_slots(cc, ct):
    pt = cc.MakeCKKSPackedPlaintext([0.0] * SLOT, 1, ct.GetLevel(), None, SLOT)
    return cc.EvalAdd(ct, pt)


def livello(ct, nome=""):
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
    if length:
        pt.SetLength(length)
    return np.array(pt.GetRealPackedValue())


def ct_rot(cc, ct, k):
    if k == 0:
        return ct
    return cc.EvalRotate(ct, int(k))


def ct_add(cc, a, b):
    return cc.EvalAdd(a, b)


def ct_mul_plain(cc, ct, mask_vec, level=None):
    if level is None:
        level = ct.GetLevel()
    pt = cc.MakeCKKSPackedPlaintext(mask_vec.tolist(), 1, level, None, SLOT)
    return cc.EvalMult(ct, pt)


def ct_rot_pow2(cc, ct, k):
    if k == 0:
        return ct
    sign = -1 if k < 0 else 1
    abs_k = abs(k)
    result = ct
    bit = 1
    while bit <= abs_k:
        if abs_k & bit:
            result = cc.EvalRotate(result, sign * bit)
        bit <<= 1
    return result


def bootstrap(cc, ct, nome=""):
    prima = ct.GetLevel()
    ct = cc.EvalBootstrap(ct)
    dopo = ct.GetLevel()
    print(f"    [bootstrap{' '+nome if nome else ''}] {prima} → {dopo}")
    return ct


def fhe_flat_e_repeat_righe_bsgs(cc, X_ct, n_rep, N, M, slot):
    num_baby = max(1, int(math.isqrt(n_rep)))
    num_giant = math.ceil(n_rep / num_baby)
    risultato = None
    for indice_riga in range(N):
        maschera_src = np.zeros(slot)
        maschera_src[indice_riga*M : indice_riga*M+M] = 1.0
        riga_i = ct_mul_plain(cc, X_ct, maschera_src)
        spostamento_iniziale = -(indice_riga * M * (n_rep - 1))
        riga_allineata_ct = ct_rot(cc, riga_i, spostamento_iniziale) if spostamento_iniziale != 0 else riga_i
        rotazioni_baby = [
            ct_rot(cc, riga_allineata_ct, -(indice_baby * M)) if indice_baby > 0 else riga_allineata_ct
            for indice_baby in range(num_baby)
        ]
        for indice_giant in range(num_giant):
            somma_parziale = None
            for indice_baby in range(num_baby):
                indice_blocco = indice_giant * num_baby + indice_baby
                if indice_blocco >= n_rep:
                    continue
                somma_parziale = rotazioni_baby[indice_baby] if somma_parziale is None else ct_add(cc, somma_parziale, rotazioni_baby[indice_baby])
            if somma_parziale is None:
                continue
            contributo_giant = ct_rot(cc, somma_parziale, -(indice_giant * num_baby * M)) if indice_giant > 0 else somma_parziale
            risultato = contributo_giant if risultato is None else ct_add(cc, risultato, contributo_giant)
    return risultato


def fhe_trasposta(cc, W_ct, N, M, slot):
    rotazioni_righe = [ct_rot(cc, W_ct, +indice_riga * (M - 1)) for indice_riga in range(N)]
    risultato = None
    for indice_colonna in range(M):
        somma_colonna = None
        for indice_riga in range(N):
            maschera = np.zeros(slot)
            maschera[indice_riga + indice_colonna] = 1.0
            termine = ct_mul_plain(cc, rotazioni_righe[indice_riga], maschera)
            somma_colonna = termine if somma_colonna is None else ct_add(cc, somma_colonna, termine)
        colonna_shiftata = ct_rot(cc, somma_colonna, -indice_colonna * (N - 1))
        risultato = colonna_shiftata if risultato is None else ct_add(cc, risultato, colonna_shiftata)
    return risultato


def fhe_repeat_flat_colmajor_bsgs(cc, W_ct, n_rep, N_mat, M_mat, slot):
    dim_blocco_colmajor = N_mat * M_mat

    ct_colmajor = fhe_trasposta(cc, W_ct, N_mat, M_mat, slot)

    num_baby = max(1, int(math.isqrt(n_rep)))
    num_giant = math.ceil(n_rep / num_baby)
    rotazioni_baby = [ct_rot(cc, ct_colmajor, -(indice_baby * dim_blocco_colmajor)) if indice_baby > 0 else ct_colmajor for indice_baby in range(num_baby)]
    risultato = None
    for indice_giant in range(num_giant):
        somma_parziale = None
        for indice_baby in range(num_baby):
            indice_blocco = indice_giant * num_baby + indice_baby
            if indice_blocco >= n_rep:
                continue
            somma_parziale = rotazioni_baby[indice_baby] if somma_parziale is None else ct_add(cc, somma_parziale, rotazioni_baby[indice_baby])
        if somma_parziale is None:
            continue
        contributo_giant = ct_rot(cc, somma_parziale, -(indice_giant * num_baby * dim_blocco_colmajor)) if indice_giant > 0 else somma_parziale
        risultato = contributo_giant if risultato is None else ct_add(cc, risultato, contributo_giant)
    return risultato


def fhe_sum_consecutive_bsgs(cc, ct, dim_blocco, slot):
    if dim_blocco == 1:
        return ct
    num_baby = max(1, int(math.isqrt(dim_blocco)))
    num_giant = math.ceil(dim_blocco / num_baby)
    rotazioni_baby = [ct_rot(cc, ct, indice_baby) if indice_baby > 0 else ct for indice_baby in range(num_baby)]
    risultato = None
    for indice_giant in range(num_giant):
        somma_parziale = None
        for indice_baby in range(num_baby):
            indice_elemento = indice_giant * num_baby + indice_baby
            if indice_elemento >= dim_blocco:
                continue
            somma_parziale = rotazioni_baby[indice_baby] if somma_parziale is None else ct_add(cc, somma_parziale, rotazioni_baby[indice_baby])
        if somma_parziale is None:
            continue
        contributo_giant = ct_rot(cc, somma_parziale, indice_giant * num_baby) if indice_giant > 0 else somma_parziale
        risultato = contributo_giant if risultato is None else ct_add(cc, risultato, contributo_giant)
    return risultato


def fhe_inner_product_batch_bsgs(cc, ct_fattore_a, ct_fattore_b, dim_contrazione, dim_a, dim_b, slot):
    num_blocchi_output = dim_a * dim_b
    ct_prodotto = cc.EvalMult(ct_fattore_a, ct_fattore_b)
    ct_somma_blocchi = fhe_sum_consecutive_bsgs(cc, ct_prodotto, dim_contrazione, slot)
    if dim_contrazione == 1:
        return ct_somma_blocchi

    passo_compattazione = dim_contrazione - 1
    num_baby = max(1, int(math.isqrt(num_blocchi_output)))
    num_giant = math.ceil(num_blocchi_output / num_baby)
    rotazioni_baby = [cc.EvalRotate(ct_somma_blocchi, indice_baby * passo_compattazione) if indice_baby > 0 else ct_somma_blocchi for indice_baby in range(num_baby)]
    risultato_ct = None
    for indice_giant in range(num_giant):
        somma_parziale_ct = None
        for indice_baby in range(num_baby):
            indice_blocco = indice_giant * num_baby + indice_baby
            if indice_blocco >= num_blocchi_output:
                continue
            posizione = indice_giant * num_baby * dim_contrazione + indice_baby
            if posizione >= num_blocchi_output * dim_contrazione:
                continue
            maschera = np.zeros(slot)
            maschera[posizione % slot] = 1.0
            termine = ct_mul_plain(cc, rotazioni_baby[indice_baby], maschera)
            somma_parziale_ct = termine if somma_parziale_ct is None else cc.EvalAdd(somma_parziale_ct, termine)
        if somma_parziale_ct is None:
            continue
        rotazione = indice_giant * num_baby * passo_compattazione
        contributo_giant_ct = cc.EvalRotate(somma_parziale_ct, rotazione) if rotazione > 0 else somma_parziale_ct
        risultato_ct = contributo_giant_ct if risultato_ct is None else cc.EvalAdd(risultato_ct, contributo_giant_ct)
    return risultato_ct


def fhe_aggiungi_bias_ct(cc, z_ct, b_ct, N, dim_bias, slot):
    maschera_bias = np.zeros(slot)
    maschera_bias[:dim_bias] = 1.0
    bias_isolato_ct = ct_mul_plain(cc, b_ct, maschera_bias)

    num_baby = max(1, int(math.isqrt(N)))
    num_giant = math.ceil(N / num_baby)

    rotazioni_baby = [bias_isolato_ct]
    for indice_baby in range(1, num_baby):
        rotazioni_baby.append(ct_rot(cc, bias_isolato_ct, -(indice_baby * dim_bias)))

    risultato = None
    for indice_giant in range(num_giant):
        somma_parziale = None
        for indice_baby in range(num_baby):
            indice_campione = indice_giant * num_baby + indice_baby
            if indice_campione >= N:
                break
            somma_parziale = rotazioni_baby[indice_baby] if somma_parziale is None else cc.EvalAdd(somma_parziale, rotazioni_baby[indice_baby])

        if indice_giant > 0:
            somma_parziale = ct_rot(cc, somma_parziale, -(indice_giant * num_baby * dim_bias))

        risultato = somma_parziale if risultato is None else cc.EvalAdd(risultato, somma_parziale)

    return cc.EvalAdd(z_ct, risultato)


def fhe_somma_batch_bsgs(cc, ct_errore, num_campioni, dim_blocco, num_slot):
    num_baby = max(1, int(math.isqrt(num_campioni)))
    num_giant = math.ceil(num_campioni / num_baby)

    rotazioni_baby = [
        ct_rot(cc, ct_errore, indice_baby * dim_blocco) if indice_baby > 0 else ct_errore
        for indice_baby in range(num_baby)
    ]

    somma_totale = None
    for indice_giant in range(num_giant):
        somma_parziale = None
        for indice_baby in range(num_baby):
            indice_blocco = indice_giant * num_baby + indice_baby
            if indice_blocco >= num_campioni:
                continue
            somma_parziale = (
                rotazioni_baby[indice_baby]
                if somma_parziale is None
                else ct_add(cc, somma_parziale, rotazioni_baby[indice_baby])
            )

        if somma_parziale is None:
            continue

        contributo_giant = (
            ct_rot(cc, somma_parziale, indice_giant * num_baby * dim_blocco)
            if indice_giant > 0
            else somma_parziale
        )
        somma_totale = (
            contributo_giant if somma_totale is None else ct_add(cc, somma_totale, contributo_giant)
        )

    maschera = np.zeros(num_slot)
    maschera[:dim_blocco] = 1.0
    pt_maschera = cc.MakeCKKSPackedPlaintext(maschera.tolist(), 1, somma_totale.GetLevel(), None, SLOT)

    return cc.EvalMult(somma_totale, pt_maschera)


def forward(cc, kp, X_ripetuto_ct, W1_ct, b1_ct, W2_ct, b2_ct):
    w1_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, W1_ct, N, M, L, SLOT)
    z1_ct = fhe_inner_product_batch_bsgs(cc, X_ripetuto_ct, w1_colmajor_ct, M, N, L, SLOT)

    z1_con_bias_ct = fhe_aggiungi_bias_ct(cc, z1_ct, b1_ct, N, L, SLOT)

    z1_con_bias_ct = force_slots(cc, z1_con_bias_ct)
    z1_con_bias_ct = cc.EvalBootstrap(z1_con_bias_ct)

    a1_ct = cc.EvalChebyshevFunction(gelu, z1_con_bias_ct, -6.0, 6.0, 11)
    a1_ct = force_slots(cc, a1_ct)
    a1_ct = cc.EvalBootstrap(a1_ct)
    z1_con_bias_ct = force_slots(cc, z1_con_bias_ct)
    z1_con_bias_ct = cc.EvalBootstrap(z1_con_bias_ct)

    a1_ripetuto_ct = fhe_flat_e_repeat_righe_bsgs(cc, a1_ct, C, N, L, SLOT)
    w2_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, W2_ct, N, L, C, SLOT)
    z2_ct = fhe_inner_product_batch_bsgs(cc, a1_ripetuto_ct, w2_colmajor_ct, L, N, C, SLOT)

    z2_con_bias_ct = fhe_aggiungi_bias_ct(cc, z2_ct, b2_ct, N, C, SLOT)

    z2_con_bias_ct = force_slots(cc, z2_con_bias_ct)
    z2_con_bias_ct = cc.EvalBootstrap(z2_con_bias_ct)
    return z2_con_bias_ct, a1_ct, z1_con_bias_ct


def backward(cc, kp, z2_con_bias_ct, a1_ct, z1_con_bias_ct, W2_ct, X_flat, y_train_onehot, POST_BOOT, cache):
    y_onehot_flat = np.zeros(SLOT)
    y_onehot_flat[:N*C] = y_train_onehot.flatten()
    y_ct = encrypt(cc, kp, y_onehot_flat)
    dz2_ct = cc.EvalSub(z2_con_bias_ct, y_ct)
    scala_pt = cc.MakeCKKSPackedPlaintext([2.0/N]*SLOT, 1, dz2_ct.GetLevel(), None, SLOT)
    dz2_ct = cc.EvalMult(dz2_ct, scala_pt)

    db2_ct = fhe_somma_batch_bsgs(cc, dz2_ct, N, C, SLOT)
    a1_trasposta_ct = fhe_trasposta(cc, a1_ct, N, L, SLOT)
    a1_trasposta_ripetuta_ct = fhe_flat_e_repeat_righe_bsgs(cc, a1_trasposta_ct, C, L, N, SLOT)
    dz2_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, dz2_ct, L, N, C, SLOT)
    dW2_ct = fhe_inner_product_batch_bsgs(cc, a1_trasposta_ripetuta_ct, dz2_colmajor_ct, N, L, C, SLOT)

    w2_trasposta_ct = fhe_trasposta(cc, W2_ct, L, C, SLOT)
    dz2_ripetuto_ct = fhe_flat_e_repeat_righe_bsgs(cc, dz2_ct, L, N, C, SLOT)
    w2_trasposta_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, w2_trasposta_ct, N, C, L, SLOT)
    dA1_ct = fhe_inner_product_batch_bsgs(cc, dz2_ripetuto_ct, w2_trasposta_colmajor_ct, C, N, L, SLOT)

    gelu_prime_ct = cc.EvalChebyshevFunction(gelu_prime, z1_con_bias_ct, -6.0, 6.0, 11)

    dA1_ct = force_slots(cc, dA1_ct)
    dA1_ct = cc.EvalBootstrap(dA1_ct)
    gelu_prime_ct = force_slots(cc, gelu_prime_ct)
    gelu_prime_ct = cc.EvalBootstrap(gelu_prime_ct)
    dz1_ct = cc.EvalMult(dA1_ct, gelu_prime_ct)

    db1_ct = fhe_somma_batch_bsgs(cc, dz1_ct, N, L, SLOT)

    dz1_colmajor_ct = fhe_repeat_flat_colmajor_bsgs(cc, dz1_ct, M, N, L, SLOT)
    livello_corrente = dz1_colmajor_ct.GetLevel()

    if livello_corrente not in cache:
        x_trasposta_flat = np.zeros(SLOT)
        x_trasposta_flat[:M*N] = X_flat[:N*M].reshape(N, M).T.flatten()
        x_trasposta_ripetuta_plain = plain_flat_e_repeat_righe(x_trasposta_flat, L, M, N, SLOT)
        cache[livello_corrente] = encrypt_at(cc, kp, x_trasposta_ripetuta_plain, livello_corrente)

    x_trasposta_ripetuta_ct = cache[livello_corrente]
    dW1_ct = fhe_inner_product_batch_bsgs(cc, x_trasposta_ripetuta_ct, dz1_colmajor_ct, N, M, L, SLOT)

    return dW1_ct, db1_ct, dW2_ct, db2_ct


def eval_accuracy(cc, kp, W1_ct, b1_ct, W2_ct, b2_ct,
                   X_tr, y_tr, X_te, y_te, y_tr_onehot, y_te_onehot):
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    W1_dec = decrypt(cc, kp, W1_ct, M*L)[:M*L].reshape(M, L)
    b1_dec = decrypt(cc, kp, b1_ct, L)[:L]
    W2_dec = decrypt(cc, kp, W2_ct, L*C)[:L*C].reshape(L, C)
    b2_dec = decrypt(cc, kp, b2_ct, C)[:C]

    def logits(X):
        Z1 = X @ W1_dec + b1_dec
        A1 = np.vectorize(gelu)(Z1)
        return A1 @ W2_dec + b2_dec

    Z2_tr = logits(X_tr)
    Z2_te = logits(X_te)
    y_pred_tr = np.argmax(Z2_tr, axis=1)
    y_pred_te = np.argmax(Z2_te, axis=1)

    acc_tr = np.mean(y_pred_tr == y_tr)
    acc_te = np.mean(y_pred_te == y_te)
    f1_tr = f1_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    f1_te = f1_score(y_te, y_pred_te, average='macro', zero_division=0)
    prec_tr = precision_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    prec_te = precision_score(y_te, y_pred_te, average='macro', zero_division=0)
    rec_tr = recall_score(y_tr, y_pred_tr, average='macro', zero_division=0)
    rec_te = recall_score(y_te, y_pred_te, average='macro', zero_division=0)
    cm_te = confusion_matrix(y_te, y_pred_te)

    mse_tr = np.mean((Z2_tr - y_tr_onehot) ** 2)
    mse_te = np.mean((Z2_te - y_te_onehot) ** 2)

    norm_W1 = np.linalg.norm(W1_dec)
    norm_W2 = np.linalg.norm(W2_dec)
    norm_b1 = np.linalg.norm(b1_dec)
    norm_b2 = np.linalg.norm(b2_dec)
    max_W1 = np.max(np.abs(W1_dec))
    max_W2 = np.max(np.abs(W2_dec))

    return (acc_tr, acc_te, f1_tr, f1_te, prec_tr, prec_te, rec_tr, rec_te, cm_te,
            mse_tr, mse_te, norm_W1, norm_W2, norm_b1, norm_b2, max_W1, max_W2)


def cm_to_tp_fp_fn_tn(cm):
    numero_classi = cm.shape[0]
    totale = cm.sum()
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    for indice_classe in range(numero_classi):
        tp = cm[indice_classe, indice_classe]
        fp = cm[:, indice_classe].sum() - tp
        fn = cm[indice_classe, :].sum() - tp
        tn = totale - tp - fp - fn
        tp_list.append(str(int(tp)))
        fp_list.append(str(int(fp)))
        fn_list.append(str(int(fn)))
        tn_list.append(str(int(tn)))
    return ";".join(tp_list), ";".join(fp_list), ";".join(fn_list), ";".join(tn_list)


if __name__ == "__main__":
    import time, psutil

    wine_full = load_wine()
    X_all_full = StandardScaler().fit_transform(wine_full.data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all_full, wine_full.target, test_size=0.2, random_state=42)
    X_raw_train = X_tr

    y_onehot_all = label_binarize(wine_full.target, classes=[0, 1, 2]).astype(float)
    y_onehot_train, y_onehot_test, _, _ = train_test_split(
        y_onehot_all, wine_full.target, test_size=0.2, random_state=42)

    print(SEP)
    print("SETUP")
    print(SEP)
    np.random.seed(42)
    W1 = np.random.randn(M, L) * np.sqrt(1.0 / M)
    W2 = np.random.randn(L, C) * np.sqrt(1.0 / L)
    b1 = np.zeros(L)
    b2 = np.zeros(C)

    csv_init()
    print(f"  CSV risultati: {CSV_PATH}")

    cc, level_budget, bootstrap_depth, depth = build_context(SECURE)
    SLOT = cc.GetRingDimension() // 2

    # N = il massimo batch compatibile con questo SLOT (divisione intera),
    # limitato alla dimensione dell'intero training set, dato che qui M*L è
    # piccolo (104) e SLOT // (M*L) supererebbe facilmente i 142 esempi
    # disponibili.
    N_MAX = min(SLOT // (M * L), len(X_raw_train))
    N_MINI_BATCHES = math.ceil(len(X_raw_train) / N_MAX)
    N = len(X_raw_train) // N_MINI_BATCHES

    print(f"  Bootstrap depth:  {bootstrap_depth}")
    print(f"  Depth totale:     {depth}  (limite={depth-2})")
    print(f"  Ring dim:         {cc.GetRingDimension()}")
    print(f"  SLOT:             {SLOT}")
    print(f"  N (batch size):   {N}  →  N_MINI_BATCHES={N_MINI_BATCHES}  "
          f"(train={len(X_raw_train)}, scartati per epoca="
          f"{len(X_raw_train) - N_MINI_BATCHES*N})")
    kp = genera_chiavi(cc, level_budget)

    print(f"\n{SEP}")
    print("CALIBRAZIONE POST_BOOT")
    print(SEP)
    POST_BOOT = calibra_post_boot(cc, kp, depth)
    print(f"  POST_BOOT={POST_BOOT}  livelli utili per ciclo={(depth-2)-POST_BOOT}")

    X_batches = []
    y_batches = []
    for i in range(N_MINI_BATCHES):
        batch = X_raw_train[i*N:(i+1)*N]
        xb = np.zeros(SLOT)
        xb[:N*M] = batch.flatten()
        X_batches.append(xb)
        y_batches.append(y_onehot_train[i*N:(i+1)*N])

    X_ripetuto_ct_batches = []
    for i in range(N_MINI_BATCHES):
        x_ripetuto_plain = plain_flat_e_repeat_righe(X_batches[i], L, N, M, SLOT)
        X_ripetuto_ct_batches.append(encrypt(cc, kp, x_ripetuto_plain))
    cache_x_trasposta = [{} for _ in range(N_MINI_BATCHES)]

    W1_flat = np.zeros(SLOT); W1_flat[:M*L] = W1.flatten()
    W2_flat = np.zeros(SLOT); W2_flat[:L*C] = W2.flatten()
    b1_flat = np.zeros(SLOT); b1_flat[:L] = b1
    b2_flat = np.zeros(SLOT); b2_flat[:C] = b2

    W1_ct = encrypt_at(cc, kp, W1_flat, POST_BOOT)
    W2_ct = encrypt_at(cc, kp, W2_flat, POST_BOOT)
    b1_ct = encrypt_at(cc, kp, b1_flat, POST_BOOT)
    b2_ct = encrypt_at(cc, kp, b2_flat, POST_BOOT)

    print(f"\n  Pesi cifrati: W1={livello(W1_ct)}  W2={livello(W2_ct)}  ← devono essere {POST_BOOT}")

    lr = 0.1
    EPOCHE = 40

    print(f"\n{SEP}")
    print(f"TRAINING — {EPOCHE} epoche")
    print(SEP)

    t_totale = time.time()
    tempo_cumulativo = 0.0

    for epoca in range(EPOCHE):
        t_ep = time.time()

        dW1_acc = None; dW2_acc = None
        db1_acc = None; db2_acc = None

        for i in range(N_MINI_BATCHES):
            z2_con_bias_ct, a1_ct, z1_con_bias_ct = forward(
                cc, kp, X_ripetuto_ct_batches[i], W1_ct, b1_ct, W2_ct, b2_ct)
            dW1_i, db1_i, dW2_i, db2_i = backward(
                cc, kp, z2_con_bias_ct, a1_ct, z1_con_bias_ct, W2_ct,
                X_batches[i], y_batches[i], POST_BOOT, cache_x_trasposta[i])

            print(f"    batch {i+1}/{N_MINI_BATCHES} completato")

            if dW1_acc is None:
                dW1_acc = dW1_i; dW2_acc = dW2_i
                db1_acc = db1_i; db2_acc = db2_i
            else:
                dW1_acc = cc.EvalAdd(dW1_acc, dW1_i)
                dW2_acc = cc.EvalAdd(dW2_acc, dW2_i)
                db1_acc = cc.EvalAdd(db1_acc, db1_i)
                db2_acc = cc.EvalAdd(db2_acc, db2_i)

        if N_MINI_BATCHES > 1:
            scale = cc.MakeCKKSPackedPlaintext([1.0/N_MINI_BATCHES]*SLOT, 1, dW1_acc.GetLevel(), None, SLOT)
            dW1_ct = cc.EvalMult(dW1_acc, scale)
            dW2_ct = cc.EvalMult(dW2_acc, scale)
            db1_ct = cc.EvalMult(db1_acc, scale)
            db2_ct = cc.EvalMult(db2_acc, scale)

            dW1_ct = force_slots(cc, dW1_ct); dW1_ct = cc.EvalBootstrap(dW1_ct)
            dW2_ct = force_slots(cc, dW2_ct); dW2_ct = cc.EvalBootstrap(dW2_ct)
            db1_ct = force_slots(cc, db1_ct); db1_ct = cc.EvalBootstrap(db1_ct)
            db2_ct = force_slots(cc, db2_ct); db2_ct = cc.EvalBootstrap(db2_ct)
        else:
            dW1_ct, dW2_ct, db1_ct, db2_ct = dW1_acc, dW2_acc, db1_acc, db2_acc

        W1_ct = cc.EvalSub(W1_ct, cc.EvalMult(dW1_ct,
                    cc.MakeCKKSPackedPlaintext([lr]*SLOT, 1, dW1_ct.GetLevel(), None, SLOT)))
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

        (acc_tr, acc_te, f1_tr, f1_te, prec_tr, prec_te, rec_tr, rec_te, cm_te,
         mse_tr, mse_te, norm_W1, norm_W2, norm_b1, norm_b2, max_W1, max_W2) = eval_accuracy(
            cc, kp, W1_ct, b1_ct, W2_ct, b2_ct, X_tr, y_tr, X_te, y_te,
            y_onehot_train, y_onehot_test)

        t_ep = time.time() - t_ep
        tempo_cumulativo += t_ep
        proc = psutil.Process(os.getpid())

        print(f"  Epoca {epoca+1:3d} | "
              f"acc={acc_tr:.4f}/{acc_te:.4f} | "
              f"f1={f1_tr:.4f}/{f1_te:.4f} | "
              f"prec={prec_tr:.4f}/{prec_te:.4f} | "
              f"rec={rec_tr:.4f}/{rec_te:.4f} | "
              f"t={t_ep:.0f}s | RAM={proc.memory_info().rss/1e9:.1f}GB")
        if (epoca + 1) % 10 == 0:
            print(f"    CM test: {cm_te[0]}")
            print(f"             {cm_te[1]}")
            print(f"             {cm_te[2]}")

        tp_str, fp_str, fn_str, tn_str = cm_to_tp_fp_fn_tn(cm_te)
        csv_append_row([
            epoca + 1,
            acc_tr, acc_te, f1_tr, f1_te,
            prec_tr, prec_te, rec_tr, rec_te,
            mse_tr, mse_te,
            norm_W1, norm_W2, norm_b1, norm_b2,
            max_W1, max_W2,
            tp_str, fp_str, fn_str, tn_str,
            t_ep, tempo_cumulativo,
        ])

    print(f"\n{SEP}")
    print(f"FINE TRAINING")
    print(f"  Tempo totale: {(time.time()-t_totale)/60:.1f} min")
    (acc_tr_fin, acc_te_fin, f1_tr_fin, f1_te_fin, prec_tr_fin, prec_te_fin,
     rec_tr_fin, rec_te_fin, cm_te_fin, mse_tr_fin, mse_te_fin,
     norm_W1_fin, norm_W2_fin, norm_b1_fin, norm_b2_fin, max_W1_fin, max_W2_fin) = eval_accuracy(
        cc, kp, W1_ct, b1_ct, W2_ct, b2_ct, X_tr, y_tr, X_te, y_te,
        y_onehot_train, y_onehot_test)
    print(f"  Accuracy  — train={acc_tr_fin:.4f}  test={acc_te_fin:.4f}")
    print(f"  F1 macro  — train={f1_tr_fin:.4f}  test={f1_te_fin:.4f}")
    print(f"  Precision — train={prec_tr_fin:.4f}  test={prec_te_fin:.4f}")
    print(f"  Recall    — train={rec_tr_fin:.4f}  test={rec_te_fin:.4f}")
    print(f"  MSE       — train={mse_tr_fin:.6f}  test={mse_te_fin:.6f}")
    print(f"  Confusion matrix (test):")
    print(f"    {cm_te_fin[0]}")
    print(f"    {cm_te_fin[1]}")
    print(f"    {cm_te_fin[2]}")
    print(f"\n  Risultati completi salvati in: {CSV_PATH}")