import numpy as np

# ============================================================
# DATA GERBANG LOGIKA
# ============================================================
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_AND = [0, 0, 0, 1]
y_OR  = [0, 1, 1, 1]
y_XOR = [0, 1, 1, 0]

MAX_EPOCH = 10000

# ============================================================
# FUNGSI AKTIVASI
# ============================================================
def step_function(x):
    return 1 if x >= 0 else 0

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# ============================================================
# INPUT HELPER
# ============================================================
def input_int(pesan, min_val=None):
    while True:
        try:
            v = int(input(pesan))
            if min_val and v < min_val:
                print(f"  Minimal {min_val}.")
            else:
                return v
        except ValueError:
            print("  Masukkan angka bulat.")

def input_float(pesan):
    while True:
        try:
            return float(input(pesan))
        except ValueError:
            print("  Masukkan angka desimal.")

def pilih(pesan, opsi):
    while True:
        v = input(pesan).strip()
        if v in opsi:
            return v
        print(f"  Pilih: {opsi}")

def garis(c="="):
    print(c * 70)

# ============================================================
# INPUT PARAMETER
# ============================================================
def input_param_slp():
    garis()
    print(" PARAMETER SLP")
    print(" [1] Manual  [2] Random -0.5 s.d. 0.5  [3] Random -1 s.d. 1")
    garis("-")
    p  = pilih(" Pilih [1/2/3]: ", ["1","2","3"])
    lr = input_float(" Learning rate: ")

    if p == "1":
        w1   = input_float(" w1 = ")
        w2   = input_float(" w2 = ")
        bias = input_float(" b  = ")
    elif p == "2":
        w1, w2, bias = np.random.uniform(-0.5, 0.5, 3)
        print(f" [Random] w1={w1:.4f} | w2={w2:.4f} | b={bias:.4f}")
    else:
        w1, w2, bias = np.random.uniform(-1.0, 1.0, 3)
        print(f" [Random] w1={w1:.4f} | w2={w2:.4f} | b={bias:.4f}")

    return float(w1), float(w2), float(bias), lr


def input_param_mlp(n_hidden):
    garis()
    print(" PARAMETER MLP")
    print(" [1] Manual  [2] Random -0.5 s.d. 0.5  [3] Random -1 s.d. 1")
    garis("-")
    p  = pilih(" Pilih [1/2/3]: ", ["1","2","3"])
    lr = input_float(" Learning rate: ")

    if p == "1":
        print(f"\n W1 (2 x {n_hidden}):")
        W1 = np.array([[input_float(f"  W1[x{i+1}→h{j+1}] = ")
                        for j in range(n_hidden)] for i in range(2)])
        print(f" b1 (1 x {n_hidden}):")
        b1 = np.array([[input_float(f"  b1[h{j+1}] = ") for j in range(n_hidden)]])
        print(f" W2 ({n_hidden} x 1):")
        W2 = np.array([[input_float(f"  W2[h{j+1}→y] = ")] for j in range(n_hidden)])
        b2 = input_float(" b2 = ")
    elif p == "2":
        W1 = np.random.uniform(-0.5, 0.5, (2, n_hidden))
        b1 = np.random.uniform(-0.5, 0.5, (1, n_hidden))
        W2 = np.random.uniform(-0.5, 0.5, (n_hidden, 1))
        b2 = float(np.random.uniform(-0.5, 0.5))
        print(f" [Random] W1=\n{np.round(W1,4)}\n b1={np.round(b1,4)}\n W2=\n{np.round(W2,4)}\n b2={b2:.4f}")
    else:
        W1 = np.random.uniform(-1.0, 1.0, (2, n_hidden))
        b1 = np.random.uniform(-1.0, 1.0, (1, n_hidden))
        W2 = np.random.uniform(-1.0, 1.0, (n_hidden, 1))
        b2 = float(np.random.uniform(-1.0, 1.0))
        print(f" [Random] W1=\n{np.round(W1,4)}\n b1={np.round(b1,4)}\n W2=\n{np.round(W2,4)}\n b2={b2:.4f}")

    return W1, b1, W2, float(b2), lr

# ============================================================
# CETAK TABEL
# ============================================================
def cetak_header_slp():
    print(f"{'x1':>4} {'x2':>4} {'Target':>7} {'Net':>9} {'Output':>5} {'Error':>5} "
          f"{'w1_baru':>9} {'w2_baru':>9} {'b_baru':>9}")
    garis("=")

def cetak_baris_slp(x1, x2, t, net, output, error, w1, w2, b):
    print(f"{x1:>4} {x2:>4} {t:>7} {net:>9.4f} {output:>5} {error:>5} "
          f"{w1:>9.4f} {w2:>9.4f} {b:>9.4f}")

def cetak_header_mlp():
    print(f"{'x1':>4} {'x2':>4} {'Target':>7} {'Output':>10} {'Error':>9} {'MSE':>11}")
    garis("=")

def cetak_baris_mlp(x1, x2, t, output, error):
    print(f"{x1:>4} {x2:>4} {t:>7.1f} {output:>10.4f} {error:>9.4f} {error**2:>11.6f}")

# ============================================================
# TRAINING SLP
# ============================================================
def train_slp(target, nama, w1, w2, bias, lr):
    garis("=")
    print(f" SLP — Gerbang {nama}")
    print(f" w1={w1:.4f} | w2={w2:.4f} | b={bias:.4f} | lr={lr} | max_epoch={MAX_EPOCH}")
    garis()

    for epoch in range(1, MAX_EPOCH + 1):
        total_error = 0
        rows = []

        for i in range(4):
            x1, x2 = int(X[i][0]), int(X[i][1])
            t       = target[i]
            net = round(w1*x1 + w2*x2 + bias, 4)
            output  = step_function(net)
            error   = t - output
            total_error += abs(error)
            w1   += lr * error * x1
            w2   += lr * error * x2
            bias += lr * error
            rows.append((x1, x2, t, net, output, error, w1, w2, bias))

        konvergen = (total_error == 0)
        tampil    = (epoch <= 10) or konvergen or (epoch == MAX_EPOCH)

        if tampil:
            garis("-")
            print(f" Epoch {epoch}")
            garis("-")
            cetak_header_slp()
            for r in rows:
                cetak_baris_slp(*r)
            garis("-")
            print(f" Total Error = {total_error}")

        if konvergen:
            print(f" --> KONVERGEN di Epoch {epoch}!")
            garis()
            return
        if epoch == MAX_EPOCH:
            print(f" --> Max epoch ({MAX_EPOCH}) tercapai. Belum konvergen.")
            garis()

# ============================================================
# TRAINING MLP — STEP FUNCTION
# ============================================================
def train_mlp_step(n_h, W1, b1, W2, b2, lr):
    garis("=")
    print(f" MLP XOR — Step Function | {n_h} hidden node")
    garis()

    for epoch in range(1, MAX_EPOCH + 1):
        total_error = 0
        rows = []

        for i in range(4):
            xi = X[i].reshape(1, -1)
            t  = y_XOR[i]

            # Forward
            net_h  = np.dot(xi, W1) + b1
            h      = np.array([[step_function(net_h[0,j]) for j in range(n_h)]])
            net_y  = float(np.dot(h, W2).item() + b2)
            output = step_function(net_y)

            error = t - output
            total_error += abs(error)

            # Update (hanya jika ada error)
            if error != 0:
                W2 += lr * error * h.T
                b2  = float(b2 + lr * error)
                W1 += lr * error * np.dot(xi.T, np.ones((1, n_h)))
                b1 += lr * error * np.ones((1, n_h))

            rows.append((int(X[i][0]), int(X[i][1]), float(t), float(output), float(error)))

        konvergen = (total_error == 0)
        tampil    = (epoch <= 10) or konvergen or (epoch == MAX_EPOCH)

        if tampil:
            garis("-")
            print(f" Epoch {epoch}")
            garis("-")
            cetak_header_mlp()
            for r in rows:
                cetak_baris_mlp(*r)
            garis("-")
            print(f" Total Error = {total_error}")

        if konvergen:
            print(f" --> KONVERGEN di Epoch {epoch}!")
            garis()
            return
        if epoch == MAX_EPOCH:
            print(f" --> Max epoch ({MAX_EPOCH}) tercapai. Belum konvergen.")
            garis()

# ============================================================
# TRAINING MLP — RELU
# ============================================================
def train_mlp_relu(n_h, W1, b1, W2, b2, lr):
    garis("=")
    print(f" MLP XOR — ReLU + Backpropagation | {n_h} hidden node")
    garis()

    for epoch in range(1, MAX_EPOCH + 1):
        total_sq = 0.0
        rows = []

        for i in range(4):
            xi = X[i].reshape(1, -1)
            t  = float(y_XOR[i])

            # Forward
            net_h  = np.dot(xi, W1) + b1
            h      = np.array([[relu(net_h[0,j]) for j in range(n_h)]])
            net_y  = float(np.dot(h, W2).item() + b2)
            output = relu(net_y)

            error = t - output
            total_sq += error ** 2

            # Backward
            delta_output = error * relu_derivative(net_y)
            delta_hidden = np.array([[delta_output * float(W2[j,0]) * relu_derivative(net_h[0,j]) for j in range(n_h)]])

            W2 += lr * h.T * delta_output
            b2  = float(b2 + lr * delta_output)
            W1 += lr * np.dot(xi.T, delta_hidden)
            b1 += lr * delta_hidden

            rows.append((int(X[i][0]), int(X[i][1]), t, output, error))

        mse       = total_sq / 4
        konvergen = (mse <= 0.005)
        tampil    = (epoch <= 10) or konvergen or (epoch == MAX_EPOCH)

        if tampil:
            garis("-")
            print(f" Epoch {epoch}")
            garis("-")
            cetak_header_mlp()
            for r in rows:
                cetak_baris_mlp(*r)
            garis("-")
            print(f" MSE = {mse:.6f}")

        if konvergen:
            print(f" --> KONVERGEN di Epoch {epoch}! (MSE <= 0.005)")
            garis()
            return
        if epoch == MAX_EPOCH:
            print(f" --> Max epoch ({MAX_EPOCH}) tercapai. Belum konvergen.")
            garis()

# ============================================================
# TRAINING MLP — SIGMOID
# ============================================================
def train_mlp_sigmoid(n_h, W1, b1, W2, b2, lr):
    garis("=")
    print(f" MLP XOR — Sigmoid + Backpropagation | {n_h} hidden node")
    garis()

    for epoch in range(1, MAX_EPOCH + 1):
        total_sq = 0.0
        rows = []

        for i in range(4):
            xi = X[i].reshape(1, -1)
            t  = float(y_XOR[i])

            # Forward
            net_h  = np.dot(xi, W1) + b1
            h      = np.array([[sigmoid(net_h[0,j]) for j in range(n_h)]])
            net_y  = float(np.dot(h, W2).item() + b2)
            output = sigmoid(net_y)

            error = t - output
            total_sq += error ** 2

            # Backward
            delta_output = error * sigmoid_derivative(output)
            delta_hidden = np.array([[delta_output * float(W2[j,0]) * sigmoid_derivative(h[0,j]) for j in range(n_h)]])

            W2 += lr * h.T * delta_output
            b2  = float(b2 + lr * delta_output)
            W1 += lr * np.dot(xi.T, delta_hidden)
            b1 += lr * delta_hidden

            rows.append((int(X[i][0]), int(X[i][1]), t, output, error))

        mse       = total_sq / 4
        konvergen = (mse <= 0.005)
        tampil    = (epoch <= 10) or konvergen or (epoch == MAX_EPOCH)

        if tampil:
            garis("-")
            print(f" Epoch {epoch}")
            garis("-")
            cetak_header_mlp()
            for r in rows:
                cetak_baris_mlp(*r)
            garis("-")
            print(f" MSE = {mse:.6f}")

        if konvergen:
            print(f" --> KONVERGEN di Epoch {epoch}! (MSE <= 0.005)")
            garis()
            return
        if epoch == MAX_EPOCH:
            print(f" --> Max epoch ({MAX_EPOCH}) tercapai. Belum konvergen.")
            garis()

# ============================================================
# PROGRAM UTAMA
# ============================================================
def main():
    while True:
        garis("=")
        print("      PROGRAM JARINGAN SYARAF TIRUAN — GERBANG LOGIKA")
        garis("=")
        print(" [1] SLP — Gerbang AND dan OR")
        print(" [2] MLP — Gerbang XOR (Step, ReLU, Sigmoid)")
        print(" [0] Keluar")
        garis("=")
        menu = pilih(" Pilih [0/1/2]: ", ["0","1","2"])

        if menu == "0":
            print(" Terima kasih!")
            break

        elif menu == "1":
            w1, w2, bias, lr = input_param_slp()
            garis()
            train_slp(y_AND, "AND", w1, w2, bias, lr)
            train_slp(y_OR,  "OR",  w1, w2, bias, lr)

        elif menu == "2":
            n_h = input_int(" Jumlah hidden node (min 2): ", min_val=2)
            W1, b1, W2, b2, lr = input_param_mlp(n_h)
            garis()
            train_mlp_step   (n_h, W1.copy(), b1.copy(), W2.copy(), b2, lr)
            train_mlp_relu   (n_h, W1.copy(), b1.copy(), W2.copy(), b2, lr)
            train_mlp_sigmoid(n_h, W1.copy(), b1.copy(), W2.copy(), b2, lr)

        print()
        if pilih(" Kembali ke menu? [y/n]: ", ["y","n","Y","N"]).lower() == "n":
            print(" Terima kasih!")
            break

if __name__ == "__main__":
    main()