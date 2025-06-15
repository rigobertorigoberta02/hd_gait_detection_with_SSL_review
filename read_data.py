import os
import numpy as np
import csv

# === RUTAS BÁSICAS (ajusta a tu gusto) =========================
PACE_DAILY_DATA_DIR   = '/mlwell-data2/dafna/PACEHD_for_ssl_paper'     # donde viven los .csv
PACE_DAILY_TARGET_DIR = '/mlwell-data2/dafna/daily_living_data_array/PACE'  # donde guardaremos .npz
ACC_SAMPLE_RATE = 100  # no lo usamos aquí, pero lo dejo por coherencia

# ----------------------------------------------------------------
def read_acc_csv(file_path):
    """
    Lee un CSV con cabecera x,y,z,label y devuelve:
        acc_data  -> np.array shape (N, 3)   ❶
        label_arr -> np.array shape (N,)     ❷
    """
    acc_rows, lab_rows = [], []
    with open(file_path, 'r') as fh:
        reader = csv.reader((ln.replace('\0', '') for ln in fh))
        next(reader, None)                      # salta cabecera
        for row in reader:
            if len(row) < 4:                     # fila corta = descártala
                continue
            acc_rows.append([float(row[0]), float(row[1]), float(row[2])])
            lab_rows.append(int(float(row[3])))  # -1/0/1
    return np.asarray(acc_rows, dtype='float32'), np.asarray(lab_rows, dtype='int8')


def main_daily():
    os.makedirs(PACE_DAILY_TARGET_DIR, exist_ok=True)
    for file in sorted(os.listdir(PACE_DAILY_DATA_DIR)):
        if not file.endswith('.csv'):
            continue

        patient_id = os.path.splitext(file)[0]   # '1', '2', ...
        csv_path   = os.path.join(PACE_DAILY_DATA_DIR, file)

        acc_data, label_arr = read_acc_csv(csv_path)

        # ⚠ El orden es importante: 1º acc, 2º labels = arr_0, arr_1
        npz_out = os.path.join(PACE_DAILY_TARGET_DIR, f'{patient_id}.npz')
        np.savez(npz_out, acc_data, label_arr)

        print(f'✓ {patient_id}.npz  ({acc_data.shape[0]} muestras)')

if __name__ == "__main__":
    main_daily()
