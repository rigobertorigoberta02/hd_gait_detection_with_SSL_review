# read_data.py — versión ultraligera para CSV x,y,z,label
import os
import csv
import numpy as np

CSV_DIR  = '/mlwell-data2/dafna/PACEHD_for_ssl_paper'          # ← tus 1.csv, 2.csv…
NPZ_DIR  = '/mlwell-data2/dafna/daily_living_data_array/PACE'  # ← destino .npz
ACC_FS   = 100  # Hz (solo informativo; los .csv ya vienen así)

os.makedirs(NPZ_DIR, exist_ok=True)

def read_acc_data(csv_path):
    acc_rows, label_rows = [], []
    with open(csv_path, 'r') as fh:
        reader = csv.reader((ln.replace('\0', '') for ln in fh))
        next(reader, None)                 # salta cabecera
        for row in reader:
            if len(row) < 4:
                continue
            acc_rows.append([float(row[0]), float(row[1]), float(row[2])])
            label_rows.append(int(float(row[3])))
    return (np.asarray(acc_rows,  dtype='float32'),
            np.asarray(label_rows, dtype='int8'))

def main():
    for fname in sorted(os.listdir(CSV_DIR)):
        if not fname.endswith('.csv'):
            continue
        pid     = os.path.splitext(fname)[0]      # '1', '2', ...
        csvfile = os.path.join(CSV_DIR, fname)

        acc, labels = read_acc_data(csvfile)
        if acc.shape[0] != labels.shape[0]:
            print(f'⚠ {pid}: longitudes distintas, lo salto')
            continue

        out_npz = os.path.join(NPZ_DIR, f'{pid}.npz')
        np.savez(out_npz, acc, labels)   # arr_0 = acc, arr_1 = labels  ✅

        print(f'✓ {pid}.npz listo  ({acc.shape[0]} muestras)')

if __name__ == '__main__':
    main()
