import os
import cv2
import numpy as np
import mysql.connector
from collections import Counter

# --- ENCAPSULATION ---
# Semua data (atribut) dan perilaku (metode) terkait pembacaan meter PDAM dibungkus dalam satu kelas.
class PDAMeterReader:
    # --- CONSTRUCTOR ---
    # Fungsi khusus __init__ digunakan untuk inisialisasi objek (membuat instance baru).
    def __init__(self, db_config, template_dir, image_path):
        self.db_config = db_config
        self.template_dir = template_dir
        self.image_path = image_path

    # --- ABSTRACTION ---
    # Metode ini menyembunyikan detail perhitungan Hu Moments dari pengguna kelas.
    def extract_hu_moments(self, image):
        moments = cv2.moments(image)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu

    # --- ABSTRACTION ---
    # Pengguna kelas cukup memanggil metode ini tanpa tahu detail query SQL-nya.
    def save_features_to_db(self, digit, features):
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        query = "INSERT INTO features (digit, hu1, hu2, hu3, hu4, hu5, hu6, hu7) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (digit, *features))
        conn.commit()
        cursor.close()
        conn.close()

    # --- ABSTRACTION ---
    def load_features_from_db(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT digit, hu1, hu2, hu3, hu4, hu5, hu6, hu7 FROM features")
            data = cursor.fetchall()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            return []
        features_db = []
        for row in data:
            digit = row[0]
            hu = np.array(row[1:], dtype=np.float64)
            features_db.append((digit, hu))
        return features_db

    # --- ABSTRACTION ---
    # Pengguna tidak perlu tahu detail perhitungan jarak dan voting.
    def recognize_digit_by_features(self, features, features_db, k=3):
        distances = []
        for digit, ref_features in features_db:
            distance = np.linalg.norm(features - ref_features)
            distances.append((distance, digit))
        distances.sort()
        nearest = [d for _, d in distances[:k]]
        counter = Counter(nearest)
        return counter.most_common(1)[0][0]

    # --- ABSTRACTION ---
    def read_pdam_meter(self):
        print("[*] Membaca gambar...")
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Gambar '{self.image_path}' tidak ditemukan atau format tidak didukung.")
            return ""
        thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        print("[*] Memuat fitur dari database...")
        features_db = self.load_features_from_db()
        if not features_db:
            print("Database fitur kosong! Jalankan train_from_templates() dulu.")
            return ""

        print("[*] Menemukan kontur digit...")
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        meter_reading = ""
        digit_counter = 1  # Counter manual untuk penomoran digit yang valid
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10 and w < 100 and h < 200:
                cropped_digit = thresh[y:y+h, x:x+w]
                cropped_digit = cv2.resize(cropped_digit, (40, 40)) 
                hu_features = self.extract_hu_moments(cropped_digit)
                predicted_digit = self.recognize_digit_by_features(hu_features, features_db, k=3)

                if predicted_digit is not None:
                    meter_reading += str(predicted_digit)

                print(f"Digit ke-{digit_counter}: {predicted_digit}, Hu Moments: {hu_features}")
                digit_counter += 1

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow(f"Digit {predicted_digit}", cropped_digit)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        print(f"\n[*] Hasil Pembacaan Meter: {meter_reading}")
        return meter_reading

    # --- ABSTRACTION ---
    def train_from_templates(self):
        for filename in os.listdir(self.template_dir):
            if filename.endswith(('.jpg', '.png')):
                try:
                    digit = int(filename.split('_')[0])
                except ValueError:
                    print(f"File {filename} dilewati (format nama tidak sesuai).")
                    continue
                path = os.path.join(self.template_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"File {filename} gagal dibaca.")
                    continue
                _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    hu_features = self.extract_hu_moments(largest)
                    self.save_features_to_db(digit, hu_features)
                    print(f"✔ Saved: {filename} => {hu_features}")
                else:
                    print(f"Tidak ditemukan kontur pada {filename}.")

# --- PEMBUATAN OBJEK DAN PENGGUNAAN (INSTANCE) ---
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'aktif2025',
        'database': 'digit_morphology'
    }
    reader = PDAMeterReader(db_config, 'templates', 'test4deret.png')
    reader.read_pdam_meter()
