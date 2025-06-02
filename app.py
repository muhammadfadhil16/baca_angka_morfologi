import os
import cv2
import numpy as np
import mysql.connector

TEMPLATE_DIR = 'templates'
IMAGE_PATH = '3.jpg'


def extract_hu_moments(contour):
    """
    Ekstrak 7 Hu Moments dari sebuah kontur.
    """
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu

def save_features_to_db(digit, features):
    """
    Simpan fitur Hu Moments ke database.
    """
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='aktif2025',
        database='digit_morphology'
    )
    cursor = conn.cursor()
    query = "INSERT INTO features (digit, hu1, hu2, hu3, hu4, hu5, hu6, hu7) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    cursor.execute(query, (digit, *features))
    conn.commit()
    cursor.close()
    conn.close()

def load_features_from_db():
    """
    Ambil semua fitur digit dari database.
    """
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='aktif2025',
            database='digit_morphology'
        )
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

def recognize_digit_by_features(features, features_db, k=3):
    """
    KNN sederhana berdasarkan Hu Moments.
    """
    distances = []
    for digit, ref_features in features_db:
        distance = np.linalg.norm(features - ref_features)
        distances.append((distance, digit))
    distances.sort()
    nearest = [d for _, d in distances[:k]]
    # Voting
    counts = np.bincount(nearest)
    return np.argmax(counts)

def read_pdam_meter(image_path):
    """
    Membaca angka dari gambar meteran menggunakan fitur Hu Moments dan KNN.
    """
    print("[*] Membaca gambar...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Gambar '{image_path}' tidak ditemukan atau format tidak didukung.")
        return ""
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    print("[*] Memuat fitur dari database...")
    features_db = load_features_from_db()
    if not features_db:
        print("Database fitur kosong! Jalankan train_from_templates() dulu.")
        return ""

    print("[*] Menemukan kontur digit...")
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    meter_reading = ""
    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10 and w < 100 and h < 200:
            cropped_digit = thresh[y:y+h, x:x+w]
            hu_features = extract_hu_moments(contour)
            predicted_digit = recognize_digit_by_features(hu_features, features_db, k=3)

            if predicted_digit is not None:
                meter_reading += str(predicted_digit)

            print(f"Digit ke-{i+1}: {predicted_digit}, Hu Moments: {hu_features}")

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow(f"Digit {predicted_digit}", cropped_digit)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(f"\n[*] Hasil Pembacaan Meter: {meter_reading}")
    return meter_reading

def train_from_templates(template_dir):
    """
    Ekstrak fitur dari semua file di folder template dan simpan ke database.
    """
    for filename in os.listdir(template_dir):
        if filename.endswith(('.jpg', '.png')):
            try:
                digit = int(filename.split('_')[0])
            except ValueError:
                print(f"File {filename} dilewati (format nama tidak sesuai).")
                continue
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"File {filename} gagal dibaca.")
                continue
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                hu_features = extract_hu_moments(largest)
                save_features_to_db(digit, hu_features)
                print(f"✔ Saved: {filename} => {hu_features}")
            else:
                print(f"Tidak ditemukan kontur pada {filename}.")

if __name__ == "__main__":
    # train_from_templates(TEMPLATE_DIR)  # Aktifkan dulu untuk isi database fitur Hu Moments
    read_pdam_meter(IMAGE_PATH)
