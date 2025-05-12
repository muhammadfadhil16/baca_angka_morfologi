import os
import cv2
import numpy as np
import mysql.connector

TEMPLATE_DIR = 'templates'
IMAGE_PATH = 'B.jpg'

def save_features_to_db(digit, features, x, y):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='digit_morphology'
    )
    cursor = conn.cursor()
    query = "INSERT INTO features (digit, aspect_ratio, extent, solidity) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (digit, *features))
    conn.commit()
    cursor.close()
    conn.close()

def extract_morphology_features(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    return aspect_ratio, extent, solidity

def load_digit_templates(template_dir):
    """
    Memuat banyak template per digit dan menyimpan fitur ke database.
    """
    templates = {}  # digit: list of (template_image, features)
    for filename in os.listdir(template_dir):
        if filename.endswith(('.jpg', '.png')):
            try:
                digit = int(filename.split('_')[0])
            except ValueError:
                continue  # Lewati jika nama file tidak sesuai format

            path = os.path.join(template_dir, filename)
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                features = extract_morphology_features(largest_contour)
                save_features_to_db(digit, features)
                templates.setdefault(digit, []).append((template_img, features))
    return templates

def recognize_digit_with_template(cropped_digit, templates):
    """
    Kenali digit menggunakan template matching dan fitur morfologi.
    """
    # Preprocessing digit yang dicrop
    _, thresh = cv2.threshold(cropped_digit, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Ambil kontur terbesar dari digit yang dicrop
    largest_contour = max(contours, key=cv2.contourArea)
    cropped_features = extract_morphology_features(largest_contour)

    best_match = None
    best_score = float('inf')

    for digit, template_list in templates.items():
        for template_img, template_features in template_list:
            # Resize template agar sesuai dengan ukuran digit yang dicrop
            resized_template = cv2.resize(template_img, (cropped_digit.shape[1], cropped_digit.shape[0]))

            # Template matching
            result = cv2.matchTemplate(cropped_digit, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Hitung jarak fitur morfologi
            template_features = np.array(template_features)
            morphology_distance = np.linalg.norm(cropped_features - template_features)

            # Kombinasikan skor template matching dan jarak fitur morfologi
            morphology_weight = 1.0
            template_weight = 2.0
            combined_score = template_weight * -max_val + morphology_weight * morphology_distance

            # Debugging: Cetak skor template matching
            print(f"Template Digit: {digit}, Max Val: {max_val}, Morphology Distance: {morphology_distance}, Combined Score: {combined_score}")

            if combined_score < best_score:
                best_score = combined_score
                best_match = digit

            # Debugging: Tampilkan hasil template matching
            cv2.imshow("Cropped Digit", cropped_digit)
            cv2.imshow("Resized Template", resized_template)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    return best_match

def read_pdam_meter(image_path):
    print("[*] Membaca gambar...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    print("[*] Memuat template digit...")
    templates = load_digit_templates(TEMPLATE_DIR)

    print("[*] Menemukan kontur digit...")
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Urutkan kontur berdasarkan posisi horizontal (x)
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    meter_reading = ""
    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10 and w < 100 and h < 200:  # Filter kontur kecil dan besar
            cropped_digit = thresh[y:y+h, x:x+w]

            # Visualisasi kontur
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Kenali digit menggunakan template matching
            recognized_digit = recognize_digit_with_template(cropped_digit, templates)
            if recognized_digit is not None:
                meter_reading += str(recognized_digit)

            # Debugging: Cetak koordinat dan hasil pengenalan
            print(f"Digit ke-{i+1}: x={x}, y={y}, w={w}, h={h}")
            print(f"  Recognized Digit: {recognized_digit}")

            # Tampilkan hasil cropping
            cv2.imshow(f"Digit ke-{i+1}: {recognized_digit}", cropped_digit)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(f"\n[*] Hasil Pembacaan Meter: {meter_reading}")
    return meter_reading

if __name__ == "__main__":
    read_pdam_meter(IMAGE_PATH)
