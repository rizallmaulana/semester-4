# Meng-import library
import cv2
import numpy as np

# Buat fungsi untuk menyesuaikan ukuran gambar
def show_resized_image(window_name, image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)

# Baca gambar asli
img = cv2.imread('img/gambar-gunung-dari-samping-3.jpg')  # Menggunakan '/' atau '\\' untuk kompatibilitas yang lebih baik
# Display original image
show_resized_image('Gambar Original', img, 600, 400)

# Konversi ke skala abu-abu
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_resized_image('Gambar Grayscale', img_gray, 600, 400)

# Mengaburkan gambar untuk deteksi tepi yang lebih baik
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
show_resized_image('Gambar Gaussian Blur', img_blur, 600, 400)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
# Menampilkan gambar Deteksi Tepi Canny
show_resized_image('Deteksi Canny Edge', edges, 600, 400)

# Salinan gambar hasil Canny untuk menggambar garis
img_hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Transformasi Hough
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Daftar untuk menyimpan kemiringan
angles = []

# Hasil Transformasi Hough
if lines is not None:
    for rho, theta in lines[:, 0]:
        # Skip lines that are close to vertical or horizontal
        if 10 * np.pi / 180 < theta < 170 * np.pi / 180:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Hitung kemiringan dalam derajat dan tambahkan ke daftar
            angle = 90 - np.degrees(theta)
            angles.append(angle)

    # Menghitung rata-rata kemiringan
    if angles:
        average_angle = np.mean(angles)
        print(f"Rata-rata derajat kemiringan: {average_angle:.2f} derajat")
else:
    average_angle = None
    print("Tidak ada garis yang terdeteksi")

# Jika ada kemiringan yang dihitung
if average_angle is not None:
    # Membuat gambar baru dengan teks kemiringan
    height, width, _ = img_hough.shape
    result_image = np.ones((height + 100, width, 3), dtype=np.uint8) * 255  # Tambahkan ruang untuk teks di bawah gambar
    result_image[:height, :width] = img_hough
    
    # Tambahkan ruang untuk teks di bawah gambar
    text = f'Kemiringan: {average_angle:.2f} derajat'
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (result_image.shape[1] - text_size[0]) // 2
    cv2.putText(result_image, text, (text_x, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Tampilkan gambar hasil akhir dengan teks kemiringan
    show_resized_image('Hasil Akhir dengan Kemiringan', result_image, 600, 500)


cv2.destroyAllWindows()