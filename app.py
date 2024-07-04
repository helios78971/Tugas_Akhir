from track import *
import torch
import streamlit as st
from streamlit_option_menu import option_menu
import os
import base64
import shutil
import cv2
import numpy as np

# Akses file css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Untuk pilih file
def file_selector(text, folder_path, output):
    filenames = os.listdir(folder_path)
    selected_filename = expander.selectbox(f'{text}', filenames)
    if selected_filename:
        return os.path.join(folder_path, selected_filename)
    else:
        expander.markdown(f"#### {output}")

# Untuk cari video dan teks
def find_videos_and_text(directory):
    if not directory:
        return None, None

    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None, None

    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    text_extensions = ['.txt']
    videos = []
    texts = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(os.path.join(root, file))
            elif any(file.lower().endswith(ext) for ext in text_extensions):
                texts.append(os.path.join(root, file))

    return videos, texts

# Untuk baca teks
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
        file_contents = file_contents.replace('\n', '  \n')
    return file_contents

# Untuk background web
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;}}
    </style>
    """,
    unsafe_allow_html = True
    )

# Hapus folder
def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        expander.markdown(f"Error: {folder_path} : {e.strerror}")

#Gambar ROI
def draw_roi(image, coordinates):
    if coordinates is not None:
        coordinates = [tuple(map(int, coord.split(','))) for coord in coordinates]
        coordinates = np.array(coordinates).reshape((-1, 1, 2))
        cv2.polylines(image, [coordinates], isClosed=True, color=(0, 255, 0), thickness=2)
    return coordinates


def sidebar(model, assigned_class_id, hasil, confidence, koordinat, track_button, stop_button, inputan):
    # Pilih kelas tertentu
    custom_class = expander.checkbox('Kustomisasi Kelas')
    assigned_class_id = [0, 1, 2, 3]
    names = ['Motor', 'Mobil', 'Bus', 'Truk']

    if custom_class:
        assigned_class_id = []
        assigned_class = expander.multiselect('Objek Kelas', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    # Atur confidence
    confidence = expander.slider('Confidence', min_value=0.0, max_value=1.0, value=0.8)

    with expander.popover("Region of Interest (ROI)"):
        kiri_atas = st.text_input("Kiri Atas (x1, y1): ", value="300, 700")
        kanan_atas = st.text_input("Kanan Atas (x2, y2): ", value="1650, 700")
        kanan_bawah = st.text_input("Kanan Bawah (x3, y3): ", value="1650, 900")
        kiri_bawah = st.text_input("Kiri Bawah (x4, y4): ", value="300, 900")
    koordinat = [kiri_atas, kanan_atas, kanan_bawah, kiri_bawah]

    # Simpan video
    simpan_video = expander.radio("Simpan Video", ('Ya', 'Tidak'))
    if simpan_video == 'Ya':
        hasil = True
    else:
        hasil = False
    
    # Pilih model
    text = 'Pilih model'
    model = file_selector(text, folder_path, output)

    kol1, kol2 = expander.columns(2)
    track_button = kol1.button('START', key="1", use_container_width=True)
    stop_button = kol2.button('STOP', key="2", use_container_width=True)  

    return model, assigned_class_id, hasil, confidence, koordinat, track_button, stop_button

if __name__ == '__main__':
    st.set_page_config(page_title="Vehicle Counting", page_icon="🚗", layout="wide")
    local_css("asset/style.css")
    add_bg_from_local('asset/bg.jpg')
    output = 'Tidak ada file/folder'

    with st.sidebar:
        selected = option_menu("Menu",
                                ["Home", "Deteksi & Hitung", "Data", "About Me"],
                                icons=['house-fill', 'car-front-fill', 'archive-fill', 'person-fill'], 
                                default_index=0,
                                orientation="vertical",
                                menu_icon="menu-app-fill",
                                styles={"container":{"background-color":"none"}})
        
    if selected == "Home":
        st.write("# Selamat Datang!")
        with st.container(border=True):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(
                    """
                    Aplikasi web ini adalah implementasi dari algoritma Deep Learning yaitu YOLOv5 & DeepSORT untuk melakukan perhitungan jumlah kendaraan secara otomatis. \n
                    👈 Silahkan gunakan menu disamping untk bernavigasi di aplikasi web ini \n
                    Cara untuk menggunakan aplikasi web ini cukup mudah:
                    1. Klik menu Deteksi & Hitung
                    2. Atur konfigurasi yang diinginkan
                    3. Tekan tombol START \n

                    Untuk analisis yang lebih mendalam dapat dilakukan pada menu Data, di menu ini kita dapat melihat hasil rekaman dari perhitungan yang telah dilakukan sebelumnya, 
                    jika rekaman tidak ada maka hanya data pada rekaman tersebut yang akan dilihat! \n
                    Penjelasan dari opsi konfigurasi dapat dilihat dibawah ini!
                """)

            with col2:
                st.image("./asset/welcome.gif", caption="Vroom vroom!!!")

            col3, col4 = st.columns([1, 4])
            with col3:
                st.image("./asset/setting.png")
            
            with col4:
                st.markdown(
                    """
                    Konfigurasi yang dapat dilakukan adalah:
                    - **Kustomisasi Kelas**: Opsi ini menyeleksi kelas yang ingin dideteksi
                    
                    - **Confidence**: Opsi ini mengatur persentase probabilitas objek yang dideteksi
                    
                    - **Region of Interest (ROI)**: Opsi ini mengatur koordinat dari bidang agar dapat melakukan perhitungan kendaraan
                    
                    - **Simpan Video**: Opsi ini akan menyimpan video hasil dari pendeteksian dan perhitungan

                    - **Pilih model**: Opsi ini menawarkan beberapa model siap pakai dari hasil training
                    """)
                
                with st.container(border=True):
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown(
                            """
                            Untuk mengatur ROI cukup mudah, hal yang harus diketahui adalah variabel x mengatur posisi horizontal dan y mengatur posisi vertikal. Untuk lebih detailnya:
                            - **x1, y1**: Mengatur lokasi dari pojok kiri atas

                            - **x2, y2**: Mengatur lokasi dari pojok kanan atas

                            - **x3, y3**: Mengatur lokasi dari pojok kanan bawah

                            - **x4, y4**: Mengatur lokasi dari pojok kiri atas
                            """)
                        st.image("./asset/roi-config.png")

                    with col6:
                        st.image("./asset/roi.png")
    
    elif selected == "Deteksi & Hitung":
        expander = st.sidebar.expander("Pengaturan")
        folder_path='./model'
        col1, col2 = st.columns(2)
        status1, status2= st.columns(2)

        # Pilih opsi video
        # input_source = expander.radio("Sumber Video",('Video Lokal', 'RTSP'))
        
        model = None
        assigned_class_id = None
        hasil = None
        confidence = None
        koordinat = None
        track_button = None
        stop_button = None

        # if input_source=='Video Lokal':
            # Upload video
        video_filename = None
        video_file_buffer = expander.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])
        
        (model, assigned_class_id, hasil, confidence, koordinat, track_button, stop_button) = sidebar(model, 
                                                                                                        assigned_class_id, 
                                                                                                        hasil, confidence, 
                                                                                                        koordinat, 
                                                                                                        track_button, 
                                                                                                        stop_button, 
                                                                                                        video_file_buffer)

        if video_file_buffer:
            video_filename = os.path.join('data/input', video_file_buffer.name)
            with open(video_filename, 'wb') as f:
                f.write(video_file_buffer.getbuffer())

        with col1:
            st.write("# Visualisasi ROI")
            with st.container(border=True, height=460):
                if video_filename:
                    visualisasi = cv2.VideoCapture(video_filename)
                    ret, frame = visualisasi.read()
                    cv2.imwrite('./data/visualisasi.jpg', frame)
                    visualisasi.release()
                    visualisasi_output = cv2.imread('./data/visualisasi.jpg')
                    draw_roi(visualisasi_output, koordinat)
                    st.image(visualisasi_output, channels="BGR")
                    
        with status1:
            st.markdown("## Status")
            mobil_text = st.markdown("##### Mobil: 0")
            bus_text = st.markdown("##### Bus: 0")
            truk_text = st.markdown("##### Truk: 0")
            motor_text = st.markdown("##### Motor: 0")
            jumlah_text = st.markdown("##### Total Kendaraan: 0")
        
        with status2:
            st.markdown("## Status Sistem")
            ram_text = st.markdown("##### Memori : 0%")
            cpu_text = st.markdown("##### CPU: 0%")
            gpu_text = st.markdown("##### Memori GPU: 0 MB")
            fps_text = st.markdown("##### FPS: 0")

        with col2:
            st.write("# Output Video")
            with st.container(border=True, height=460):
                stframe = st.empty()
        
        opt = parse_opt()
        if track_button:
            reset()
            opt.conf_thres = confidence
            opt.source = f'data/input/{video_file_buffer.name}'
            with torch.no_grad():
                detect(opt, stframe, mobil_text, bus_text, truk_text, motor_text, jumlah_text, ram_text, cpu_text, gpu_text, fps_text, koordinat, assigned_class_id, hasil, model)
            
        elif stop_button:
            st.stop()
        
        # if input_source=='RTSP':
        #     rtsp_input = expander.text_input("Alamat IP")
        #     (model, assigned_class_id, hasil, confidence, koordinat, track_button, stop_button) = sidebar(model, 
        #                                                                                                   assigned_class_id, 
        #                                                                                                   hasil, confidence, 
        #                                                                                                   koordinat, 
        #                                                                                                   track_button, 
        #                                                                                                   stop_button, 
        #                                                                                                   rtsp_input)
        #     with col1:
        #         st.write("# Visualisasi ROI")
        #         with st.container(border=True, height=460):
        #             if rtsp_input:
        #                 visualisasi = cv2.VideoCapture(rtsp_input)
        #                 ret, frame = visualisasi.read()
        #                 cv2.imwrite('./data/visualisasi.jpg', frame)
        #                 visualisasi.release()
        #                 visualisasi_output = cv2.imread('./data/visualisasi.jpg')
        #                 draw_roi(visualisasi_output, koordinat)
        #                 st.image(visualisasi_output, channels="BGR")

        #     with col2:
        #         st.write("# Output Video")
        #         with st.container(border=True, height=460):
        #             stframe = st.empty()
            
        #     with status1:
        #         st.markdown("## Status")
        #         mobil_text = st.markdown("##### Mobil: 0")
        #         bus_text = st.markdown("##### Bus: 0")
        #         truk_text = st.markdown("##### Truk: 0")
        #         motor_text = st.markdown("##### Motor: 0")
        #         jumlah_text = st.markdown("##### Total Kendaraan: 0")
            
        #     with status2:
        #         st.markdown("## Status Sistem")
        #         ram_text = st.markdown("##### Memori : 0%")
        #         cpu_text = st.markdown("##### CPU: 0%")
        #         gpu_text = st.markdown("##### Memori GPU: 0 MB")
        #         fps_text = st.markdown("##### FPS: 0")
            
        #     opt = parse_opt()
        #     if track_button:
        #         if str(rtsp_input) == '':
        #             st.markdown('**No Input**')
                
        #         else:
        #             reset()
        #             opt.conf_thres = confidence
        #             opt.source = rtsp_input
        #             with torch.no_grad():
        #                 detect(opt, stframe, mobil_text, bus_text, truk_text, motor_text, jumlah_text, ram_text, cpu_text, gpu_text, fps_text, koordinat, assigned_class_id, hasil, model)

        #     elif stop_button:
        #         st.stop()

    elif selected == "Data":
        expander = st.sidebar.expander("Pengaturan")
        folder_path='./data/output'
        text = 'Pilih rekaman'
        filename = file_selector(text, folder_path, output)
        hapus = expander.button('Hapus rekaman?', key="1", use_container_width=True)
        datavideo, datateks = find_videos_and_text(filename)
        kolom1, kolom2 = st.columns(2)
        with kolom1:
            st.write("# Data Rekaman")
            with st.container(border=True, height=550):
                if datateks:
                    for teks in datateks:
                        file_contents = read_text_file(teks)
                        st.markdown(f'{file_contents}')
        
        with kolom2:
            st.write("# Video")
            with st.container(border=True, height=550):
                if datavideo:
                    for video in datavideo:
                        st.video(video)
                
                else:
                    st.markdown('### Tidak ada rekaman!')
        
        if hapus:
            remove_folder(filename)
    
    elif selected == "About Me":
        st.write("# Hello! 👋🏻")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("./asset/profile.png", use_column_width=True, caption='"Seperti kata pepatah, GAS KAN!!!!"')

        with col2:
            st.markdown(
                """
                ##### Perkenalkan nama saya Frederick Benaya Situmorang \n
                Saya berada di End Game semester sekarang, aplikasi web ini adalah proyek untuk tugas akhir saya, 
                jika ada pertanyaan atau kendala dengan aplikasi web ini, kalian bisa Open Issue di [sini](https://github.com/helios78971/Proyek_TA/issues) ya gaes yak. \n
                
                ##### Info lebih lanjut:
                - Github Account: [link](https://github.com/helios78971)
                - Email: stone78971@gmail.com
                - LinkedIn: [link](https://www.linkedin.com/in/frederick-benaya-situmorang/)
                
                ##### Happy Coding! 🎉🎉🎉
                """)