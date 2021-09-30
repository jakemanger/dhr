import os
from pydicom import dcmread
from pydicom.filereader import read_file_meta_info, read_dicomdir

imgs = [
    '/Volumes/1tb_ssd/mctv_analysis/Head Scans/P_crassipes_FEG200205_108/FEG200205_1080000.dcm',
    '/Volumes/1tb_ssd/mctv_analysis/Head Scans/P_gracilis_FEG190213_003a_head_HIGHESTpriority/FEG190213_003a000.dcm',
    '/Volumes/1tb_ssd/mctv_analysis/Head Scans/Paraphronima_FEG181024_03_head_sp3_1423158b/Para_sp3_1423158b0000.dcm',
    '/Volumes/1tb_ssd/mctv_analysis/Head Scans/P_crassipes_FEG190801_034_head/FEG190801_03400000.dcm',
    '/Volumes/1tb_ssd/mctv_analysis/Head Scans/P_crassipes_FEG191022_077B_highpriority/FEG191022_07700000.dcm'
]

for img in imgs:
    proxy_img = dcmread(img)
    print(f'\nImage: {img}')
    print(f'File meta info: {proxy_img}')