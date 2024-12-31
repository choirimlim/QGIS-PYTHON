import cv2
import numpy as np
from osgeo import gdal
from qgis.core import QgsProject, QgsRasterLayer
from qgis.utils import iface

# 1. 이미지 매칭을 통해 GCP 추출
def extract_gcp_from_images(image1_path, image2_path):
    # 이미지를 불러옵니다.
    image1 = cv2.imread(image1_path, 0)  # 기준 이미지
    image2 = cv2.imread(image2_path, 0)  # 대상 이미지

    # ORB 방법을 사용하여 특징점을 찾습니다.
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # FLANN 기반 매처를 사용하여 특징점 매칭을 찾습니다.
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.match(des1, des2)

    # 매칭된 특징점들의 좌표를 추출하여 GCP로 활용할 수 있습니다.
    gcp_points = []
    for match in matches[:10]:  # 상위 10개의 매칭 점을 사용
        gcp_points.append(kp2[match.trainIdx].pt)  # 대상 이미지에서의 GCP 점들

    return gcp_points

# 2. GDAL을 사용하여 지오레퍼런싱 수행
def georeference_image(input_raster_path, output_raster_path, gcp_points):
    # GCP 점을 (픽셀X, 픽셀Y, 지리적 X, 지리적 Y) 형식으로 변환
    gcp_coords = [(pt[0], pt[1], 0, 0) for pt in gcp_points]  # 예시 좌표 (지리적 좌표는 0, 0으로 설정)

    # GDAL을 사용해 이미지의 지오레퍼런싱을 수행합니다.
    gdal.Translate(output_raster_path, input_raster_path, outputSRS='EPSG:4326', GCPs=gcp_coords)

# 3. QGIS에 결과 레이어 추가
def add_georeferenced_layer_to_qgis(output_raster_path):
    # 지오레퍼런싱된 이미지를 QGIS에 추가
    raster_layer = QgsRasterLayer(output_raster_path, 'Georeferenced Raster')
    if raster_layer.isValid():
        QgsProject.instance().addMapLayer(raster_layer)
        iface.mapCanvas().refresh()
    else:
        print("Failed to add layer")

# 4. 자동화된 지오레퍼런싱 전체 과정 실행
def automate_georeferencing(image1_path, image2_path, output_raster_path):
    # GCP 추출
    gcp_points = extract_gcp_from_images(image1_path, image2_path)

    # 지오레퍼런싱 수행
    georeference_image(image1_path, output_raster_path, gcp_points)

    # QGIS에 레이어 추가
    add_georeferenced_layer_to_qgis(output_raster_path)

# 5. 실행 예시
if __name__ == "__main__":
    input_raster_path = '/path/to/your/image1.tif'  # 기준 이미지 경로
    reference_image_path = '/path/to/your/reference_image.tif'  # 참조 이미지 경로
    output_raster_path = '/path/to/output/georeferenced_image.tif'  # 출력 이미지 경로

    # 전체 자동화 과정 실행
    automate_georeferencing(input_raster_path, reference_image_path, output_raster_path)
