import cv2
import os
import numpy as np
from app.core.features.texture import extract_lbp_features, extract_glcm_features

def run_test():
    # 1. Cấu hình đường dẫn
    IMAGE_PATH = "data/Leaves/1001.jpg" # Đảm bảo file này tồn tại
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Lỗi: Không tìm thấy ảnh tại {IMAGE_PATH}")
        return

    print(f"🚀 Bắt đầu trích xuất đặc trưng cho: {IMAGE_PATH}")
    print("-" * 50)

    try:
        # 2. Trích xuất đặc trưng Texture
        # LBP (Local Binary Patterns) - Đại diện cho độ nhám
        lbp_vec = extract_lbp_features(IMAGE_PATH)
        
        # GLCM (Gray-Level Co-occurrence Matrix) - Đại diện cho vân lá
        glcm_vec = extract_glcm_features(IMAGE_PATH)

        # 3. Tổng hợp đặc trưng (Feature Fusion/Concatenation)
        # Đây là bước quan trọng nhất để tạo ra Vector cuối cùng
        final_vector = np.concatenate([lbp_vec, glcm_vec])

        # 4. Hiển thị kết quả kiểm tra
        print(f"✅ Trích xuất LBP thành công. Kích thước: {len(lbp_vec)}")
        print(f"✅ Trích xuất GLCM thành công. Kích thước: {len(glcm_vec)}")
        print("-" * 50)
        print(f"🌟 VECTOR TỔNG HỢP (Final Feature Vector):")
        print(f"   - Tổng số chiều: {len(final_vector)}")
        print(f"   - Dữ liệu: {final_vector}")
        print("-" * 50)

        # 5. Gợi ý lưu vào Database
        print("💡 Bước tiếp theo: Lưu vector này vào PostgreSQL (pgvector).")

    except Exception as e:
        print(f"💥 Đã xảy ra lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    run_test()
