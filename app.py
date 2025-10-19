import streamlit as st
import pandas as pd
import joblib

# Tải mô hình đã được huấn luyện
# Tệp model.joblib phải nằm cùng thư mục với app.py
try:
    model = joblib.load('model.joblib')
    # print("Tải mô hình thành công") # Dùng để gỡ lỗi
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy tệp 'model.joblib'. Hãy chắc chắn nó ở cùng thư mục.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()


# --- Định nghĩa các tùy chọn cho các trường phân loại ---
# Lấy từ phân tích dữ liệu trước đó
gender_options = ['Female', 'Male']
academic_level_options = ['Undergraduate', 'Graduate', 'High School']
platform_options = [
    'Instagram', 'Twitter', 'TikTok', 'YouTube', 'Facebook', 'LinkedIn', 
    'Snapchat', 'WeChat', 'Reddit', 'Pinterest'
]


# --- Giao diện người dùng Streamlit ---

st.set_page_config(page_title="Dự đoán Điểm Nghiện MXH", layout="centered")
st.title('👩‍💻 Dự đoán Điểm Nghiện Mạng Xã Hội')
st.write("Demo mô hình Cây Quyết Định (max_depth=3) để dự đoán điểm 'Addicted_Score' của sinh viên.")

# Tạo các cột để bố cục đẹp hơn
col1, col2 = st.columns(2)

with col1:
    # --- Các đặc trưng số ---
    conflicts = st.slider(
        'Mức độ xung đột (Conflicts_Over_Social_Media)', 
        min_value=0, max_value=5, value=2, 
        help="Mức độ bạn gặp xung đột (cãi vã, bất đồng) vì MXH (0-5)."
    )
    
    usage_hours = st.slider(
        'Giờ dùng TB (Avg_Daily_Usage_Hours)', 
        min_value=1.0, max_value=10.0, value=4.5, step=0.1,
        help="Số giờ trung bình bạn dùng MXH mỗi ngày."
    )
    
    mental_health = st.slider(
        'Điểm SK Tinh thần (Mental_Health_Score)', 
        min_value=1, max_value=10, value=5,
        help="Bạn tự đánh giá sức khỏe tinh thần của mình (1-10)."
    )
    
    sleep_hours = st.slider(
        'Giờ ngủ (Sleep_Hours_Per_Night)', 
        min_value=4.0, max_value=9.0, value=6.5, step=0.1,
        help="Số giờ trung bình bạn ngủ mỗi đêm."
    )

with col2:
    # --- Các đặc trưng phân loại ---
    gender = st.selectbox('Giới tính (Gender)', options=gender_options)
    
    academic_level = st.selectbox('Trình độ (Academic_Level)', options=academic_level_options)
    
    platform = st.selectbox('Nền tảng chính (Most_Used_Platform)', options=platform_options)


# --- Nút Dự đoán ---
if st.button('🚀 Dự đoán Điểm Nghiện'):
    # 1. Tạo DataFrame từ dữ liệu đầu vào
    input_data = {
        'Gender': [gender],
        'Academic_Level': [academic_level],
        'Mental_Health_Score': [mental_health],
        'Avg_Daily_Usage_Hours': [usage_hours],
        'Most_Used_Platform': [platform],
        'Sleep_Hours_Per_Night': [sleep_hours],
        'Conflicts_Over_Social_Media': [conflicts]
    }
    
    # Đảm bảo đúng thứ tự cột như khi huấn luyện
    # (Mặc dù pipeline của chúng ta có thể xử lý việc này, cẩn thận vẫn hơn)
    feature_order = [
        'Gender', 'Academic_Level', 'Mental_Health_Score', 
        'Avg_Daily_Usage_Hours', 'Most_Used_Platform', 
        'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media'
    ]
    
    input_df = pd.DataFrame(input_data)[feature_order]

    # 2. Gọi mô hình để dự đoán
    try:
        prediction = model.predict(input_df)
        predicted_score = round(prediction[0], 2) # Lấy kết quả và làm tròn
        
        st.success(f"### Điểm Nghiện Dự Đoán: {predicted_score:.2f} / 10")
        
        # Thêm diễn giải
        if predicted_score > 7:
            st.warning("Cảnh báo: Mức độ có dấu hiệu nghiện cao.")
        elif predicted_score > 4:
            st.info("Thông báo: Mức độ sử dụng ở mức trung bình đến khá.")
        else:
            st.success("Thông báo: Mức độ sử dụng và kiểm soát tốt.")
            
    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
