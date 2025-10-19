import streamlit as st
import pandas as pd
import joblib

# --- Xây dựng giao diện Streamlit ---
st.title("📈 Dự đoán Điểm Nghiện Mạng Xã hội của Sinh viên")

st.write("""
Nhập vào các thông tin dưới đây để mô hình cây quyết định (độ sâu=3) 
dự đoán điểm nghiện (Addicted_Score) của sinh viên.
""")

# Tạo các cột để bố cục đẹp hơn
col1, col2 = st.columns(2)

with col1:
    # --- Nhập liệu cho 6 đặc trưng ---
    gender = st.selectbox(
        'Giới tính (Gender)',
        ('Female', 'Male', 'Others') # Cung cấp các lựa chọn
    )

    academic_level = st.selectbox(
        'Cấp bậc học vấn (Academic_Level)',
        ('Undergraduate', 'Graduate', 'High School') # Dựa trên dữ liệu mẫu
    )

    platform = st.selectbox(
        'Nền tảng dùng nhiều nhất (Most_Used_Platform)',
        ('Instagram', 'TikTok', 'Facebook', 'Twitter', 'YouTube', 'Snapchat', 'LinkedIn', 'WeChat')
    )

with col2:
    mental_score = st.slider(
        'Điểm sức khỏe tinh thần (Mental_Health_Score)',
        min_value=1, max_value=10, value=5, step=1 # Thang điểm 1-10
    )

    usage_hours = st.slider(
        'Giờ dùng trung bình mỗi ngày (Avg_Daily_Usage_Hours)',
        min_value=0.0, max_value=12.0, value=4.0, step=0.5 # Giả định tối đa 12h
    )

    sleep_hours = st.slider(
        'Giờ ngủ mỗi đêm (Sleep_Hours_Per_Night)',
        min_value=3.0, max_value=12.0, value=7.0, step=0.5 # Giả định
    )

# --- Nút dự đoán ---
if st.button('🚀 Dự đoán Điểm Nghiện'):
    try:
        # Tạo một DataFrame từ dữ liệu nhập vào
        # Cấu trúc phải khớp với 6 đặc trưng đã huấn luyện
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Academic_Level': [academic_level],
            'Mental_Health_Score': [mental_score],
            'Avg_Daily_Usage_Hours': [usage_hours],
            'Most_Used_Platform': [platform],
            'Sleep_Hours_Per_Night': [sleep_hours]
        })
        
        #st.write("Dữ liệu đầu vào:")
        #st.dataframe(input_data)

        # Sử dụng pipeline đã tải để dự đoán
        prediction = pipeline.predict(input_data)
        
        # Hiển thị kết quả
        st.success(f"**Điểm Nghiện (Addicted_Score) dự đoán là: {prediction[0]:.2f}**")
        
        st.info("Lưu ý: Đây là dự đoán từ mô hình DecisionTreeRegressor với độ sâu=3.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")

