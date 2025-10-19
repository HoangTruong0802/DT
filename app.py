import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
# Tên tệp dữ liệu (phải nằm chung thư mục với app.py)
DATA_FILE = "Students Social Media Addiction.csv"

# --- Hàm Huấn luyện Mô hình ---
# Sử dụng @st.cache_resource để huấn luyện mô hình 1 LẦN DUY NHẤT
# và lưu lại (cache) để dùng cho mọi người dùng
@st.cache_resource
def get_model(file_path):
    """
    Hàm này tải dữ liệu, tiền xử lý và huấn luyện mô hình.
    Nó chỉ chạy 1 lần duy nhất khi app khởi động.
    """
    # 1. Tải dữ liệu
    df = pd.read_csv(file_path)

    # 2. Xác định đặc trưng (6 cột) và mục tiêu
    target_column = 'Addicted_Score'
    features = [
        'Gender',
        'Academic_Level',
        'Mental_Health_Score',
        'Avg_Daily_Usage_Hours',
        'Most_Used_Platform',
        'Sleep_Hours_Per_Night'
    ]

    X_all = df[features]
    y_all = df[target_column]

    numerical_features = [
        'Mental_Health_Score',
        'Avg_Daily_Usage_Hours',
        'Sleep_Hours_Per_Night'
    ]
    categorical_features = [
        'Gender',
        'Academic_Level',
        'Most_Used_Platform'
    ]

    # 3. Tạo bộ tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 4. Tạo Pipeline (Bao gồm Tiền xử lý + Mô hình Hồi quy)
    model = DecisionTreeRegressor(max_depth=3,random_state=42,min_samples_leaf = 2)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # 5. Huấn luyện mô hình trên 100% dữ liệu
    pipeline.fit(X_all, y_all)

    # 6. Trả về các giá trị duy nhất để dùng cho selectbox
    unique_levels = df['Academic_Level'].unique()
    unique_platforms = df['Most_Used_Platform'].unique()

    return pipeline, unique_levels, unique_platforms

# --- Tải mô hình ---
# Lời gọi hàm này sẽ được cache lại
try:
    pipeline, unique_levels, unique_platforms = get_model(DATA_FILE)
    model_loaded = True
except FileNotFoundError:
    st.error(f"Lỗi: Không tìm thấy tệp dữ liệu '{DATA_FILE}'.")
    st.error("Vui lòng đảm bảo tệp CSV nằm cùng thư mục với tệp app.py.")
    model_loaded = False
except Exception as e:
    st.error(f"Lỗi khi tải hoặc huấn luyện mô hình: {e}")
    model_loaded = False


# --- BẮT ĐẦU XÂY DỰNG GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Dự đoán Nghiện MXH", layout="wide") # Đặt tiêu đề trang
st.title("🤖 Demo Mô hình Dự đoán Điểm Nghiện Mạng Xã Hội")
st.write("Nhập thông tin của sinh viên vào thanh bên trái để mô hình dự đoán điểm nghiện (`Addicted_Score`).")
st.write("---")

# Chỉ hiển thị giao diện nhập liệu nếu model đã tải thành công
if model_loaded:
    # --- Thanh bên (Sidebar) để nhập liệu ---
    st.sidebar.header("Nhập thông tin sinh viên:")

    # 1. Giới tính
    gender = st.sidebar.selectbox(
        "Giới tính (Gender):",
        ['Female', 'Male'] # Giả sử chỉ có 2 giá trị này
    )

    # 2. Trình độ học vấn
    academic_level = st.sidebar.selectbox(
        "Trình độ học vấn (Academic_Level):",
        unique_levels # Lấy từ dữ liệu gốc
    )

    # 3. Nền tảng sử dụng nhiều nhất
    most_used_platform = st.sidebar.selectbox(
        "Nền tảng hay dùng (Most_Used_Platform):",
        unique_platforms # Lấy từ dữ liệu gốc
    )

    # 4. Sức khỏe tinh thần (thanh trượt)
    mental_health = st.sidebar.slider(
        "Điểm Sức khỏe tinh thần (1-10):",
        min_value=1, max_value=10, value=7, step=1 # Giá trị mặc định là 7
    )

    # 5. Giờ sử dụng trung bình (thanh trượt)
    usage_hours = st.sidebar.slider(
        "Giờ dùng trung bình/ngày:",
        min_value=0.0, max_value=12.0, value=4.0, step=0.1 # Mặc định 4.0 giờ
    )

    # 6. Giờ ngủ (thanh trượt)
    sleep_hours = st.sidebar.slider(
        "Giờ ngủ/đêm:",
        min_value=4.0, max_value=10.0, value=7.0, step=0.1 # Mặc định 7.0 giờ
    )

    # --- Nút dự đoán ---
    if st.sidebar.button("Nhấn để Dự đoán"):

        # 1. Tạo DataFrame từ dữ liệu nhập vào
        # DataFrame này phải có tên cột Y HỆT như lúc huấn luyện
        input_data = {
            'Gender': [gender],
            'Academic_Level': [academic_level],
            'Mental_Health_Score': [mental_health],
            'Avg_Daily_Usage_Hours': [usage_hours],
            'Most_Used_Platform': [most_used_platform],
            'Sleep_Hours_Per_Night': [sleep_hours]
        }
        input_df = pd.DataFrame(input_data)

        st.subheader("Thông tin bạn đã nhập:")
        st.dataframe(input_df) # Hiển thị lại dữ liệu nhập

        # 2. Gọi pipeline để dự đoán
        # Pipeline sẽ tự động tiền xử lý (OneHotEncoder) dữ liệu này
        prediction = pipeline.predict(input_df)

        # Lấy giá trị dự đoán (là một con số)
        predicted_score = prediction[0]

        # 3. Hiển thị kết quả
        st.subheader("Kết quả Dự đoán:")

        # Sử dụng st.metric để hiển thị con số thật đẹp
        st.metric(
            label="Điểm Nghiện Dự đoán (Addicted_Score)",
            value=f"{predicted_score:.5f}", # Làm tròn 5 chữ số
        )

        # Đánh giá nhanh mức độ
        if predicted_score >= 8.0:
            st.error("🚨 Mức độ nghiện dự đoán: Rất Cao")
        elif predicted_score >= 6.0:
            st.warning("⚠️ Mức độ nghiện dự đoán: Cao")
        elif predicted_score >= 4.0:
            st.info("ℹ️ Mức độ nghiện dự đoán: Trung bình")
        else:
            st.success("✅ Mức độ nghiện dự đoán: Thấp")

    else:
        st.info("👈 Nhập thông tin ở thanh bên trái và nhấn nút 'Nhấn để Dự đoán'.")


