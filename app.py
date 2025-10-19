import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

DATA_FILE = "Students Social Media Addiction.csv"

# --- Hàm Huấn luyện Mô hình ---
@st.cache_resource
def get_model(file_path):
    """
    Hàm này tải dữ liệu, CHIA TÁCH, tiền xử lý, huấn luyện
    và CHẤM ĐIỂM mô hình.
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
    # 80% để huấn luyện (train), 20% để kiểm tra (test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
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

    # 4. Tạo Pipeline với mô hình đã "khử nhiễu" 
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=7,
        min_samples_leaf=10,
        min_samples_split=20
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # 5. Huấn luyện mô hình CHỈ trên 80% dữ liệu (tập Train) ---
    pipeline.fit(X_train, y_train)

    # 6. Chấm điểm mô hình trên 20% dữ liệu lạ (tập Test) ---
    y_pred = pipeline.predict(X_test)
    model_score = r2_score(y_test, y_pred) # Tính điểm R-squared

    # 7. Trả về các giá trị duy nhất để dùng cho selectbox
    unique_levels = df['Academic_Level'].unique()
    unique_platforms = df['Most_Used_Platform'].unique()

    # 8.  Trả về cả điểm số (score)
    return pipeline, unique_levels, unique_platforms, model_score

# --- Tải mô hình ---
try:
    pipeline, unique_levels, unique_platforms, model_score = get_model(DATA_FILE)
    model_loaded = True
except FileNotFoundError:
    st.error(f"Lỗi: Không tìm thấy tệp dữ liệu '{DATA_FILE}'.")
    st.error("Vui lòng đảm bảo tệp CSV nằm cùng thư mục với tệp app.py.")
    model_loaded = False
except Exception as e:
    st.error(f"Lỗi khi tải hoặc huấn luyện mô hình: {e}")
    model_loaded = False


# --- BẮT ĐẦU XÂY DỰNG GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Dự đoán Nghiện MXH", layout="wide")
st.title("🤖 Demo Mô hình Dự đoán Điểm Nghiện Mạng Xã Hội")
st.write("Nhập thông tin của sinh viên vào thanh bên trái để mô hình dự đoán điểm nghiện (`Addicted_Score`).")
st.write("---")

    # --- Thanh bên (Sidebar) để nhập liệu
    gender = st.sidebar.selectbox("Giới tính (Gender):", ['Female', 'Male'])
    academic_level = st.sidebar.selectbox("Trình độ học vấn (Academic_Level):", unique_levels)
    most_used_platform = st.sidebar.selectbox("Nền tảng hay dùng (Most_Used_Platform):", unique_platforms)
    mental_health = st.sidebar.slider("Điểm Sức khỏe tinh thần (1-10):", 1, 10, 7, 1)
    usage_hours = st.sidebar.slider("Giờ dùng trung bình/ngày:", 0.0, 12.0, 4.0, 0.1)
    sleep_hours = st.sidebar.slider("Giờ ngủ/đêm:", 4.0, 10.0, 7.0, 0.1)

    # --- Nút dự đoán
    if st.sidebar.button("Nhấn để Dự đoán"):
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
        st.dataframe(input_df)

        prediction = pipeline.predict(input_df)
        predicted_score = prediction[0]

        st.subheader("Kết quả Dự đoán:")
        st.metric(
            label="Điểm Nghiện Dự đoán (Addicted_Score)",
            value=f"{predicted_score:.5f}", # (Vẫn giữ 5 chữ số)
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




