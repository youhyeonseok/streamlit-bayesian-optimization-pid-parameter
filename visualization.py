import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def visualizeResults(pred,y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(range(len(pred)), pred, label='Predict', color='b', marker='o')
    ax.scatter(range(len(y_test)), y_test, label='True Label', color='r', marker='x')
    ax.legend()
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Predict | True Label')

    # Streamlit에 그래프를 표시
    st.pyplot(fig)

    print('RMSE: {0:.4f} (s)'.format(mean_squared_error(y_test,pred)**0.5))
    print("MAPE: {0:.4f} * 100%".format(mean_absolute_percentage_error(y_test, pred)))

def visualizeTestData():
    trained_model = []
    if st.session_state["RandomForest"] != None:
        trained_model.append("RandomForest")
    if st.session_state["MLPsklearn"] != None:
        trained_model.append("MLPsklearn")
    if st.session_state["MLPkeras"] != None:
        trained_model.append("MLPkeras")
    model_name = st.selectbox("테스트 학습모델 선택", set(trained_model))

    b1 = st.button("테스트 실행")
    if b1:
        with st.spinner('테스트 진행중...'):
            x_test = st.session_state["x_test"]
            y_test = st.session_state["y_test"]

            model = st.session_state[model_name]
            pred = model.predict(x_test)
            visualizeResults(pred,y_test)
        st.toast('테스트 완료', icon='😍')