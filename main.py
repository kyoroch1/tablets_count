from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Streamlitアプリの設定
st.title("錠剤カウントアプリ")
st.write("任意の画像中の錠剤の数をカウントします")

# 画像のアップロード
uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 物体検出の実行
    model = YOLO('best_231224.pt')
    results = model.predict(image,conf=0.5)

    # 物体検出結果の表示
    for r in results:
      im_array = r.plot(line_width=3,labels=False)  # plot a BGR numpy array of predictions
      im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    #物体検出後の画像を表示
    st.image(im, caption="検出結果",use_column_width=True)
    count_number = len(results[0])
    st.write("検出した物体の数は",count_number,"個です。")
else :
    pass