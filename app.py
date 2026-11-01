import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os

st.set_page_config(
    page_title="WAFAA-AI",
    page_icon="💜",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f5edff, #fff8ff);
}
h1 {
    color: #7045a5;
    text-align: center;
}
.subtitle {
    color: #806a98;
    text-align: center;
}
.stButton button {
    background-color: #8055b5;
    color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 👶🏻 WAFAA-AI")
st.markdown(
    "<p class='subtitle'>Because Every Cry Has a Meaning 💜</p>",
    unsafe_allow_html=True
)

st.write("### تحليل صوت بكاء الطفل")
st.write("ارفعي تسجيلًا صوتيًا لبدء التحليل.")

MODEL_PATH = "WAFAA_AI_V1.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("تم تحميل نموذج الذكاء الاصطناعي")
except Exception as e:
    st.error(f"تعذر تحميل النموذج: {e}")
    st.stop()

audio_file = st.file_uploader(
    "اختاري ملفًا صوتيًا",
    type=["wav", "mp3", "m4a", "ogg"]
)

if audio_file:
    st.audio(audio_file)

    if st.button("💜 تحليل التسجيل"):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".audio", delete=False
            ) as temp:
                temp.write(audio_file.getvalue())
                temp_path = temp.name

            y, sr = librosa.load(
                temp_path,
                sr=16000,
                mono=True
            )

            if len(y) < sr:
                y = np.pad(y, (0, sr - len(y)))

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=13
            )

            features = (
                list(np.mean(mfcc, axis=1))
                + list(np.std(mfcc, axis=1))
            )

            features += [
                float(np.mean(
                    librosa.feature.zero_crossing_rate(y)
                )),
                float(np.mean(
                    librosa.feature.rms(y=y)
                )),
                float(np.mean(
                    librosa.feature.spectral_centroid(
                        y=y, sr=sr
                    )
                ))
            ]

            x = np.array(features).reshape(1, -1)

            prediction = model.predict(x)[0]

            st.markdown("### 💜 نتيجة التحليل")
            st.write(f"**التصنيف:** {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x)[0]

                for label, probability in sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):
                    st.write(f"{label}: {probability * 100:.1f}%")

        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()

st.caption(
    "WAFAA-AI نموذج بحثي تجريبي، وليس جهازًا طبيًا أو وسيلة لتشخيص حالة الطفل."
  )
