import streamlit as st
import numpy as np
import wave
import io
import tensorflow as tf
from tensorflow import keras

# Thông số
frame_length = 256
frame_step = 160
fft_length = 384
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


loaded_model = keras.models.load_model(r'C:\Users\SpringT\Dropbox\Tài liệu ptit\KÌ 6\AI\BTL\streamlit-stt-app\model\model.h5')

print("loaded model")

# Hàm tiền xử lý một mẫu âm thanh duy nhất
def encode_single_sample(audio_bytes):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Decode the wav file
    audio, _ = tf.audio.decode_wav(audio_bytes)
    audio = tf.squeeze(audio, axis=-1)
    # 2. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 3. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 4. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 5. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram

def decode_batch_predictions(pred):
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def predict_from_audio(audio_bytes):
    # Tiền xử lý âm thanh
    spectrogram = encode_single_sample(audio_bytes)

    # Thêm trục batch để phù hợp với đầu vào của mô hình
    spectrogram = tf.expand_dims(spectrogram, axis=0)

    # Giả sử bạn có mô hình đã huấn luyện, thay thế `model` bằng mô hình thực tế
    # predictions = model.predict(spectrogram)

    # Ở đây, chúng ta sẽ giả sử `predictions` là đầu ra của mô hình
    predictions = np.random.rand(1, 100, 29)  

    # Giải mã dự đoán thành văn bản
    transcription = decode_batch_predictions(predictions)
    return transcription[0]

st.title("Chuyển đổi Âm thanh sang Văn bản")

st.write("Tải lên tệp âm thanh của bạn và chúng tôi sẽ chuyển đổi nó sang văn bản.")

uploaded_file = st.file_uploader("Chọn tệp âm thanh", type=["wav", "mp3"])

if uploaded_file is not None:
    # Đọc tệp âm thanh
    audio_bytes = uploaded_file.read()

    # Lưu tệp âm thanh tạm thời
    audio_file = io.BytesIO(audio_bytes)
    
    # Đọc tệp âm thanh bằng wave (nếu là tệp wav)
    try:
        with wave.open(audio_file, 'rb') as wf:
            st.write(f"Channels: {wf.getnchannels()}")
            st.write(f"Sample width: {wf.getsampwidth()}")
            st.write(f"Frame rate: {wf.getframerate()}")
            st.write(f"Number of frames: {wf.getnframes()}")
    except wave.Error as e:
        st.error(f"Lỗi khi đọc tệp âm thanh: {e}")
    
    # Hiển thị trình phát âm thanh
    st.audio(audio_bytes, format='audio/wav')

    # Dự đoán từ tệp âm thanh
    st.write("Đang chuyển đổi âm thanh sang văn bản...")
    transcription = predict_from_audio(audio_bytes)
    
    # Hiển thị kết quả
    st.write("**Kết quả:**")
    st.text(transcription)
