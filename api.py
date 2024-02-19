import numpy as np
import warnings; warnings.filterwarnings('ignore')


# Load model
from models.load_model import load_model, get_infos_from_tag
modelTag = "993"
model = load_model(modelTag, FULLCONV=True, turn_off_classifier=True)
(model_input_size, model_srate) = get_infos_from_tag(modelTag)

path = "../thesis-code/sample_data/audio/pouring_water_in_a_glass.wav"
from prediction import get_audio, sliding_norm, predict_fullConv
audio = get_audio(path, model_input_size, model_srate)
# normalize audio :
# Since input size is not fixed, use a sliding window for normalizing each sample
audio = sliding_norm(audio, frame_sizes=model_input_size)
audio = np.reshape(audio, (len(audio), 1, 1))
audio = np.array([audio])

activations = model.predict(audio, verbose=0)[0, :, 0, :]
print(activations.shape)
# import ipdb; ipdb.set_trace()


# (timeVec, frequencies, confidence, activations) = predict_fullConv(
#     model, audio, viterbi=True, model_srate=model_srate,
# )
# import ipdb; ipdb.set_trace()