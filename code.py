# Install Dependencies

!pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
!pip install transformers

# Import Model
# Import the model class and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Speak to it!

utterance = "I want to order a Pizza"

# Tokenize the utterance
inputs = tokenizer(utterance, return_tensors="pt")
inputs

# Passing through the utterances to the Blenderbot model
res = model.generate(**inputs)
res

# Decoding the model output
tokenizer.decode(res[0])

# Decoding the inputs
tokenizer.decode(inputs['input_ids'][0])
