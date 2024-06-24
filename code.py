import micropip

# Install transformers using micropip
await micropip.install('transformers')

# Import the model class and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# Speak to it!
utterance = "I want to order a Pizza"

# Tokenize the utterance
inputs = tokenizer(utterance, return_tensors="pt") # pytorch tensor format

# Passing through the utterances to the Blenderbot model
res = model.generate(**inputs)

# Decoding the model output
print(tokenizer.decode(res[0]))

# Decoding the inputs
print(tokenizer.decode(inputs['input_ids'][0]))
