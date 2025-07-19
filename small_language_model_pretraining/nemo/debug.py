from nemo.collections.common.tokenizers import AutoTokenizer
from pprint import pprint
tokenizer = AutoTokenizer(pretrained_model_name="/data/llama3_8b_ref/model/Llama-3.1-8B")
print (f'tokenizer: {tokenizer}')
print(dir(tokenizer))


print (tokenizer.vocab_size)
# pprint(vars(tokenizer))

