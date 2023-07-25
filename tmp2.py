from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
text = '登鹳雀楼->王之涣\n夜雨寄北->'
text = '下面这句话是疑问句还是陈述句，是疑问句的话就回答A，是陈述句就回答B，请用一个字母回答：“今天天气真不错！”，'
inputs = tokenizer(text, return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
