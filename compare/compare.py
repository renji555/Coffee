from blender import Blender

blender = Blender()
blender.loadranker("llm-blender/PairRM")

# inputs是多个问题
# candidates_texts是每个问题不同模型生成的不同答案
# ranks是不同模型生成结果的排序
inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [["get out!", "hi! I am fine, thanks!", "bye!"], 
                    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=1)
print(ranks)