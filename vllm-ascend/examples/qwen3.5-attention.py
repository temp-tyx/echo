import os
import json
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from vllm.config import CompilationConfig, CUDAGraphMode


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

image_path1 = ["/cache/t00932669/aisf-dataset/generalModel/business/VLM/AISF/AIHomeHub/seatDetection/screenshot/screenshot_123050_20260401_091010571.jpg",
               "/cache/t00932669/aisf-dataset/generalModel/business/VLM/AISF/AIHomeHub/seatDetection/screenshot/screenshot_123050_20260401_091014161.jpg",               ]
image_path2 = ["/cache/t00932669/aisf-dataset/generalModel/business/VLM/AISF/AIHomeHub/seatDetection/screenshot/screenshot_123050_20260401_091012441.jpg",
               "/cache/t00932669/aisf-dataset/generalModel/business/VLM/AISF/AIHomeHub/seatDetection/screenshot/screenshot_123050_20260401_091015888.jpg"]

SYSTEM_PROMPT = """### 任务：人物学习场景注意力状态判定
**核心指令：** 你必须作为一个严格的逻辑分类器。请按以下分类标准按顺序逐条检查，判定是否属于该类别。
**如果属于，则输出是，如果不属于，输出否。**
这些照片按时间先后顺序输入，最后一张为当前帧。相邻帧时间间隔约为 1 秒。请结合多帧变化分析图中人物的状态。
### [判定流程]
划定**学习区**：桌面的书本、整个桌子，除此之外都是非学习区
物品分类：棋类->玩具，闹钟->玩具，杯子等饮品->零食
**0. 集中**
* **命中条件：** 图中人物在学习，或者视线看向学习区，或者两张图片中有一张图片看向学习区，或者没有明显不集中行为；且没有以下动作：
- a. 头部直接趴在桌面上或枕在手臂上睡觉；
- b. 手上持有手机等电子设备、玩具、零食；
- c. 两张图片都出现了头部**偏离**学习区、眼睛的视线相对于学习区有大幅度的偏移；或人物**背对**、脸**偏离**学习区。
* **排除：** 眼睛前视有可能在思考，视线没有大幅度偏移学习区，判定为集中学习。
**1. 使用电子设备**
* **命中条件：** 手持手机、平板等电子设备并把玩；或面部明显朝向电子屏幕。
* **注：** 闹钟、鼠标、智能手表不计入此类（按玩具处理）。

**2. 玩玩具**
* **命中条件：** 手持明显为玩具、闹钟、遥控器等物品。
* 注意：在办公或学习场景下，请优先辨别该物体是否为文具（如橡皮、修正带、便利贴），而非玩具。如果是文具，则不可判定为玩玩具。如果配合有进食动作，优先认为是食物（归到吃零食）
* **强制排除：** 笔、修正带、便利贴、橡皮、积木、尺子等学习用品、文具、食物、电子设备、水杯/饮料瓶。对于无法确认为玩具的物品，严禁判为玩玩具

**3. 吃零食**
* **命中条件：** 手持可明显辨认的食物，或正在使用水杯、酸奶、饮料瓶等，或对拿在手上的物品有进食动作。

**4. 东张西望**
* **命中条件：** 两张图片中人物都存在**头部或视线**大幅偏离了桌面或手中的书本或文具、或人物**背对**桌面的现象，看起来没有在认真学习。
* **排除：** 任意一张图片，如果**不满足**人物**头部或视线**大幅度偏离学习区的状态时，严禁判为"东张西望"。
请对每一帧分别判断人物视线/头部是否在学习区。
**5. 遮挡 (严格受限)**
* **命中条件：** 当身体或面部被遮挡（如书本），导致无法观察面部朝向及状态。
* **排除：** 若能看到脸、或被电子设备遮挡（归类1）、或背身（归类4），严禁判定为遮挡。
**6. 其他 (兜底不集中)**
* **命中条件：** 没有"集中学习"，没有看向学习区、但又不符合上述 1-5 任何一项具体特征。
### [输出格式]
请仅输出 JSON 格式
{"手持物品": "图中人物的手持物品/无/不确定", "物品种类": "学习用品/玩具/电子产品/食物/其他/无", "使用电子设备": true/false, "玩玩具": true/false, "吃零食": true/false, "第一帧在学习区": true/false, "第二帧在学习区": true/false, "集中": true/false, "其他": true/false, "遮挡": true/false}
"""


def main():
    MODEL_PATH = "/softwarePlatform/c00879303/Qwen3-5/Qwen3.5-4B"

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=24576,
        tensor_parallel_size=1,
        enforce_eager=False,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            # cudagraph_capture_sizes=[4,8,12,16,24,32,36,40,44,48,52,56,60,64,68,72,76,80]
        ),
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 3,
        },
    )

    sampling_params = SamplingParams(max_tokens=128,temperature=0)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    send_request(processor, llm, sampling_params, 2)



def send_request(processor, llm, sampling_params, bs):
    requests_data = []
    for i in range(bs):
        image_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path1[i],
                    },
                    {
                        "type": "image",
                        "image": image_path2[i],
                    },
                    {"type": "text", "text": "\n"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            image_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, _, _ = process_vision_info(image_messages, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        requests_data.append(llm_inputs)

    print("\n" + "=" * 60)
    outputs = llm.generate(requests_data, sampling_params=sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print("Generated text:", generated_text)



if __name__ == "__main__":
    main()