import sys
print(sys.path)



import time
import io
import gradio as gr
import cv2
import base64
import openai
import os

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from PIL import Image

from prompts import VISION_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FINAL_EVALUATION_PROMPT
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


global_dict = {}

######
# SETTINGS
VIDEO_FRAME_LIMIT = 2000

######

def validate_api_key():
    #client = openai.api_key

    try:
        # Make your OpenAI API request here
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello world"},
            ]
        )
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        response = None
        error = e
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        response = None
        error = e
        pass
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        response = None
        error = e
        pass

    if response:
        return True
    else:
        raise gr.Error(f"OpenAI returned an API Error: {error}")


def _process_video(video_file):
    # Read and process the video file
    video = cv2.VideoCapture(video_file.name)

    if 'video_file' not in global_dict:
        global_dict.setdefault('video_file', video_file.name)
    else:
        global_dict['video_file'] = video_file.name

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    if len(base64Frames) > VIDEO_FRAME_LIMIT:
        raise gr.Warning(f"Video's play time is too long. (>1m)")
    print(len(base64Frames), "frames read.")

    if not base64Frames:
        raise gr.Error(f"Cannot open the video.")
    return base64Frames


def _make_video_batch(video_file):

    frames = _process_video(video_file)

    TOTAL_FRAME_COUNT = len(frames)
    BATCH_SIZE = int(1)
    TOTAL_BATCH_SIZE = int(TOTAL_FRAME_COUNT * 5 / 100)  # 5 = total_batch_percent
    BATCH_STEP = int(TOTAL_FRAME_COUNT / TOTAL_BATCH_SIZE)
    
    base64FramesBatch = []

    for idx in range(0, TOTAL_FRAME_COUNT, BATCH_STEP * BATCH_SIZE):
        # print(f'## {idx}')
        temp = []
        for i in range(BATCH_SIZE):
            # print(f'# {idx + BATCH_STEP * i}')
            if (idx + BATCH_STEP * i) < TOTAL_FRAME_COUNT:
                temp.append(frames[idx + BATCH_STEP * i])
            else:
                continue
        base64FramesBatch.append(temp)
    
    for idx, batch in enumerate(base64FramesBatch):
        # assert len(batch) <= BATCH_SIZE
        print(f'##{idx} - batch_size: {len(batch)}')

    if 'batched_frames' not in global_dict:
        global_dict.setdefault('batched_frames', base64FramesBatch)
    else:
        global_dict['batched_frames'] = base64FramesBatch

    return base64FramesBatch


def show_batches(video_file):
    
    batched_frames = _make_video_batch(video_file) 
    
    images1 = []
    images2 = []
    for i, l in enumerate(batched_frames):
        print(f"#### Batch_{i+1}")
        for j, img in enumerate(l):
            print(f'## Image_{j+1}')
            image_bytes = base64.b64decode(img.encode("utf-8"))
            # Convert the bytes to a stream (file-like object)
            image_stream = io.BytesIO(image_bytes)
            # Open the image as a PIL image
            image = Image.open(image_stream)
            images1.append((image, f"batch {i+1}"))
            images2.append((image, f"batch {i+1}"))
        print("-"*100)
    
    return images1, images2




def call_gpt_vision(rubrics, progress=gr.Progress()) -> list:
    frames = global_dict.get('batched_frames')
    openai.api_key = OPENAI_API_KEY

    full_result_vision = []
    full_text_vision = ""
    idx = 0

    for batch in progress.tqdm(frames):
        VISION_PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(rubrics=rubrics),
                    *map(lambda x: {"image": x, "resize": 300}, batch),
                ],
            },
        ]
        
        params = {
        "model": "gpt-4-vision-preview",
        "messages": VISION_PROMPT_MESSAGES,
        "max_tokens": 1024,
        }

        try:
            result = openai.chat.completions.create(**params)
            print(result.choices[0].message.content)
            full_result_vision.append(result)
        except Exception as e:
            print(f"Error: {e}")
            full_text_vision += f'### BATCH_{idx+1}\n' + "-"*50 + "\n" + f"Error: {e}" +  "\n" + "-"*50 + "\n"
            idx += 1
            pass
        
        if 'full_result_vision' not in global_dict:
            global_dict.setdefault('full_result_vision', full_result_vision)
        else:
            global_dict['full_result_vision'] = full_result_vision
        
    return full_text_vision



def get_full_result():
    full_result_vision = global_dict.get('full_result_vision')
    full_result_audio = global_dict.get('full_text_audio')
    
    result_text_video = ""
    result_text_audio = ""


    for idx, res in enumerate(full_result_vision):
        result_text_video += f'<Video Evaluation_{idx+1}>\n'
        result_text_video += res.choices[0].message.content
        result_text_video += "\n"
        result_text_video += "-"*5
        result_text_video += "\n"
    result_text_video += "*"*5 + "END of Video" + "*"*5 

    if full_result_audio:
        result_text_audio += '<Audio Evaluation>\n'
        result_text_audio += full_result_audio
        result_text_audio += "\n"
        result_text_audio += "-"*5
        result_text_audio += "\n"
        result_text_audio += "*"*5 + "END of Audio" + "*"*5 

        result_text = result_text_video + "\n\n" + result_text_audio
    else:
        result_text = result_text_video
    
    if 'result_text' not in global_dict:
            global_dict.setdefault('result_text', result_text)
    else:
        global_dict['result_text'] = result_text
        

    return result_text


def get_final_anser():
    chain = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4",
        max_tokens=1024,
        temperature=0,
    )
    prompt = PromptTemplate.from_template(FINAL_EVALUATION_PROMPT)
    runnable = prompt | chain | StrOutputParser()
    result_text = global_dict.get('result_text')

    final_eval = runnable.invoke({"evals": result_text})
    return final_eval

"""
# Define the Gradio app
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# GPT-4 Vision for Evaluation")
        gr.Markdown("## Make Batched Snapshots")
        with gr.Row():
            with gr.Column(scale=1):
                video_upload = gr.File(
                    label="Upload your video (video under 1 minute is the best..!)",
                    file_types=["video"],
                )
                
                # case1 - 영상의 길이를 지정해주기 (15초, 30초 내외... - 너무 짧은 영상을 넣으면, 이 값을 고정했을 때 스냅샷이 잘 안 뽑힐 수 있음)
                # case2 - 영상의 길이를 지정하지 않기 (어차피 1분 이내밖에 안됨)
                total_batch_percent = gr.Slider(
                    label="How many snapshots do you wnat to take for the evaluation? Shorter videos need more snapshots.",
                    info="Choose between 1(less) and 100(more)",
                    value=3,
                    minimum=1,
                    maximum=100,
                    step=1
                )
                process_button = gr.Button("Process")         
                gallery1 = gr.Gallery(
                    label="Batched Snapshots of Video",
                    columns=[3],
                    rows=[10],
                    object_fit="contain",
                    height="auto",
                )
            
        gr.Markdown("## Set Evaluation Rubric")
        with gr.Row():
             with gr.Column(scale=1):
                rubric_video_input = gr.Textbox(
                    label="2. Video Evaluation Rubric",
                    info="Enter your evaluation rubric here...",
                    placeholder="Here's what the performer should *SHOW* as follows:\n1. From standing, bend your knees and straighten your arms in front of you.\n2. Place your hands on the floor, shoulder width apart with fingers pointing forward and your chin on your chest.\n3. Rock forward, straighten legs and transfer body weight onto shoulders.\n4. Rock forward on a rounded back placing both feet on the floor.\n5. Stand using arms for balance, without hands touching the floor.",
                    lines=7
                )
                evaluate_button = gr.Button("Evaluate")
        with gr.Row():
             with gr.Column(scale=1):
                gallery2 = gr.Gallery(
                    label="Batched Snapshots of Video",
                    columns=[3],
                    rows=[10],
                    object_fit="contain",
                    height="auto",
                )
             with gr.Column(scale=1):
                video_output_box = gr.Textbox(
                    label="Video Batched Snapshots Eval...",
                    lines=8,
                    interactive=False
                )

        gr.Markdown("## Get Summarized Result")
        with gr.Row():
             with gr.Column(scale=1):
                output_box_fin_fin = gr.Textbox(
                    label="Final Evaluation",
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )

       
        process_button.click(fn=show_batches, inputs=[video_upload], outputs=[gallery1, gallery2])
        evaluate_button.click(fn=call_gpt_vision, inputs=[rubric_video_input], outputs=video_output_box).success(fn=get_full_result, inputs=[], outputs=[]).success(fn=get_final_anser, inputs=[video_output_box], outputs=output_box_fin_fin)
        ### then -> success

    demo.launch()"""


def mainpage():
    with gr.Blocks() as start_page:
        gr.Markdown("M-WAVE: Model Walk Analysis and Virtual Evaluation")
        gr.Button("start").click(video_rubric)
    start_page.launch()

def video_rubric():
    with gr.Blocks() as video_rubric_page:
        gr.Markdown("비디오 업로드 페이지")
        with gr.Row():
            with gr.Column(scale=1):

                rubric_video_input = gr.Textbox(
                    label="2. Video Evaluation Rubric",
                    info="Enter your evaluation rubric here...",
                    placeholder="Here's what the performer should *SHOW* as follows:\n1. From standing, bend your knees and straighten your arms in front of you.\n2. Place your hands on the floor, shoulder width apart with fingers pointing forward and your chin on your chest.\n3. Rock forward, straighten legs and transfer body weight onto shoulders.\n4. Rock forward on a rounded back placing both feet on the floor.\n5. Stand using arms for balance, without hands touching the floor.",
                    lines=7
                )

                video_upload = gr.File(
                    label="Upload your video (video under 1 minute is the best..!)",
                    file_types=["video"],
                )
                process_button = gr.Button("Process")
        process_button.click(fn=_make_video_batch, inputs=[video_upload], outputs=None).success(fn=call_gpt_vision, inputs=[rubric_video_input], outputs=[]).success(fn=get_full_result, inputs=[], outputs=[]).success(show_result)  #output=None? []?

    video_rubric_page.launch()

def show_result():
    with gr.Blocks() as show_result_page:
        gr.Markdown("결과 페이지")
        with gr.Row():
            with gr.Column(scale=1):
                output_box_fin_fin = gr.Textbox(
                    label="Final Evaluation",
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )
        gr.Timer(interval=0.1, repeat=False).call(fn=get_final_anser, inputs=[], outputs=[output_box_fin_fin])

    show_result_page.launch()


if __name__ == "__main__":
    mainpage()


